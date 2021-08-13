import numpy as np
import pybnesian as pbn
import os
from sklearn.model_selection import KFold
import multiprocessing as mp
import pathlib
import glob

SEED = 0
EVALUATION_FOLDS = 10
PARALLEL_THREADS = 10
PATIENCE = [0, 5]

import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()

class PluginEstimator(pbn.BandwidthSelector):
    
    def __init__(self):
        pbn.BandwidthSelector.__init__(self)
        self.ks = importr("ks")
        # self.R = robjects.r

    def bandwidth(self, df, variables):
        data = df.to_pandas().loc[:, variables].dropna().to_numpy()

        if data.shape[0] <= len(variables):
            raise pbn.SingularCovarianceData("[instances] The data covariance could not be estimated because the matrix is singular.")

        cov = np.cov(data, rowvar=False)
        if np.linalg.matrix_rank(cov) < len(variables):
            raise pbn.SingularCovarianceData("[rank] The data covariance could not be estimated because the matrix is singular.")

        try:
            if len(variables) == 1:
                return np.asarray([self.ks.hpi(data)])
            else:
                return self.ks.Hpi(data)
        except rpy2.rinterface_lib.embedded.RRuntimeError as rerror:
            if "scale estimate is zero for input data" in str(rerror):
                raise pbn.SingularCovarianceData("[scalest 1d] The data covariance could not be estimated because the matrix is singular.")
            else:
                raise rerror

def remove_crossvalidated_nan(dataset, folds):
    to_delete = set()

    # Outer for: Performance CV
    for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(EVALUATION_FOLDS, shuffle=True, random_state=SEED).split(dataset)):
        train_data = dataset.iloc[train_indices,:]
        # Inner for: Validation CV
        for k in folds:
            vl = pbn.ValidatedLikelihood(train_data, k=k, seed=SEED)
            for (train_fold, _) in vl.cv_lik.cv:
                train_fold_pandas = train_fold.to_pandas().select_dtypes('float64')
                d = train_fold_pandas.columns[np.isclose(train_fold_pandas.var(), 0)].tolist()
                to_delete.update(d)

    return to_delete

def linear_dependent_features(dataset):
    to_delete = set()

    # Outer for: Performance CV
    for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(EVALUATION_FOLDS, shuffle=True, random_state=SEED).split(dataset)):
        train_data = dataset.iloc[train_indices,:]
        # Inner for: Validation CV
        vl = pbn.ValidatedLikelihood(train_data, seed=SEED)
        for (train_fold, _) in vl.cv_lik.cv:
            train_fold_pandas = train_fold.to_pandas().select_dtypes('float64')
            train_fold_pandas = train_fold_pandas.drop(to_delete, axis=1)

            rank = np.linalg.matrix_rank(train_fold_pandas.cov())

            if rank < train_fold_pandas.shape[1]:

                for c in train_fold_pandas:
                    new_df = train_fold_pandas.drop(c, axis=1)
                    new_rank = np.linalg.matrix_rank(new_df.cov())

                    if rank == new_rank:
                        to_delete.add(c)

    return to_delete


def preprocess_dataframe(df):
    index_constant = np.where(df.nunique() == 1)[0]
    constant_columns = [df.columns[i] for i in index_constant]
    df = df.drop(constant_columns, axis=1)

    df = df.dropna()
    cat_data = df.select_dtypes('object').astype('category')
    for c in cat_data:
        df = df.assign(**{c : cat_data[c]})

    numeric_data = df.select_dtypes('int64').astype('double')
    for c in numeric_data:
        df = df.assign(**{c : numeric_data[c]})
    df.reset_index(drop=True, inplace=True)

    to_remove_features = remove_crossvalidated_nan(df, [10])
    df = df.drop(to_remove_features, axis=1)

    to_remove_features = linear_dependent_features(df)
    df = df.drop(to_remove_features, axis=1)    

    return df

class CVLikelihoodCheckInvalid(pbn.Score):

    def __init__(self, df, test_df, invalid_limit = 0.05, k=10, seed=0, arguments = pbn.Arguments()):
        pbn.Score.__init__(self)

        self.cv = pbn.CrossValidation(df, k, seed)
        self.arguments = arguments
        self.train_df = df
        self.test_df = test_df
        self.invalid_limit = invalid_limit

    def has_variables(self, vars):
        return all([v in self.test_df.columns for v in vars])
    
    def compatible_bn(self, model):
        return self.has_variables(model.nodes())

    def local_score(self, model, variable, evidence):
        return self.local_score_node_type(model, model.underlying_node_type(self.data(), variable), variable, evidence)

    def local_score_node_type(self, model, variable_type, variable, evidence):
        args, kwargs = self.arguments.args(variable, variable_type)
        cpd = variable_type.new_factor(model, variable, evidence, *args, **kwargs)

        loglik = 0

        test_invalid_threshold = self.invalid_limit * self.test_df.shape[0]
        test_invalid_total = 0

        for train_df, validation_df in self.cv.loc([variable] + evidence):
            cpd.fit(train_df)
            loglik += cpd.slogl(validation_df)
            if np.isnan(loglik):
                return -np.inf

            test_ll = cpd.logl(self.test_df)
            test_invalid_total += np.isnan(test_ll).sum()

            if test_invalid_total > test_invalid_threshold:
                return -np.inf

        return loglik

    def data(self):
        return self.train_df

class ValidatedLikelihoodCheckInvalid(pbn.ValidatedScore):

    def __init__(self, df, test_df, test_ratio=0.2, invalid_limit = 0.05, k=10, seed=0, arguments = pbn.Arguments()):
        pbn.ValidatedScore.__init__(self)

        self.holdout = pbn.HoldoutLikelihood(df, test_ratio, seed, arguments)
        self.cv = CVLikelihoodCheckInvalid(self.holdout.training_data(), test_df, invalid_limit, k, seed, arguments)

    def has_variables(self, vars):
        return self.holdout.has_variable(vars)
    
    def compatible_bn(self, model):
        return self.holdout.compatible_bn(model)

    def local_score(self, model, variable, evidence):
        return self.cv.local_score(model, variable, evidence)

    def local_score_node_type(self, model, variable_type, variable, evidence):
        return self.cv.local_score_node_type(model, variable_type, variable, evidence)

    def vlocal_score(self, model, variable, evidence):
        return self.holdout.local_score(model, variable, evidence)

    def vlocal_score_node_type(self, model, variable_type, variable, evidence):
        return self.holdout.local_score_node_type(model, variable_type, variable, evidence)

    def data(self):
        return self.cv.data()


def train_hc_clg_bic(df_name, train_df, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/CLG/BIC/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = pbn.GreedyHillClimbing()
    bic = pbn.BIC(train_df)
    arc_set = pbn.ArcOperatorSet()

    cb_save = pbn.SaveModel(fold_folder)
    start_model = pbn.CLGNetwork(list(train_df.columns.values))

    bn = hc.estimate(arc_set, bic, start_model, callback=cb_save)
    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def train_hc_clg_vl(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/CLG/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = pbn.GreedyHillClimbing()
    # vl = pbn.ValidatedLikelihood(train_df, seed=SEED)
    vl = ValidatedLikelihoodCheckInvalid(train_df, test_df, seed=SEED)
    arc_set = pbn.ArcOperatorSet()

    cb_save = pbn.SaveModel(fold_folder)
    start_model = pbn.CLGNetwork(list(train_df.columns.values))

    bn = hc.estimate(arc_set, vl, start_model, patience=patience, callback=cb_save)
    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def train_hc_hspbn_clg(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/HSPBN/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = pbn.GreedyHillClimbing()
    # vl = pbn.ValidatedLikelihood(train_df, seed=SEED)
    vl = ValidatedLikelihoodCheckInvalid(train_df, test_df, seed=SEED)
    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])

    cb_save = pbn.SaveModel(fold_folder)
    start_model = pbn.SemiparametricBN(list(train_df.columns.values))

    bn = hc.estimate(pool, vl, start_model, patience=patience, callback=cb_save)
    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def train_hc_hspbn_hckde(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/HSPBN_HCKDE/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = pbn.GreedyHillClimbing()
    # vl = pbn.ValidatedLikelihood(train_df, seed=SEED)
    vl = ValidatedLikelihoodCheckInvalid(train_df, test_df, seed=SEED)
    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])

    cb_save = pbn.SaveModel(fold_folder)
    node_types = [(name, pbn.CKDEType()) for name in train_df.select_dtypes('double').columns.values] +\
                 [(name, pbn.DiscreteFactorType()) for name in train_df.select_dtypes('category').columns.values]

    start_model = pbn.SemiparametricBN(list(train_df.columns.values), node_types)

    bn = hc.estimate(pool, vl, start_model, patience=patience, callback=cb_save)

    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def train_hc_models(df_name, df):
    chunks = int(np.ceil(EVALUATION_FOLDS / PARALLEL_THREADS))
    
    fold_indices = list(KFold(EVALUATION_FOLDS, shuffle=True, random_state=SEED).split(df))

    for ch in range(chunks):
        num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
        with mp.Pool(processes=num_threads) as p:
            p.starmap(train_hc_clg_bic, [(df_name, df.iloc[fold_indices[idx_fold][0],:], idx_fold)
                                            for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

    for patience in PATIENCE:
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                p.starmap(train_hc_clg_vl,
                    [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                    for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

    for patience in PATIENCE:
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                p.starmap(train_hc_hspbn_clg, 
                    [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                    for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

    for patience in PATIENCE:
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                p.starmap(train_hc_hspbn_hckde,
                    [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                    for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

def test_hc_clg_bic(df_name, train_df, test_df, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/CLG/BIC/' + str(idx_fold)
    all_models = sorted(glob.glob(fold_folder + '/*.pickle'))
    final_model = pbn.load(all_models[-1])

    final_model.fit(train_df)
    return final_model.logl(test_df)

def test_hc_clg_vl(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/CLG/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    all_models = sorted(glob.glob(fold_folder + '/*.pickle'))
    final_model = pbn.load(all_models[-1])

    final_model.fit(train_df)
    return final_model.logl(test_df)

def test_hc_hspbn_clg(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/HSPBN/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    all_models = sorted(glob.glob(fold_folder + '/*.pickle'))
    final_model = pbn.load(all_models[-1])

    args = pbn.Arguments({
        pbn.CKDEType(): (PluginEstimator(),)
    })

    final_model.fit(train_df)
    return final_model.logl(test_df)

def test_hc_hspbn_hckde(df_name, train_df, test_df, patience, idx_fold):
    fold_folder = 'models/' + df_name + '/HillClimbing/HSPBN_HCKDE/ValidationLikelihood_' + str(patience) + '/' + str(idx_fold)
    all_models = sorted(glob.glob(fold_folder + '/*.pickle'))
    final_model = pbn.load(all_models[-1])

    args = pbn.Arguments({
        pbn.CKDEType(): (PluginEstimator(),)
    })

    final_model.fit(train_df)
    return final_model.logl(test_df)

def unfold_predictions(results):
    p = np.asarray([])
    for r in results:
        p = np.hstack((p, r))

    return p

def test_hc_models(df_name, df):
    chunks = int(np.ceil(EVALUATION_FOLDS / PARALLEL_THREADS))
    
    fold_indices = list(KFold(EVALUATION_FOLDS, shuffle=True, random_state=SEED).split(df))

    bic_result = []
    for ch in range(chunks):
        num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
        with mp.Pool(processes=num_threads) as p:
            tmp_result = p.starmap(test_hc_clg_bic, 
                                  [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], idx_fold)
                                    for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

        bic_result.extend(tmp_result)

    bic_result = unfold_predictions(bic_result)

    clg_vl_result = []
    for patience in PATIENCE:
        result = []
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                tmp_result = p.starmap(test_hc_clg_vl, 
                        [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                        for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

            result.extend(tmp_result)

        clg_vl_result.append(unfold_predictions(result))

    hspbn_vl_result = []
    for patience in PATIENCE:
        result = []
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                tmp_result = p.starmap(test_hc_hspbn_clg,
                        [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                        for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

            result.extend(tmp_result)

        hspbn_vl_result.append(unfold_predictions(result))

    hspbn_hckde_vl_result = []
    for patience in PATIENCE:
        result = []
        for ch in range(chunks):
            num_threads = int(np.minimum(PARALLEL_THREADS, EVALUATION_FOLDS -  PARALLEL_THREADS*ch))
            with mp.Pool(processes=num_threads) as p:
                tmp_result = p.starmap(test_hc_hspbn_hckde,
                        [(df_name, df.iloc[fold_indices[idx_fold][0],:], df.iloc[fold_indices[idx_fold][1],:], patience, idx_fold)
                                for idx_fold in range(ch*PARALLEL_THREADS, ch*PARALLEL_THREADS + num_threads)]
                )

            result.extend(tmp_result)

        hspbn_hckde_vl_result.append(unfold_predictions(result))

    return (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

def common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result):
    is_invalid_instance = np.logical_or(np.isnan(bic_result), np.isinf(bic_result))

    for clg_vl in clg_vl_result:
        local_invalid = np.logical_or(np.isnan(clg_vl), np.isinf(clg_vl))
        is_invalid_instance = np.logical_or(is_invalid_instance, local_invalid)

    for hspbn_vl in hspbn_vl_result:
        local_invalid = np.logical_or(np.isnan(hspbn_vl), np.isinf(hspbn_vl))
        is_invalid_instance = np.logical_or(is_invalid_instance, local_invalid)
    
    for hspbn_hckde_vl in hspbn_hckde_vl_result:
        local_invalid = np.logical_or(np.isnan(hspbn_hckde_vl), np.isinf(hspbn_hckde_vl))
        is_invalid_instance = np.logical_or(is_invalid_instance, local_invalid)

    return (bic_result[~is_invalid_instance],
            [ll[~is_invalid_instance] for ll in clg_vl_result],
            [ll[~is_invalid_instance] for ll in hspbn_vl_result],
            [ll[~is_invalid_instance] for ll in hspbn_hckde_vl_result])

def print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result):
    print("CLG-BIC logl: " + str(bic_result.sum()))

    for p, r in zip(PATIENCE, clg_vl_result):
        print("CLG ValidationLiklihood p = " + str(p) + ", logl: " + str(r.sum()))

    for p, r in zip(PATIENCE, hspbn_vl_result):
        print("HSPBN-CLG ValidationLiklihood p = " + str(p) + ", logl: " + str(r.sum()))

    for p, r in zip(PATIENCE, hspbn_hckde_vl_result):
        print("HSPBN-HCKDE ValidationLiklihood p = " + str(p) + ", logl: " + str(r.sum()))