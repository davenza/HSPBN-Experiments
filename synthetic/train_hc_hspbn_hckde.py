import os
import glob
import pandas as pd
import util
import generate_dataset_hspbn
import pybnesian as pbn
import pathlib
import math
import multiprocessing as mp

patience = util.PATIENCE

def run_hc_hspbn_hckde(idx_dataset, i):
    hc = pbn.GreedyHillClimbing()
    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])
    
    df = pd.read_csv('data/synthetic_' + str(idx_dataset).zfill(3) + '_' + str(i) + '.csv')
    df = generate_dataset_hspbn.preprocess_dataset(df)

    vl = pbn.ValidatedLikelihood(df, k=10, seed=util.SEED)
    for p in patience:
        result_folder = 'models/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/HillClimbing/HSPBN_HCKDE/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = pbn.SaveModel(result_folder)
        
        node_types = [(name, pbn.CKDEType()) for name in df.select_dtypes('double').columns.values] +\
                 [(name, pbn.DiscreteFactorType()) for name in df.select_dtypes('category').columns.values]
        start_model = pbn.SemiparametricBN(list(df.columns.values), node_types)

        bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p)

        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")
        with open(result_folder + '/end.lock', 'w') as f:
            pass


for i in util.INSTANCES:
    for idx_dataset in range(0, math.ceil(util.NUM_SIMULATIONS / 10)):

        num_processes = min(10, util.NUM_SIMULATIONS - idx_dataset*10)
        with mp.Pool(processes=num_processes) as p:
            p.starmap(run_hc_hspbn_hckde, [(10*idx_dataset + ii, i) for ii in range(num_processes)]
                        )