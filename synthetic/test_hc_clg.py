import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import util
from generate_new_bns import FixedDiscreteFactorType, FixedCLGType, NormalMixtureType,\
                        FixedDiscreteFactor, NormalMixtureCPD, FixedCLG, ProbabilisticModel
from generate_dataset import preprocess_dataset

def compare_models(num_instances):
    truth_ll = np.empty((util.NUM_SIMULATIONS,))

    ll_bic = np.empty((util.NUM_SIMULATIONS,))
    shd_bic = np.empty((util.NUM_SIMULATIONS,))
    hamming_bic = np.empty((util.NUM_SIMULATIONS,))

    ll_vl = np.empty((util.NUM_SIMULATIONS,))
    shd_vl = np.empty((util.NUM_SIMULATIONS,))
    hamming_vl = np.empty((util.NUM_SIMULATIONS,))

    for i in range(util.NUM_SIMULATIONS):
        test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
        test_df = preprocess_dataset(test_df)

        true_model = ProbabilisticModel.load('ground_truth_models/model_' + str(i) + '.pickle')
        truth_ll[i] = true_model.ground_truth_bn.slogl(test_df)

    print("True model loglik: " + str(truth_ll.mean()))

    for p in util.PATIENCE:
        for i in range(util.NUM_SIMULATIONS):
            true_model = ProbabilisticModel.load('ground_truth_models/model_' + str(i) + '.pickle')

            train_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
            train_df = preprocess_dataset(train_df)
            test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
            test_df = preprocess_dataset(test_df)

            result_folder = 'models/' +  str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/CLG/BIC_' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            all_models = sorted(glob.glob(result_folder + '/*.pickle'))
            final_model = load(all_models[-1])
            final_model.fit(train_df)

            ll_bic[i] = final_model.slogl(test_df)
            shd_bic[i] = util.shd(final_model, true_model.expected_bn)
            hamming_bic[i] = util.hamming(final_model, true_model.expected_bn)

            result_folder = 'models/' +  str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/CLG/ValidationLikelihood_' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            all_models = sorted(glob.glob(result_folder + '/*.pickle'))
            final_model = load(all_models[-1])
            final_model.fit(train_df)

            ll_vl[i] = final_model.slogl(test_df)
            shd_vl[i] = util.shd(final_model, true_model.expected_bn)
            hamming_vl[i] = util.hamming(final_model, true_model.expected_bn)

        print("Loglik, BIC p " + str(p) + ": " + str(ll_bic.mean()))
        print("Hamming, BIC p " + str(p) + ": " + str(hamming_bic.mean()))
        print("SHD, BIC p " + str(p) + ": " + str(shd_bic.mean()))
        print()

        print("Loglik, ValidationScore p " + str(p) + ": " + str(ll_vl.mean()))
        print("Hamming, ValidationScore p " + str(p) + ": " + str(hamming_vl.mean()))
        print("SHD, ValidationScore p " + str(p) + ": " + str(shd_vl.mean()))
        print()

if __name__ == '__main__':

    for i in util.INSTANCES:
        print(str(i) + " instances")
        print("=======================")
        compare_models(i)

