import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import util
from generate_dataset_spbn import slogl_model, preprocess_dataset

true_model = load('true_model.pickle')

def compare_models(true_model, num_instances):
    ll = np.empty((util.NUM_SIMULATIONS,))
    shd = np.empty((util.NUM_SIMULATIONS,))
    hamming = np.empty((util.NUM_SIMULATIONS,))

    for i in range(util.NUM_SIMULATIONS):
        train_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
        train_df = preprocess_dataset(train_df)
        test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
        test_df = preprocess_dataset(test_df)

        result_folder = 'models/' +  str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/CLG/BIC'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        all_models = sorted(glob.glob(result_folder + '/*.pickle'))
        final_model = load(all_models[-1])
        final_model.fit(train_df)

        ll[i] = final_model.slogl(test_df)
        shd[i] = util.shd(final_model, true_model)
        hamming[i] = util.hamming(final_model, true_model)

    print("Loglik, BIC: " + str(ll.mean()))
    print("Hamming, BIC: " + str(hamming.mean()))
    print("SHD, BIC: " + str(shd.mean()))
    print()

    for p in util.PATIENCE:
        for i in range(util.NUM_SIMULATIONS):
            train_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
            train_df = preprocess_dataset(train_df)
            test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
            test_df = preprocess_dataset(test_df)

            result_folder = 'models/' +  str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/CLG/ValidationLikelihood_' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            all_models = sorted(glob.glob(result_folder + '/*.pickle'))
            final_model = load(all_models[-1])
            final_model.fit(train_df)

            ll[i] = final_model.slogl(test_df)
            shd[i] = util.shd(final_model, true_model)
            hamming[i] = util.hamming(final_model, true_model)

        print("Loglik, ValidationScore p " + str(p) + ": " + str(ll.mean()))
        print("Hamming, ValidationScore p " + str(p) + ": " + str(hamming.mean()))
        print("SHD, ValidationScore p " + str(p) + ": " + str(shd.mean()))
        print()

if __name__ == '__main__':
    ll = np.empty((util.NUM_SIMULATIONS,))
    for i in range(util.NUM_SIMULATIONS):
        test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
        ll[i] = slogl_model(test_df)

    print("True model loglik: " + str(ll.mean()))
    print()

    for i in util.INSTANCES:
        print(str(i) + " instances")
        print("=======================")
        compare_models(true_model, i)

