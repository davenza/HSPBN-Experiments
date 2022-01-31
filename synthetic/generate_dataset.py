from generate_new_bns import FixedDiscreteFactorType, FixedCLGType, NormalMixtureType,\
                        FixedDiscreteFactor, NormalMixtureCPD, FixedCLG, ProbabilisticModel
import util


def preprocess_dataset(df):
    for c in ['A', 'B', 'C', 'E']:
        df[c] = df[c].astype('category')
        df[c].cat.categories = df[c].cat.categories.astype('string')
    
    return df

if __name__ == "__main__":

    for i in range(util.NUM_SIMULATIONS):
        model = ProbabilisticModel.load('ground_truth_models/model_' + str(i) + '.pickle')

        dataset200 = model.ground_truth_bn.sample(200, seed=i*100, ordered=True).to_pandas()
        dataset200.to_csv("data/synthetic_" + str(i).zfill(3) + "_200.csv", index=False)

        dataset2000 = model.ground_truth_bn.sample(2000, seed=1 + (i*100), ordered=True).to_pandas()
        dataset2000.to_csv("data/synthetic_" + str(i).zfill(3) + "_2000.csv", index=False)

        dataset10000 = model.ground_truth_bn.sample(10000, seed=2 + (i*100), ordered=True).to_pandas()
        dataset10000.to_csv("data/synthetic_" + str(i).zfill(3) + "_10000.csv", index=False)

        dataset_test = model.ground_truth_bn.sample(1000, seed=3 + (i*100), ordered=True).to_pandas()
        dataset_test.to_csv("data/synthetic_" + str(i).zfill(3) + "_test.csv", index=False)




