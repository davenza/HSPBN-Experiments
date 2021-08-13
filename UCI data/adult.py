import pandas as pd
import util

def preprocess_dataframe(df):
    df = df.drop("education-num", axis=1)
    return util.preprocess_dataframe(df)

if __name__ == '__main__':
    train_df = pd.read_csv('data/Adult/adult.data', na_values="?")
    test_df = pd.read_csv('data/Adult/adult.test', na_values="?")

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = preprocess_dataframe(df)

    util.train_hc_models('Adult', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('Adult', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)