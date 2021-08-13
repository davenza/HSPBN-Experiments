import pandas as pd
import util

def preprocess_dataframe(df):
    cat_columns = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12", "A15"]
    cont_columns = ["A2", "A3", "A7", "A10", "A13", "A14"]

    for c in cat_columns:
        df[c] = df[c].astype('category')
        df[c].cat.categories = df[c].cat.categories.astype('string')

    for c in cont_columns:
        df[c] = df[c].astype('double')
        
    to_remove_features = util.remove_crossvalidated_nan(df, [10])
    df = df.drop(to_remove_features, axis=1)

    to_remove_features = util.linear_dependent_features(df)
    df = df.drop(to_remove_features, axis=1)

    return df

if __name__ == '__main__':
    df = pd.read_csv('data/AustralianStatlog/australian.dat')
    df = preprocess_dataframe(df)

    util.train_hc_models('AustralianStatlog', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('AustralianStatlog', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)