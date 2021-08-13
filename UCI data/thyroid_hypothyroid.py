import pandas as pd
import util

def preprocess_dataframe(df):
    # TBG is completely null
    df = df.drop('TBG',axis=1)
    df = df.dropna()

    cat_columns = ["class", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
            "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor",
            "lithium", "goitre", "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured",
            "TBG_measured"]
    cont_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

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
    df = pd.read_csv('data/Thyroid/hypothyroid.data', na_values="?")
    df = preprocess_dataframe(df)

    util.train_hc_models('Thyroid-hypothyroid', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('Thyroid-hypothyroid', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)