import pandas as pd
import util

def preprocess_dataframe(df):
    cat_columns = ["Wilderness_" + str(i) for i in range(1, 5)] + ["Soil_" + str(i) for i in range(1, 41)] +\
                    ["Cover_Type"]
    cont_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am","Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

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
    df = pd.read_csv('data/Cover Type/covtype_truncated.csv')
    df = preprocess_dataframe(df)

    util.train_hc_models('CoverType', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('CoverType', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)