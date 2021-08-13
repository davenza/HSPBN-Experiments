import numpy as np
import pandas as pd
import util

def preprocess_dataframe(df):
    index_constant = np.where(df.nunique() == 1)[0]
    constant_columns = [df.columns[i] for i in index_constant]
    df = df.drop(constant_columns, axis=1)

    cat_columns = ["protocol_type", "service", "flag", "logged_in", "is_guest_login", "connection_type"]
    cont_columns = ["duration", "src_bytes", "hot", "num_compromised", "root_shell", "num_file_creations",
                "num_access_files", "count", "srv_count", "serror_rate", "srv_serror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_srv_rerror_rate"]

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
    df = pd.read_csv('data/KDD Cup/kddcup_truncated.csv')    
    df = preprocess_dataframe(df)

    util.train_hc_models('KDDCup', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('KDDCup', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)