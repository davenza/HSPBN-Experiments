import pandas as pd
import util

if __name__ == '__main__':
    df = pd.read_csv('data/Credit Approval/crx.data', na_values="?")
    df = util.preprocess_dataframe(df)

    util.train_hc_models('CreditApproval', df)
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models('CreditApproval', df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    util.print_logl_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)