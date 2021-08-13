import numpy as np
import pandas as pd
import plot_cd_diagram

import tikzplotlib
import util
import adult
import australian_statlog
import cover_type
import german_statlog
import kdd
import liver_disorders
import thyroid_hypothyroid
import thyroid_sick

def result_string(df_name, df):
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = util.test_hc_models(df_name, df)

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) =\
        util.common_instance_results(bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result)

    return df_name + ',' + str(bic_result.sum()) +\
                ',' + ','.join([str(ll.sum()) for ll in clg_vl_result]) +\
                ',' + ','.join([str(ll.sum()) for ll in hspbn_vl_result]) +\
                ',' + ','.join([str(ll.sum()) for ll in hspbn_hckde_vl_result])


def save_summary_results():
    string_file = "Dataset,CLG_BIC," +\
                ','.join(["CLG_" + str(p) for p in util.PATIENCE]) + ',' +\
                ','.join(["HSPBN_" + str(p) for p in util.PATIENCE]) + ',' +\
                ','.join(["HSPBN_HCKDE_" + str(p) for p in util.PATIENCE]) + '\n'

    abalone = pd.read_csv('data/Abalone/abalone.data')
    abalone = util.preprocess_dataframe(abalone)
    string_file += result_string('Abalone', abalone) + '\n'
        
    adult_train = pd.read_csv('data/Adult/adult.data', na_values="?")
    adult_test = pd.read_csv('data/Adult/adult.test', na_values="?")
    adult_df = pd.concat([adult_train, adult_test], ignore_index=True)
    adult_df = adult.preprocess_dataframe(adult_df)
    string_file += result_string('Adult', adult_df) + '\n'

    australian = pd.read_csv('data/AustralianStatlog/australian.dat')
    australian = australian_statlog.preprocess_dataframe(australian)
    string_file += result_string('AustralianStatlog', australian) + '\n'

    cover = pd.read_csv('data/Cover Type/covtype_truncated.csv')
    cover = cover_type.preprocess_dataframe(cover)
    string_file += result_string('CoverType', cover) + '\n'
    
    credit_approval = pd.read_csv('data/Credit Approval/crx.data', na_values="?")
    credit_approval = util.preprocess_dataframe(credit_approval)
    string_file += result_string('CreditApproval', credit_approval) + '\n'

    german = pd.read_csv('data/GermanStatlog/german.data')
    german = german_statlog.preprocess_dataframe(german)
    string_file += result_string('GermanStatlog', german) + '\n'

    kdd_cup = pd.read_csv('data/KDD Cup/kddcup_truncated.csv')
    kdd_cup = kdd.preprocess_dataframe(kdd_cup)
    string_file += result_string('KDDCup', kdd_cup) + '\n'

    liver = pd.read_csv('data/Liver disorders/bupa.data')
    liver = liver_disorders.preprocess_dataframe(liver)
    string_file += result_string('LiverDisorders', liver) + '\n'

    hypothyroid = pd.read_csv('data/Thyroid/hypothyroid.data', na_values="?")
    hypothyroid = thyroid_hypothyroid.preprocess_dataframe(hypothyroid)
    string_file += result_string('Thyroid-hypothyroid', hypothyroid) + '\n'

    sick = pd.read_csv('data/Thyroid/sick.data', na_values="?")
    sick = thyroid_sick.preprocess_dataframe(sick)
    string_file += result_string('Thyroid-sick', sick)

    with open('result_summary.csv', 'w') as f:
        f.write(string_file)

def plot_cd_diagrams(rename_dict):
    df_algorithms = pd.read_csv('result_summary.csv')
    df_algorithms = df_algorithms.set_index('Dataset')

    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    names = [rename_dict[s] for s in names]

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="cd")
    tikzplotlib.save("plots/Nemenyi.tex", standalone=True, axis_width="14cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="holm")
    tikzplotlib.save("plots/Holm.tex", standalone=True, axis_width="14cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], textspace=1.1, posthoc_method="bergmann")
    tikzplotlib.save("plots/Bergmann.tex", standalone=True, axis_width="14cm", axis_height="5cm")


if __name__ == '__main__':
    rename_dict = {
        'CLG_BIC': r'CLGBN-BIC',
        'CLG_0': r'CLGBN $\lambda=0$',
        'CLG_5': r'CLGBN $\lambda=5$',
        'HSPBN_0': r'HSPBN-CLG $\lambda=0$',
        'HSPBN_5': r'HSPBN-CLG $\lambda=5$',
        'HSPBN_HCKDE_0': r'HSPBN-HCKDE $\lambda=0$',
        'HSPBN_HCKDE_5': r'HSPBN-HCKDE $\lambda=5$',
    }
    
    save_summary_results()
    plot_cd_diagrams(rename_dict)