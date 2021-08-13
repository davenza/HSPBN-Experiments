import math
import numpy as np
import pandas as pd
import util
import pybnesian as pbn
from scipy.stats import norm
import pathlib

def generate_dataset(size, seed = 0):
    np.random.seed(seed)
    def generate_A():
        a_dict = np.asarray(['a1', 'a2'])
        a_values = a_dict[np.random.choice(a_dict.size, size, p=[0.75, 0.25])]
        return pd.Series(a_values, name="A", dtype="category")

    def generate_B(evidence):
        a1_indices = evidence["A"] == 'a1'

        b_dict = np.asarray(['b1', 'b2', 'b3'])

        b_values = np.empty_like(evidence["A"].to_numpy())
        b_values[a1_indices] = b_dict[np.random.choice(b_dict.size, np.sum(a1_indices), p=[0.33, 0.33, 0.34])]
        b_values[~a1_indices] = b_dict[np.random.choice(b_dict.size, np.sum(~a1_indices), p=[0, 0.8, 0.2])]

        return pd.Series(b_values, name="B", dtype="category")

    def generate_C():
        c_dict = np.asarray(['c1', 'c2', "c3", "c4"])
        c_values = c_dict[np.random.choice(c_dict.size, size, p=[0.25, 0.25, 0.25, 0.25])]
        return pd.Series(c_values, name="C", dtype="category")
    
    def generate_D(evidence):
        a_values = evidence['A'].to_numpy()
        b_values = evidence['B'].to_numpy()

        a1b1_indices = np.logical_and(a_values == 'a1', b_values == 'b1')
        a1b2_indices = np.logical_and(a_values == 'a1', b_values == 'b2')
        a1b3_indices = np.logical_and(a_values == 'a1', b_values == 'b3')
        a2b1_indices = np.logical_and(a_values == 'a2', b_values == 'b1')
        a2b2_indices = np.logical_and(a_values == 'a2', b_values == 'b2')
        a2b3_indices = np.logical_and(a_values == 'a2', b_values == 'b3')

        d_values = np.empty((size,), dtype=float)

        d_values[a1b1_indices] = np.random.normal(-3., math.sqrt(1), size=a1b1_indices.sum())
        d_values[a1b2_indices] = np.random.normal(-1.5, math.sqrt(0.7), size=a1b2_indices.sum())
        d_values[a1b3_indices] = np.random.normal(0., math.sqrt(0.5), size=a1b3_indices.sum())
        d_values[a2b1_indices] = np.random.normal(1., math.sqrt(1.25), size=a2b1_indices.sum())
        d_values[a2b2_indices] = np.random.normal(2., math.sqrt(1.5), size=a2b2_indices.sum())
        d_values[a2b3_indices] = np.random.normal(3., math.sqrt(2.), size=a2b3_indices.sum())

        return pd.Series(d_values, name="D", dtype=float)

    def generate_E(evidence):
        e_dict = np.asarray(['e1', 'e2'])
        e_values = e_dict[np.random.choice(e_dict.size, size, p=[0.5, 0.5])]
        return pd.Series(e_values, name="E", dtype="category")

    def generate_G(evidence):
        c1_indices = evidence["C"] == 'c1'
        c2_indices = evidence["C"] == 'c2'
        c3_indices = evidence["C"] == 'c3'
        c4_indices = evidence["C"] == 'c4'

        g_values = np.empty((size,), dtype=float)

        d_values = evidence["D"].to_numpy()

        g_values[c1_indices] = np.random.normal(0, math.sqrt(1), size=c1_indices.sum())
        g_values[c2_indices] = -3. + 2.5 * d_values[c2_indices] + np.random.normal(0, math.sqrt(0.5), size=c2_indices.sum())
        g_values[c3_indices] = 2. - 1.25 * d_values[c3_indices] + np.random.normal(0, math.sqrt(2), size=c3_indices.sum())
        g_values[c4_indices] = 5. * d_values[c4_indices] + np.random.normal(0, math.sqrt(0.25), size=c4_indices.sum())

        return pd.Series(g_values, name="G", dtype="float")

    def generate_H(evidence):
        e1_indices = evidence["E"] == 'e1'
        e2_indices = evidence["E"] == 'e2'

        d_values = evidence['D'].to_numpy()

        h_values = np.empty((size,), dtype=float)

        h_values[e1_indices] = util.sample_mixture([0.5, 0.5], [-1, 1], [1, 1], [d_values[e1_indices]], 
                                    np.asarray([[0, -1], [0, 1.]]), e1_indices.sum(), seed=seed)
        h_values[e2_indices] = util.sample_mixture([0.3, 0.4, 0.3], [-2., 0., 5.], [0.5, 1., 2.], [d_values[e2_indices]],
                                    np.asarray([[-2., 0.5], [0., 0.], [5., -1.5]]), e2_indices.sum(), seed=seed+1)

        return pd.Series(h_values, name="H", dtype="float")

    def generate_I(evidence):
        g_values = evidence['G'].to_numpy()
        h_values = evidence['H'].to_numpy()

        i_values = g_values * h_values + np.random.normal(loc=0, scale=math.sqrt(0.5), size=size)

        return pd.Series(i_values, name="I", dtype="float")
    
    df = pd.DataFrame()
    df['A'] = generate_A()
    df['B'] = generate_B(df)
    df['C'] = generate_C()
    df['D'] = generate_D(df)
    df['E'] = generate_E(df)
    df['G'] = generate_G(df)
    df['H'] = generate_H(df)
    df['I'] = generate_I(df)

    return df

def preprocess_dataset(df):
    for c in ['A', 'B', 'C', 'E']:
        df[c] = df[c].astype('category')
        df[c].cat.categories = df[c].cat.categories.astype('string')
    
    return df

def slogl_model(df):
    a_values = df['A'].to_numpy()
    b_values = df['B'].to_numpy()
    c_values = df['C'].to_numpy()
    d_values = df['D'].to_numpy()
    e_values = df['E'].to_numpy()
    g_values = df['G'].to_numpy()
    h_values = df['H'].to_numpy()
    i_values = df['I'].to_numpy()

    ################# 
    # Node A
    ################# 

    a_dict = np.asarray(['a1', 'a2'])
    a_prob = np.asarray([0.75, 0.25])
    a_indices = np.where(a_values[:, np.newaxis] == a_dict)[1]

    ll_a = a_prob[a_indices]
    slogl = np.log(ll_a).sum()

    ################# 
    # Node B
    ################# 

    b_dict = np.asarray(['b1', 'b2', 'b3'])
    b_prob = np.asarray([[0.33, 0.33, 0.34], [0, 0.8, 0.2]])
    b_indices = np.where(b_values[:, np.newaxis] == b_dict)[1]

    ll_b = b_prob[a_indices, b_indices]
    slogl += np.log(ll_b).sum()

    ################# 
    # Node C
    ################# 

    c_dict = np.asarray(['c1', 'c2', "c3", "c4"])
    c_prob = np.asarray([0.25, 0.25, 0.25, 0.25])
    c_indices = np.where(c_values[:, np.newaxis] == c_dict)[1]

    ll_c = c_prob[c_indices]
    slogl += np.log(ll_c).sum()

    ################# 
    # Node D
    ################# 

    a1b1_indices = np.logical_and(a_values == 'a1', b_values == 'b1')
    a1b2_indices = np.logical_and(a_values == 'a1', b_values == 'b2')
    a1b3_indices = np.logical_and(a_values == 'a1', b_values == 'b3')
    a2b1_indices = np.logical_and(a_values == 'a2', b_values == 'b1')
    a2b2_indices = np.logical_and(a_values == 'a2', b_values == 'b2')
    a2b3_indices = np.logical_and(a_values == 'a2', b_values == 'b3')

    ll_d = np.empty((d_values.shape[0],), dtype=float)

    ll_d[a1b1_indices] = norm(-3., math.sqrt(1.)).logpdf(d_values[a1b1_indices])
    ll_d[a1b2_indices] = norm(-1.5, math.sqrt(0.7)).logpdf(d_values[a1b2_indices])
    ll_d[a1b3_indices] = norm(0., math.sqrt(0.5)).logpdf(d_values[a1b3_indices])
    ll_d[a2b1_indices] = norm(1., math.sqrt(1.25)).logpdf(d_values[a2b1_indices])
    ll_d[a2b2_indices] = norm(2., math.sqrt(1.5)).logpdf(d_values[a2b2_indices])
    ll_d[a2b3_indices] = norm(3., math.sqrt(2.)).logpdf(d_values[a2b3_indices])

    slogl += ll_d.sum()

    ################# 
    # Node E
    ################# 

    e_dict = np.asarray(['e1', 'e2'])
    e_prob = np.asarray([0.5, 0.5])
    e_indices = np.where(e_values[:, np.newaxis] == e_dict)[1]

    ll_e = e_prob[e_indices]
    slogl += np.log(ll_e).sum()

    ################# 
    # Node G
    #################

    c1_indices = c_values == 'c1'
    c2_indices = c_values == 'c2'
    c3_indices = c_values == 'c3'
    c4_indices = c_values == 'c4'

    ll_g = np.empty((g_values.shape[0],), dtype=float)

    ll_g[c1_indices] = norm(0., math.sqrt(1.)).logpdf(g_values[c1_indices])
    ll_g[c2_indices] = norm(-3. + 2.5 * d_values[c2_indices], math.sqrt(0.5)).logpdf(g_values[c2_indices])
    ll_g[c3_indices] = norm(2. - 1.25 * d_values[c3_indices], math.sqrt(2.)).logpdf(g_values[c3_indices])
    ll_g[c4_indices] = norm(5. * d_values[c4_indices], math.sqrt(0.25)).logpdf(g_values[c4_indices])

    slogl += ll_g.sum()

    ################# 
    # Node H
    #################

    e1_indices = e_values == 'e1'
    e2_indices = e_values == 'e2'

    ll_h = np.empty((h_values.shape[0],), dtype=float)

    ll_h[e1_indices] = np.log(0.5 * norm(-d_values[e1_indices], math.sqrt(1.)).pdf(h_values[e1_indices]) +\
                              0.5 * norm(d_values[e1_indices], math.sqrt(1.)).pdf(h_values[e1_indices]))
    ll_h[e2_indices] = np.log(0.3 * norm(-2. + 0.5 * d_values[e2_indices], math.sqrt(0.5)).pdf(h_values[e2_indices]) +\
                              0.4 * norm(0., math.sqrt(1.)).pdf(h_values[e2_indices]) +\
                              0.3 * norm(5. - 1.5 * d_values[e2_indices], math.sqrt(2.)).pdf(h_values[e2_indices]))

    slogl += ll_h.sum()

    ################# 
    # Node I
    #################
    
    ll_i = norm(g_values * h_values, math.sqrt(0.5)).logpdf(i_values)
    slogl += ll_i.sum()

    return slogl

if __name__ == "__main__":
    pathlib.Path('data/').mkdir(parents=True, exist_ok=True)

    for i in range(util.NUM_SIMULATIONS):
        dataset200 = generate_dataset(200, seed=i*100)
        dataset200.to_csv("data/synthetic_" + str(i).zfill(3) + "_200.csv", index=False)

        dataset2000 = generate_dataset(2000, seed=1 + (i*100))
        dataset2000.to_csv("data/synthetic_" + str(i).zfill(3) + "_2000.csv", index=False)

        dataset10000 = generate_dataset(10000, seed=2 + (i*100))
        dataset10000.to_csv("data/synthetic_" + str(i).zfill(3) + "_10000.csv", index=False)

        dataset_test = generate_dataset(1000, seed=3 + (i*100))
        dataset_test.to_csv("data/synthetic_" + str(i).zfill(3) + "_test.csv", index=False)

    model = pbn.SemiparametricBN(['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I'],
                            [('A', 'B'), ('A', 'D'), ('B', 'D'), ('C', 'G'), 
                                ('D', 'G'), ('D', 'H'), ('E', 'H'), ('G', 'I'), ('H', 'I')],
                            [('A', pbn.DiscreteFactorType()), 
                            ('B', pbn.DiscreteFactorType()), 
                            ('C', pbn.DiscreteFactorType()), 
                            ('D', pbn.LinearGaussianCPDType()),
                            ('E', pbn.DiscreteFactorType()),
                            ('G', pbn.LinearGaussianCPDType()),
                            ('H', pbn.CKDEType()),
                            ('I', pbn.CKDEType())])

    model.save("true_model.pickle")