import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
import pybnesian as pbn
from pybnesian import load
import util
from generate_new_bns import FixedDiscreteFactorType, FixedCLGType, NormalMixtureType,\
                        FixedDiscreteFactor, NormalMixtureCPD, FixedCLG, ProbabilisticModel
from generate_dataset import preprocess_dataset
import struct


def compare_models(num_instances, bandwidth_selection="normal_reference"):
    clg_bic = np.empty((util.NUM_SIMULATIONS,))
    clg_vl = np.empty((util.NUM_SIMULATIONS,))
    hspbn_clg_vl = np.empty((util.NUM_SIMULATIONS,))
    hspbn_hckde_vl = np.empty((util.NUM_SIMULATIONS,))

    for p in util.PATIENCE:
        for i in range(util.NUM_SIMULATIONS):

            bic_folder = 'models/' + str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/CLG/BIC_' + str(p)

            with open(bic_folder + '/time', 'rb') as f:
                clg_bic[i] = struct.unpack('<d', f.read())[0]

            clg_vl_folder = 'models/' + str(i).zfill(3) + '/' + str(num_instances) +\
                             '/HillClimbing/CLG/ValidationLikelihood_' + str(p)

            with open(clg_vl_folder + '/time', 'rb') as f:
                clg_vl[i] = struct.unpack('<d', f.read())[0]

            hspbn_clg_folder = 'models/' + str(i).zfill(3) + '/' + str(num_instances) +\
                             '/HillClimbing/HSPBN/' + str(p)

            with open(hspbn_clg_folder + '/time', 'rb') as f:
                hspbn_clg_vl[i] = struct.unpack('<d', f.read())[0]

            hspbn_hckde_folder = 'models/' + str(i).zfill(3) + '/' + str(num_instances) +\
                             '/HillClimbing/HSPBN_HCKDE/' + str(p)

            with open(hspbn_hckde_folder + '/time', 'rb') as f:
                hspbn_hckde_vl[i] = struct.unpack('<d', f.read())[0]

        print("BIC p = " + str(p) + ": " + str(clg_bic.mean()))
        print("CLG-VL p = " + str(p) + ": " + str(clg_vl.mean()))
        print("HSPBN-CLG p = " + str(p) + ": " + str(hspbn_clg_vl.mean()))
        print("HSPBN-HCKDE p = " + str(p) + ": " + str(hspbn_hckde_vl.mean()))

if __name__ == '__main__':

    for i in util.INSTANCES:
        print(str(i) + " instances")
        print("=======================")
        compare_models(i)

