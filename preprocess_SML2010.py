#Here we preprocess the SML2010 home temperature data.
#Download SML2010 dataset at: 
#   https://archive.ics.uci.edu/ml/datasets/SML2010
#To run, enter into terminal 
#   python3 preprocess_SML2010.py '/Users/My/path/to/SML2010_data/'
#We use this data to evaluate the use of a periodic kernel 
#privacy mechanism

#

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import decimate

import argparse
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

def get_traces(base_dir, downsamp = 4): 
    """returns preprocessed traces of tempterature 
    Inputs: 
        -base_dir: string of base directory containing SML2010 files e.g. 
         "/Users/me/SML2010_data/"
        -downsamp: rate by which we should downsample temperature data (recorded every 15 minutes) 
    """
    data_1 = pd.read_csv(
                base_dir + 'NEW-DATA-1.T15.txt',
                delimiter = ' ')

    temps = data_1['3:Temperature_Comedor_Sensor'].values[800:]
    times = np.arange(len(temps)) * 15  * (1 / 60) * (1 / 24)
    data = np.concatenate((times[:,None], temps[:,None]), axis = 1)

    days = np.concatenate((np.arange(0, 20, 2)[:,None], np.arange(2,21,2)[:,None]), axis = 1)
    downsamp = 4

    traces = []

    for day in days:
        day_data_idx = (data[:,0] > day[0]) * (data[:,0] < day[1])
        day_data = data[day_data_idx,:]
        day_data[:,1] = day_data[:,1] - day_data[:,1].mean()
        day_data[:,1] = day_data[:,1] / day_data[:,1].std()
        day_data[:,0] = day_data[:,0] * 24 * 60 
        
        day_data_times = day_data[::downsamp,0] 
        day_data_vals = decimate(day_data[:,1], downsamp, ftype = 'iir')
        day_data = np.concatenate((day_data_times[:,None], day_data_vals[:,None]), axis = 1)
        
        traces.append(day_data)
    return traces

def pers_and_len_scales(traces):  
    """Fit a periodic GP to each temperature trace (1 day) in the list of 
    temperature traces
    Inputs: 
        - traces: output of get_traces() 
    Outputs: 
        - len_scales: list of optimal length scales, one per day of temp data
        - pers: listof optimal periods, one per day of temp data 
    """
    len_scales = []
    pers = []
    
    for trace in traces: 
        temp_gp = GaussianProcessRegressor(kernel = ExpSineSquared(length_scale = 20, periodicity = 24 * 60, 
                                                               length_scale_bounds=(0.01, 100), 
                                                               periodicity_bounds=(16, 34)), 
                                                               alpha = 0.1)
        temp_gp.fit(np.arange(48)[:,None], trace[:,1][:,None])
        len_scales.append(temp_gp.kernel_.get_params()['length_scale'])
        pers.append(temp_gp.kernel_.get_params()['periodicity'])

    return len_scales, pers

# create parser to get base directory  
parser = argparse.ArgumentParser()
parser.add_argument("bdir") 
args = parser.parse_args()
base_dir = args.bdir 

#make data directory if it doesn't already exist 
Path("./saved_data").mkdir(parents=True, exist_ok=True)

#get data
print('Getting SML2010 temperature traces from ' + base_dir + '...') 
traces = get_traces(base_dir)

#get lengthscales and periods of traces 
#no need to normalize -- length scales are already l_eff
print('Getting length scales of traces and saving to ./saved_data/l_eff_temp.npy ...')
len_scales, pers = pers_and_len_scales(traces)
np.save('./saved_data/l_eff_temp.npy', len_scales)
