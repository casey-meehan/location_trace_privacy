#Preprocesses GeoLife data and saves into ./saved_data directory 
#Download the GeoLife dataset at 
#https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

#In our experiments we focus on traces that are approximately 5m in length 
#and are less than 50 locations in length. This includes about 55% of the original GeoLife dataset. 
#We then limit the number of traces / user to 50 (some users have contributed a large share of traces) 
#to avoid overrepresentation on their behalf. This leaves us with about 20% of the dataset. 
#These limitations are largely to restrict the problem cleanly, but make little difference in the validity 
#of our claims. Locations in traces are highly correlated virtually no matter how the data is sliced. 

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from pathlib import Path

def preproc_data(base_dir): 
    """Preprocesses geolife data into numpy arrays of user's XY trace data 
    Inputs: 
        -base_dir: string of base directory containing geolife files e.g. 
         "/Users/me/GeoLife_Data/Geolife_Trajectories/Data/"
    Outputs: 
        -user_datas: list containing a single list per user. Each user's list contains a series of numpy array traces. 
        Each numpy array trace has first column of time (s), second column of X coordinate and third column of Y 
        coordinate. The XY locations always start at zero (are normalized) and are in units of meters. 
    """
    n_users = 182
    user_datas = []

    for user in tqdm(range(n_users)): 
        #iterate through each trace of user 
        user_traces = []
        if (len(str(user)) == 1): 
            user_id = '00' + str(user)
        elif (len(str(user)) == 2): 
            user_id = '0' + str(user)
        else: 
            user_id = str(user)
        dir_name = base_dir + user_id + '/Trajectory/'
        for filename in os.listdir(dir_name): 
            #load trajectory
            trajectory_raw = pd.read_csv(
                dir_name + filename,
                delimiter = ',', header = None, skiprows = 6,
                names = ['lat','lon', '0', 'altitude', 't_1899', 'date', 'time' ])

            #modify time
            traj = trajectory_raw.copy()
            traj.drop(columns = '0', inplace = True)

            traj['t_1899'] = (trajectory_raw['t_1899'] - trajectory_raw['t_1899'].values[0]) * 24 * 60 * 60
            traj_newcols = np.array(traj.columns)
            traj_newcols[traj_newcols == 't_1899'] = 'seconds'
            traj.columns = traj_newcols

            #get X_vals: 
            R_earth = 6.371e6
            deg_to_rad = np.pi / 180
            traj['X'] = R_earth * np.cos(deg_to_rad * traj['lat'].values) * np.cos(deg_to_rad * traj['lon'].values)
            traj['Y'] = R_earth * np.cos(deg_to_rad *  traj['lat'].values) * np.sin(deg_to_rad * traj['lon'].values)

            #Normalize to first point: 
            origin_x = traj['X'].values[0]
            origin_y = traj['Y'].values[0]

            traj['X'] = traj['X'] - origin_x
            traj['Y'] = traj['Y'] - origin_y

            txy = traj.copy()
            txy = traj[['seconds', 'X', 'Y']]
            txy.columns = ['T', 'X', 'Y']
            if not np.isnan(txy.values).any(): 
                user_traces.append(txy.values)

        user_datas.append(user_traces)
    return user_datas

def downsample_data(user_datas, min_time = 280, max_time = 320, max_samps = 50): 
    """Focus on traces of a certain duration and number of samples: 
    Truncate/downsample traces to be between min_time and max_time in duration 
    and no more than max_samps in length. 
    Inputs: 
        - min_time: minimum trace duration 
        - max_time: maximum trace duration 
        - max_samps: maximum length of trace 
    Outputs: 
        -user_datas_: downsampled data 
    """
    user_datas_ = []
    for user in range(len(user_datas)): 
        traces = [] 
        for trace in range(len(user_datas[user])): 
            the_trace = user_datas[user][trace].copy()
            trace_time = the_trace[-1,0]
            trace_len = len(the_trace)
            
            if trace_time >= max_time: 
                #truncate to max_time 
                time_idx = the_trace[:,0] <= max_time
                the_trace = the_trace[time_idx,:]
                trace_len = len(the_trace)
                trace_time = the_trace[-1,0]
                
            if trace_len > max_samps: 
                #downsample 
                downsample_idx = (np.arange(1, max_samps+1) * np.floor(trace_len / max_samps) - 1).astype(int)
                the_trace = the_trace[downsample_idx,:]
                trace_time = the_trace[-1,0]
                
            if trace_time >= min_time:
                traces.append(the_trace)
            
        user_datas_.append(traces)
            
    return user_datas_

def normalize_each_data(user_datas, trace_limit = 10): 
    """normalize each trace individually by subtracting off the mean, and dividing by *its* standard dev
    Inputs: 
        - user_datas: output of preproc_data() or of downsample_data(...)
        - trace_limit: minimum number of traces for a user to be included
    Outputs: 
        - user_datas_: normalized user data 
    """
    user_datas_ = []
    for user in range(len(user_datas)): 
        traces = []
        user_vars = []
        for trace in range(len(user_datas[user])): 
            #de-mean
            demean = (user_datas[user][trace][:,1:] - np.mean(user_datas[user][trace][:,1:], axis = 0)).copy()
            variance = np.var(demean, axis = 0)
            if not (variance == 0).any():
                normalize = demean / np.sqrt(variance)
                times = user_datas[user][trace][:,0][:,None].copy()
                traces.append(np.concatenate((times, normalize), axis = 1))
                user_vars.append(variance)
                
        if len(traces) >= trace_limit: 
            user_datas_.append(traces)
        
    return user_datas_

def get_lengthscales(user_datas): 
    """Finds the optimal x and y RBF length scale for each trace of each user
    This helps establish an empirical distribution of 'reasonable' length scales. 
    Note that you may receive SGD convergence warnings on one or two traces. We have 
    found these to be permissible (the resulting fit is still good) 
    Inputs: 
        - user_datas: output of preproc_data(), downsample_data(...), or normalize_each_data(...)
    Outputs: 
        - user_kernels_x: list containing one list per user containing the optimal x-lengthscale for each
          of their traces 

        - user_kernels_y: list containing one list per user containing the optimal y-lengthscale for each
          of their traces 
    """
    user_kernels_x = []
    user_kernels_y = []
    #for each user
    for user in tqdm(range(len(user_datas))): 
        #if > 50 traces, take a subsample of traces (some users have a ton of traces, 
        #and we don't want to overrep.)
        n_traces = len(user_datas[user])
        if n_traces > 50: 
            traces_to_choose = np.arange(50)
        else: 
            traces_to_choose = np.arange(n_traces)
        lenscales_x = []
        lenscales_y = []
        #for each trace
        for trace_idx in traces_to_choose:
            trace = user_datas[user][trace_idx]

            #fit x coordinates
            gp_x = GaussianProcessRegressor(kernel = RBF(length_scale = 8, length_scale_bounds=(1, 40)),
                                           alpha = 0.0025)
            gp_x.fit(trace[:,0][:,None], trace[:,1][:,None])

            #fit y coordinates
            gp_y = GaussianProcessRegressor(kernel = RBF(length_scale = 8, length_scale_bounds=(1, 40)), 
                                           alpha = 0.0025)
            gp_y.fit(trace[:,0][:,None], trace[:,2][:,None])
            
            lenscales_x.append(gp_x.kernel_.get_params()['length_scale'])
            lenscales_y.append(gp_y.kernel_.get_params()['length_scale'])
            
        user_kernels_x.append(lenscales_x)
        user_kernels_y.append(lenscales_y)
    return user_kernels_x, user_kernels_y

def get_effective_lengthscales(user_datas, user_kernels_x, user_kernels_y): 
    """gets array of effective lengthscales (normalized by ave sampling period in trace) 
    for both the x and y coordinates 
    Inputs: 
        - user_datas: output of preproc_data(), downsample_data(...), or normalize_each_data(...)
        - user_kernels_x/y: outputs of get_lengthscales() 
    
    Outputs: 
        - l_eff_x: numpy array containing effective lengthscale of x coordinate 
        - l_eff_y: numpy array containing effective lengthscale of x coordinate 
    """
    #list containing one list per user with the average sampling period of each of their traces
    Ps = []
    for user in user_datas: 
        user_Ps = []
        for trace in user: 
            ave_samp_per = np.median(np.diff(trace[:,0]))
            user_Ps.append(ave_samp_per)
        Ps.append(user_Ps)
    Ps = np.array(Ps)
    
    l_eff_xs = []
    l_eff_ys = []
    for user_idx in range(len(user_kernels_x)): 
        for trace_idx in range(len(user_kernels_x[user_idx])): 
            l_opt_x = user_kernels_x[user_idx][trace_idx]
            l_opt_y = user_kernels_y[user_idx][trace_idx]
            P_sample = Ps[user_idx][trace_idx]
            
            #get effective lengthscale for each trace 
            if (P_sample > 0):
                l_eff_xs.append(l_opt_x / P_sample)
                l_eff_ys.append(l_opt_y / P_sample)
            
    l_eff_x = np.array(l_eff_xs)
    l_eff_y = np.array(l_eff_ys)    

    return l_eff_x, l_eff_y

# create parser to get base directory  
parser = argparse.ArgumentParser()
parser.add_argument("bdir") 
args = parser.parse_args()
base_dir = args.bdir 

#get GeoLife data in numpy arrays 
print('Getting GeoLife data...') 
#user_datas = preproc_data(base_dir) 
user_datas = np.load('./user_datas.npy', allow_pickle=True)

#downsample data to traces around 5m in duration and no more than 50pts in length 
print('Downsampling...') 
user_datas_ds = downsample_data(user_datas, min_time = 240, max_time = 320, max_samps=50)

#normalize mean / variance of data 
#(mean doesn't matter with cond. prior and variance could be estimated from noisy Z) 
#makes kernel fully defined by lengthscale 
print('Normalizing...') 
user_datas_norm =  normalize_each_data(user_datas_ds, trace_limit = 10)

#get x/y lengthscales 
print('Getting lengthscales...') 
user_kernels_x, user_kernels_y = get_lengthscales(user_datas_norm) 

#get effective lengthscales 
print('Getting effective lengthscales...') 
l_eff_xs, l_eff_ys = get_effective_lengthscales(user_datas_norm, user_kernels_x, user_kernels_y)  
l_eff_x = np.array(l_eff_xs)
l_eff_y = np.array(l_eff_ys)

#save lengthscale data to ./saved_data (make directory if it doesn't already exist)
print('Saving effective lengthscales to \'./saved_data/\'...') 
Path("./saved_data").mkdir(parents=True, exist_ok=True)
np.save('./saved_data/l_eff_x.npy', l_eff_x)
np.save('./saved_data/l_eff_y.npy', l_eff_y)
