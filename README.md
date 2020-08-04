# Location Trace Privacy 

Welcome to the anonymized github repo for the NeurIPS paper under reivew: "Location Trace Privacy Under Conditional Priors" 

## Installation 

To make use of this repo yourself, first clone it, and then install [anaconda](https://www.anaconda.com/) (if you do not already have it installed). 
Go to the cloned directory and activate anaconda. Create a new envirnoment using the requirements.txt file, and activate it: 

    conda create --name location_priv_test --file requirements.txt
    conda activate location_priv_test

Note that the cvxpy package has recently undergone some updates that make the latest editions incompatible with the code used in the submission. As such, we must install cvxpy manually with pip as follows: 

- Check to make sure you are using the version of pip local to this specific anaconda environment: 

        which pip

- You should see something like 

        /anaconda3/envs/location_priv_test/bin/pip

- Install the compatible version of cvxpy to your conda environment using pip 

        pip install cvxpy==1.0.25 

## Usage

There are three .py files related to the paper:

- preprocess_geolife.py: this produces the effective lengthscales of the location trace data (GeoLife) l_eff_x.npy and l_eff_y.npy seen in saved_data/  
- preprocess_SML2010.py: this produces the effective lengthscales of the home temperature data (SML2010) l_eff_temp.npy seen in saved_data/  
- figure_3.py: This runs all experiments and produces all figures seen in Figure 3 of the paper. 

The preprocessed data is relatively small in size, and is included with the github repo. As such, it is not necessary to run either of the preprocess_xxx.py files to run the experiments. You may simply enter 

    python3 figure_3.py 

