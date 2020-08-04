# Location Trace Privacy 

Welcome to the anonymized github repo for the NeurIPS paper under reivew: "Location Trace Privacy Under Conditional Priors" 

## Installation 

To make use of this repo yourself, first clone it, and then install [anaconda](https://www.anaconda.com/) (if you do not already have it installed). 
Go to the cloned directory and activate anaconda. Create a new envirnoment using the requirements.txt file: 

    conda create --name location_priv_test --file requirements.txt

Note that the cvxpy package has recently undergone some updates that make the latest editions incompatible with the code used in the submission. As such, we must install cvxpy manually with pip as follows: 

- Check to make sure you are using the version of pip local to this specific anaconda environment: 
    which pip
- You should see something like 
    /anaconda3/envs/location_priv_test/bin/pip
- Install the compatible version of cvxpy to your conda environment using pip 
    pip install cvxpy==1.0.25 

## Usage

There are three .py files related to the paper:    
    - 
