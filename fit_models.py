#!/usr/bin/env python

"""
This script is intended for statistical seasonal forecasting of temperatures in
different regions in Europe. It  uses random sampling and LASSO regression
to build an ensemble of models which are saved in a pickle object to be analyzed later.

All input data is preprocessed inside the READ_DEFINE_AND_PROCESS_EVERYTHING routine. 
"""






import sys, ast, imp, multiprocessing
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.utils import resample

# Directories
basedir = str(sys.argv[4])
in__dir = str(sys.argv[5])
out_dir = str(sys.argv[6])

# Own functions
sys.path.append(basedir)
try:     fcts = imp.reload(fcts)
except:  import functions as fcts

# Unsuccessful fittings flood the log file without this
import warnings; warnings.filterwarnings("ignore")



# Read and process all input data, define variables etc.
data = fcts.READ_DEFINE_AND_PROCESS_EVERYTHING(basedir, in__dir)

print('Fitting models for',data['basename'])



models_out = []
for l,ssn in enumerate(data['seasons']):
    
    print('Training years:',data['trn_yrs'])
    print('Testing years:',data['tst_yrs'])
    
    # Indexes for selecting data from the training and testing periods
    trn_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['trn_yrs']))
    tst_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['tst_yrs']))
    all_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['all_yrs']))
    
    # Fit LassoLarsCV models using the handy BaggingRegressor meta-estimator
    ensemble = fcts.bagging_LassoLarsCV(data['X'].values[trn_idx], data['Y'][data['y_var']].values[trn_idx],
                                        data['vrbl_names'],data['n_smpls'], data['n_jobs'])
    
    # Append the models to the output list, including also the season information
    for i,mdl in enumerate(ensemble):
        models_out.append([ssn, mdl[0], mdl[1], mdl[2], mdl[3]])
    
    print('Completed',len(ensemble),'succesful fittings for', ssn, data['y_var'], data['y_area'])



# Save results into a Pandas pickle object
columns = [ 'Season','Fitted model','Optimal predictors','Indexes of optimal predictors', 
            'N of optimal predictors']


out = pd.DataFrame(data=np.array(models_out), columns=columns)
out.to_pickle(out_dir+data['basename']+'.pkl')




