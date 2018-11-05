#!/usr/bin/env python

"""
This script is intended for statistical seasonal forecasting of temperatures in
different regions in Europe. It  uses random sampling and LASSO regression
to build an ensemble of models which are saved in a pickle object to be analyzed later.

All input data is preprocessed inside the READ_DEFINE_AND_PROCESS_EVERYTHING routine. 
"""



# Data directories
basedir = '/lustre/tmp/kamarain/seasonal_prediction/'
in__dir = '/lustre/tmp/kamarain/netcdf_input/'
out_dir = basedir+'results/'


import sys, ast, imp, multiprocessing
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.utils import resample

sys.path.append(basedir)
try:     fcts = imp.reload(fcts)
except:  import functions as fcts

# Unsuccessful fittings flood the log file without this
import warnings; warnings.filterwarnings("ignore")



# Read and process all input data, define variables etc.
data = fcts.READ_DEFINE_AND_PROCESS_EVERYTHING(basedir, in__dir)

print('Fitting models for',data['basename'])


fitted_models = []
for l,ssn in enumerate(data['seasons']):
    
    print('Training years:',data['trn_yrs'])
    print('Testing years:',data['tst_yrs'])
    
    # Indexes for selecting data from the training and testing periods
    trn_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['trn_yrs']))
    tst_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['tst_yrs']))
    
    # Climatological reference forecast
    climatology = np.full(data['y_eur'][data['y_var']].shape[0], 0)
    
    # Counters
    successful_fittings = 0; sample_count = 0
    
    # Details of sampling
    sample_size = int(data['p_smpl']*len(data['trn_yrs']))
    n_samples   = int(data['n_smpls'])
    max_n_fittings = 3*n_samples
    
    # Continue fitting until either n_samples or max_n_fittings is reached
    while((successful_fittings <= n_samples) & (sample_count <= max_n_fittings)):
        sample_count += 1
        
        # Random sampling with separate fitting and validation periods
        smp_yrs = resample(data['trn_yrs'], replace=False, n_samples=sample_size, random_state=data['rstate']+sample_count) 
        vld_yrs = list(np.array(data['trn_yrs'])[~np.isin(data['trn_yrs'],smp_yrs)])
        vld_yrs.sort(); smp_yrs.sort()
        
        # Indexes for separating the training data to the fitting and validation periods
        smp_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], smp_yrs))
        vld_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], vld_yrs))
        
        # Try fitting a LASSO model
        ftd_mdl, optimal_predictors, opt_prd_idxs, regr_coefs, lambda_from_cv = fcts.search_predictors_Lasso(
                            data['X'].values[smp_idx], data['Y'][data['y_var']].values[smp_idx], 
                            data['vrbl_names'], data['n_folds'], data['n_jobs'])       
        
        if(len(optimal_predictors)>0):
            successful_fittings += 1
            
            # Predict values for the sample, validation, and testing years
            prediction_smp = ftd_mdl.predict(data['X'].values[smp_idx])
            prediction_vld = ftd_mdl.predict(data['X'].values[vld_idx])
            prediction_tst = ftd_mdl.predict(data['X'].values[tst_idx])
            
            # Calculate different validation scores for different sets of years 
            rmse_tr  = fcts.calc_rmse(data['Y'][data['y_var']][smp_idx].values, prediction_smp)
            rmse_vl  = fcts.calc_rmse(data['Y'][data['y_var']][vld_idx].values, prediction_vld)
            rmse_ts  = fcts.calc_rmse(data['Y'][data['y_var']][tst_idx].values, prediction_tst)
            corr_tr  = fcts.calc_corr(data['Y'][data['y_var']][smp_idx].values, prediction_smp)
            corr_vl  = fcts.calc_corr(data['Y'][data['y_var']][vld_idx].values, prediction_vld)
            corr_ts  = fcts.calc_corr(data['Y'][data['y_var']][tst_idx].values, prediction_tst)
            msss_clim_tr = fcts.calc_msss(prediction_smp, data['Y'][data['y_var']][smp_idx].values, climatology[smp_idx])
            msss_clim_vl = fcts.calc_msss(prediction_vld, data['Y'][data['y_var']][vld_idx].values, climatology[vld_idx])
            msss_clim_ts = fcts.calc_msss(prediction_tst, data['Y'][data['y_var']][tst_idx].values, climatology[tst_idx])
            
            # Append results and fitted models to the result list
            fitted_models.append([  ssn, ftd_mdl, sample_count, optimal_predictors, opt_prd_idxs, 
                                    regr_coefs,len(optimal_predictors), lambda_from_cv,
                                    corr_tr, corr_vl, corr_ts, 
                                    rmse_tr, rmse_vl, rmse_ts,
                                    msss_clim_tr, msss_clim_vl, msss_clim_ts])
            
            
            print(data['y_var'], data['y_area'], ssn, sample_count, successful_fittings, corr_ts, corr_vl, corr_tr)

               


# Save results into a Pandas pickle object
columns = [ 'Season','Fitted model','Sample','Optimal predictors','Indexes of optimal predictors', 
            'Regression coefficients','N of optimal predictors', 'Lambda',
            'ACC in fitting sample','ACC in validation sample','ACC in test sample',
            'RMS in fitting sample','RMS in validation sample','RMS in test sample',
            'MSSS_clim in fitting sample','MSSS_clim in validation sample','MSSS_clim in test sample']

out = pd.DataFrame(data=np.array(fitted_models), columns=columns)
out.to_pickle(out_dir+data['basename']+'.pkl')



