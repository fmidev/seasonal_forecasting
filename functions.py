#!/usr/bin/env python



import itertools
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs




# --- Bias correction with quantile mapping ---

def interp_extrap(x, xp, yp):
    """      
    This function is used by q_mapping().
    Projects the x values onto y using the values of 
    [xp,yp], extrapolates conservatively if needed.
    """
    
    y = np.interp(x, xp, yp)
    y[x<xp[ 0]] = yp[ 0] + (x[x<xp[ 0]]-xp[ 0])
    y[x>xp[-1]] = yp[-1] + (x[x>xp[-1]]-xp[-1])
    return y

def q_mapping(obs,ctr,scn,nq):
    """ 
    Quantile mapping. Three (n,) or (n,m) numpy arrays expected for input, 
        where n = time dimension and m = station index or grid cell index.
    
    First argument represents the truth, usually observations.
    
    Second is a sample from a model.
    
    Third is another sample which is to be corrected based on the quantile 
        differences between the two first arguments. Second and third 
        can be the same.
    
    Fourth is the number of quantiles to be used, i.e. the accuracy of correction.

    Linear extrapolation is applied if the third sample contains values 
        outside the quantile range defined from the two first arguments.
    """
    
    # Calculate quantile locations to be used in the next step
    q_intrvl = 100/float(nq); qtl_locs = np.arange(0,100+q_intrvl,q_intrvl) 

    # Calculate quantiles
    q_obs = np.percentile(obs, list(qtl_locs), axis=0)
    q_ctr = np.percentile(ctr, list(qtl_locs), axis=0) 
    
    if(len(obs.shape)==1):
        # Project the data using the correction function 
        return interp_extrap(scn,q_ctr,q_obs)
    
    if(len(obs.shape)==2):
        # Project the data using the correction function, separately for each location 
        out = np.full(scn.shape,np.nan)
        for i in range(out.shape[1]):
            out[:,i] = interp_extrap(scn[:,i],q_ctr[:,i],q_obs[:,i])

        return out








# --- Data analysis ---

def isnumber(a):
    try:
        float(repr(a))
        ans = True
    except:
        ans = False
    return ans

def count_prdctr_freqs(results, column_out):
    """ 
    Count relative frequencies of different items in a Pandas FataFrame object.
    Only one column allowed in the input DataFrame.
    Return a new DataFrame containing frequency results for each item.
    """
    
    from collections import Counter
    
    c = Counter()
    for itm in results: 
        item = itm
        if(isnumber(item)): 
            item = [item]
        
        c.update(Counter(item))
    
    df_out = pd.DataFrame(list(dict(c).items()), columns=[column_out, 'Frequency (%)'])
    df_out = df_out.sort_values(by=['Frequency (%)'], ascending=False)
    df_out['Frequency (%)'] = (df_out['Frequency (%)']/results.shape[0])*100.
    
    return df_out






# --- Evaluation tools ---

def calc_rmse(a,b,*kwargs): 
    return np.sqrt(np.nanmean((a-b)**2))

def calc_corr(a,b,*kwargs): 
    return np.corrcoef(a,b)[0,1]

def calc_msss(fcs,obs,ref):
    mse_f = np.nanmean((fcs-obs)**2)
    mse_r = np.nanmean((ref-obs)**2) 
    return 1 - mse_f/mse_r

def calc_bootstrap(fcs,obs,ref,func, bootstrap_range, L, B):
    """ 
    Calculates moving block bootstrap estimates for an
    evaluation metric defined and calculated inside 'func' argument.

    fcs:                forecasted time series
    obs:                observed time series
    ref:                reference forecast time series, e.g. persistence

    bootstrap_range:    lower and upper percentage points to be estimated
    L:                  length of the bootstrap block
    B:                  number of bootstrap samples

    Returns a list, where items are
      [ <lower boundary value>,    
        <result from evaluation function>,   
        <lower boundary value>,
        <indication of statistical significance of the lower boundary> ] 
    """
    
    from sklearn.utils import resample
    
    idxs = np.arange(len(fcs))
    results = []
    
    random_state = 0
    for smp in range(B):
        block_sample = np.array([]).astype(int)
        while(len(block_sample) < len(fcs)):
            random_state += 1
            rolls = resample(idxs, n_samples=1, random_state=random_state)[0]
            block = np.roll(idxs, rolls)[0:L]
            block_sample = np.append(block_sample, block)

        block_sample = block_sample[0:len(idxs)]
        results.append(func(fcs[block_sample],obs[block_sample],ref[block_sample]))
    
    try:
        out = [ np.percentile(results, bootstrap_range[0]), 
                func(fcs,obs,ref), #np.mean(results), 
                np.percentile(results, bootstrap_range[1])]
    except:
        out = [ np.percentile(results, 2.5), 
                func(fcs,obs,ref), #np.mean(results), 
                np.percentile(results, 97.5)]

    # For indicating the statistical significance 
    # of the lower boundary:
    if(out[0]>0): 
        out.append('*')
    else:
        out.append('')
    
    return out








# --- Manipulators ---

def manipulate_data(ds, var, apply_latw=True, apply_detrending=True):
    """
    (1) Resample to seasonal resolution, 
    (2) remove seasonal cycles,
    (3) weight data so that values near 90N approach zero (if needed),
    (4) transform dimensions from (time, lat, lon) to (time, gridcell),
    (5) reject grid cells containing only NaNs (land areas for SST etc.),
    (6) and remove linear trends from different seasons (if needed)
    """
    if((var=='SD')|(var=='sd')): 
        ds[var] = ds[var].where(ds[var]>=0, other=0.0)
        ds[var] = ds[var].where(ds[var]==0, other=1.0)
        #ds[var].values = Gauss_filter(ds[var].values, (0,3,3))
    
    
    ds = ds.resample(time='3M').mean()
    ds = ds.groupby('time.season') - ds.groupby('time.season').mean('time')
    
    if(apply_latw): ds[var].values = lat_weighting(ds[var].values, 
                                                       ds.lat, ds.lon)
    
    try:
        ds = ds.stack(gridcell=('lat', 'lon')).dropna(dim='gridcell',how='any')
    except: 
        print('Stacking did not work')
    
    if(apply_detrending): ds = remove_trends(ds, var)
    
    return ds

def Gauss_filter(data, sigma=(0,2,2), mode='wrap'):
    """ Smooth data (spatially as default) """   
    import scipy.ndimage.filters as flt
    return flt.gaussian_filter(data, sigma=sigma, mode=mode)

def lat_weighting(data, lat, lon):
    """ Apply latitude weighting """
    rad = np.pi/180.; coslat = np.cos(lat*rad)
    sqrtcos = np.sqrt(coslat); sqrtcos[np.isnan(sqrtcos)] = 0
    zz,weights = np.meshgrid(np.ones(len(lon)),sqrtcos) 
    return weights*data

def remove_trends(ds, var):
    import scipy.signal as sgn
    seasons = ('DJF', 'MAM', 'JJA', 'SON')
    ssn2mon = {'DJF':1, 'MAM':4, 'JJA':7, 'SON':10}
    
    for i,ssn in enumerate(seasons):
        try:    idx = ds['time.season']==ssn
        except: idx = ds.index.month==ssn2mon[ssn]
        ds[var][idx] = sgn.detrend(ds[var][idx].values, axis=0, type='linear')
    
    return ds

def apply_PCA(data, ncomp):
    """ Decomposition of data with principal component analysis """
    import sklearn.decomposition as dc
    
    pca = dc.PCA(n_components=ncomp, whiten=False, svd_solver='full')
    cps = pca.fit_transform(data)
    return cps,pca

def inpaint_nans(data):
    from scipy import interpolate
    
    valid_mask = ~np.isnan(data)
    values = data[valid_mask]
    
    if(len(data.shape)==2):
        coords = np.array(np.nonzero(valid_mask)).T
        it = interpolate.LinearNDInterpolator(coords, values, fill_value='extrapolate')
    
    if(len(data.shape)==1):
        coords = np.nonzero(valid_mask)[0]
        it = interpolate.interp1d(coords, values, fill_value='extrapolate')
        
    return it(list(np.ndindex(data.shape))).reshape(data.shape)






# --- Reading and data extraction --

def bool_index_to_int_index(bool_index):
    return np.where(bool_index)[0]

def read_monthly_indices_from_CLIMEXP(name_of_index):
    """ 
    Try reading various monthly indices from KNMI's Climate Explorer 
    """
    
    import urllib, datetime
    import xarray as xr
    import numpy as np
    import pandas as pd

    name_to_url =   { 
                    'M1i':'http://climexp.knmi.nl/data/iM1.dat',# 1910 ->
                    'M2i':'http://climexp.knmi.nl/data/iM2.dat',# 1910 ->
                    'M3i':'http://climexp.knmi.nl/data/iM3.dat',
                    'M4i':'http://climexp.knmi.nl/data/iM4.dat',# 1910 ->
                    'M5i':'http://climexp.knmi.nl/data/iM5.dat',
                    'M6i':'http://climexp.knmi.nl/data/iM6.dat' # 1910 ->
                    } 

    url_string = name_to_url[name_of_index]

    fp2 = urllib.request.urlopen(url_string)
    data_extracted  = fp2.readlines()
    data_asarray = []
    for row in range(len(data_extracted)): 
        try:    dline = np.array(data_extracted[row].split()).astype(float) 
        except: dline = []
        if(len(dline)>0):
            data_asarray.append(np.array(data_extracted[row].split()).astype(float))

    data = np.array(data_asarray); dates = np.array([])
    data_years = data[:,0].astype(int); 

    if(data.shape[1] > 3): 
        data_tser = data[:,1:13].ravel()
        
        for y in data_years: 
            for m in range(1,13): 
                dates = np.append(dates, datetime.date(y, m, 1))

    if(data.shape[1] <= 3): 
        data_tser = data[:,2]
        
        for row in data: 
            dates = np.append(dates, datetime.date(int(row[0]), int(row[1]), 1))

    data_tser[data_tser<-990] = np.nan
    if(name_of_index=='Volc'):
        data_tser[data_tser==0] = np.nan
        data_tser = np.sqrt(data_tser)

    data_tser[np.isinf(data_tser)] = np.nan
    data_tser = inpaint_nans(data_tser)


    date_range = [1800, 2150]; idxs = np.zeros(dates.shape, bool)
    for i,date in enumerate(dates):
        if((date.year >= date_range[0]) & (date.year <= date_range[1])): idxs[i] = True

    ds = xr.Dataset(data_vars   = {name_of_index:    ('time',data_tser[idxs])}, 
                    coords      = {'time':      dates[idxs].astype(np.datetime64)})
    
    return ds.resample(time='1M').mean()




def prepare_X_array(Y, y_var, X_vars, X_var_definitions, 
                    X_clxp_definitions, include_persistence=False):
    """
    (1) Read CLIMEXP variables, process them to seasonal anomalies, 
            lag them, and append to X
    (2) Extract principal components from preprocessed gridded datasets,
            lag them, and append to X
    """

    from sklearn.preprocessing import StandardScaler, Imputer
    from sklearn.utils import resample
    
    # An empty dataframe containing only dates as an index variable,
    # and probably persistence variables
    X = Y[y_var].to_dataframe().drop(columns=y_var)
    if(include_persistence):
        X['LCL_'+y_var+'-1'] = np.roll(Y[y_var], 1)
        X['LCL_'+y_var+'-2'] = np.roll(Y[y_var], 2)
        X['LCL_'+y_var+'-1'][0:1] = Y[y_var].values[0]
        X['LCL_'+y_var+'-2'][0:2] = Y[y_var].values[0]
    
    # Variables from Climate Explorer (https://climexp.knmi.nl/)
    for i,vr in enumerate(X_clxp_definitions): 
        index = read_monthly_indices_from_CLIMEXP(vr)
        
        index = index.resample(time='3M').mean()
        index = index.groupby('time.season') - index.groupby('time.season').mean('time')
        
        X = pd.merge(X, index.to_dataframe(), left_index=True, right_index=True,  how='left')
        
        for lag in X_clxp_definitions[vr]:
            X[vr+'-'+str(lag)] = np.roll(X[vr], lag)
            X[vr+'-'+str(lag)][0:lag] = X[vr][0]
        
        X = X.drop(columns=vr)
        print(X)
    
    # Remove unnecessary columns
    drops = ['season','season_x','season_y','month_x','month_y','month',y_var]
    for i,dr in enumerate(drops):
        try:    X = X.drop(columns=dr)
        except: pass

    # Variables from gridded datasets
    for i,vr in enumerate(X_var_definitions):
        
        data_for_pca = X_vars[i].values
        cps, pca = apply_PCA(data_for_pca,X_var_definitions[vr][1])
        cps = pca.transform(X_vars[i].values)
        
        for cpn in range(X_var_definitions[vr][1]):
            vrb_name = vr+str(cpn+1)
            X.loc[:,vrb_name] = cps[:,cpn]
            
            for lag in X_var_definitions[vr][2]:
                X[vrb_name+'-'+str(lag)] = np.roll(X[vrb_name], lag)
                X[vrb_name+'-'+str(lag)][0:lag] = X[vrb_name][0]
            
            X = X.drop(columns=vrb_name)
    
    # Fill (potential) holes with mean values and normalize to N(0,1) 
    X[:] = Imputer().fit_transform(X.values)
    X[:] = StandardScaler().fit_transform(X.values)

    return X 


def read_manipulate_Y_data(y_var, in__dir, year_range, y_area):
    
    name2code = {   
                    'MSL':['msl', 'msl_era20c_monthly_1900-2010.nc'],
                    'T2M':['t2m','te2m_era20c_monthly_1900-2010.nc']
                } 

        
    y_eur = read_and_select(in__dir+name2code[y_var][1], name2code[y_var][0], year_range, y_area)
        
    # Manipulate
    y_eur = manipulate_data(y_eur, name2code[y_var][0],  apply_latw=False, apply_detrending=True)
    y_eur[y_var] = y_eur[name2code[y_var][0]]
    Y = y_eur.mean('gridcell')

    return y_eur, Y




def read_manipulate_X_data(in__dir, year_range, X_var_definitions):
    
    X_vars = []
    name2code = {   
                    'TCW':['tcw', 'tcw_era20c_monthly_1900-2010.nc'],
                    'SMO':['sum', 'smo_era20c_monthly_1900-2010.nc'],
                    'SST':['sst',  'HadISST_sst.nc'                ], # 'SSTK','sst_era20c_monthly_1900-2010.nc'],
                    'SIC':['sic',  'HadISST_ice.nc'                ], # 'CI', 'aice_era20c_monthly_1900-2010.nc'],
                    'SNC':['sd',  'snw_era20c_monthly_1900-2010.nc'],
                    'Z70':['z',   'z70_era20c_monthly_1900-2010.nc'],
                    'MSL':['msl', 'msl_era20c_monthly_1900-2010.nc'],
                    'T2M':['t2m','te2m_era20c_monthly_1900-2010.nc'],
                } 
    
    for i, vrb in enumerate(X_var_definitions): 
        
        data = read_and_select(in__dir+name2code[vrb][1], name2code[vrb][0], 
                                year_range, X_var_definitions[vrb][0])

        if((vrb=='SIC')|(vrb=='SNC')): 
            data = manipulate_data(data, name2code[vrb][0],  apply_latw=True, apply_detrending=False)
        else:
            data = manipulate_data(data, name2code[vrb][0],  apply_latw=True, apply_detrending=True)
        
        X_vars.append(data[name2code[vrb][0]])
   

    return X_vars




def read_and_select(fle, var, year_range, area):
    """ 
    (1) Transform longitudes from [0,360] to [-180,180], 
    (2) reverse latitudes (if needed), 
    (3) and select area of interest
    """
    
    try:
        ds = xr.open_dataset(fle).sel(
                time=slice(year_range[0]+'-01-01', year_range[1]+'-12-31'))
    except: 
        ds = xr.open_dataset(fle)
    
    try:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'}) 
    except: 
        pass

    
    if(ds.lon.values.max() > 350):
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        rolls = np.sum(ds.lon.values < 0); ds = ds.roll(lon=rolls*(-1))

    if(ds.lat.values[0] > ds.lat.values[-1]):
        ds['lat'] = np.flipud(ds['lat'])
        ds[var].values = np.flip(ds[var], axis=1)

    if((var=='sst')|(var=='sic')): 
        mask = ds[var].values == -1000.
        ds[var].values[mask] = np.nan
    
    if(  area=='europe'): ds = ds.squeeze().sel(lat=slice( 33,73), lon=slice(-12,40)) 
    elif(area=='wsteur'): ds = ds.squeeze().sel(lat=slice(42,59),  lon=slice(-10,17))
    elif(area=='easeur'): ds = ds.squeeze().sel(lat=slice(38,56),  lon=slice(17,43))
    elif(area=='meditr'): ds = ds.squeeze().sel(lat=slice(30,45),  lon=slice(0,25))
    elif(area=='scandi'): ds = ds.squeeze().sel(lat=slice( 55,71), lon=slice(  4,34)) 
    elif(area=='norhem'): ds = ds.squeeze().sel(lat=slice(0,87)) 
    elif(area=='norpol'): ds = ds.squeeze().sel(lat=slice( 50,87))
    else: ds = ds.squeeze()
    
    return ds



def READ_DEFINE_AND_PROCESS_EVERYTHING(basedir, in__dir):
    """
    Reads and processes all input data.
    Defines variables and gives values for them.
    Puts everything into one dictionary and returns it.
    
    y_var:          Name of the predictand
    y_area:         Target domain

    rstate:         Seed for the random number generator
    yr1:            First year to be used
    yr2:            Last year to be used
    n_folds:        Number of folds in cross-validation
    p_smpl:         Proportion of data used for fitting in the training years
    n_smpls:        Number of random samples
    tst_len:        Number of years selected for testing

    ncomps_sst:     Number of PCs for SST
    ncomps_snc:     Number of PCs for SNC

    lags_sst:       Lags for SST
    lags_snc:       Lags for SNC

    experiment:     #'CONTROL' #'INCLPERSIS' #'DETREND' #'CUTFIRSTYRS' #'SKIPQMAP'
    """

    import sys, ast, multiprocessing
    import pandas as pd
    import numpy as np

    # Output dictionanry
    dc = {}
    
    # Read command line arguments
    try:
        y_var       = dc['y_var']       = str(sys.argv[1])  
        y_area      = dc['y_area']      = str(sys.argv[2])
        experiment  = dc['experiment']  = str(sys.argv[3])
    except:
        y_var       = dc['y_var']       = 'T2M'
        y_area      = dc['y_area']      = 'europe'
        experiment  = dc['experiment']  = 'CONTROL'
    
    # Define details    
    if(experiment=='CUTFIRSTYRS'):         
        yr1 = dc['yr1']             = '1935'
    else:
        yr1 = dc['yr1']             = '1900' 
          
    yr2 = dc['yr2']                 = '2010'
    rstate = dc['rstate']           = 70               
    n_folds = dc['n_folds']         = 5
    p_smpl = dc['p_smpl']           = 0.5
    n_smpls = dc['n_smpls']         = 1000
    tst_len = dc['tst_len']         = 25

    ncomps_sst = dc['ncomps_sst']   = 6
    ncomps_snc = dc['ncomps_snc']   = 6

    if(experiment=='NO_LAGS'):
        lags_sst = dc['lags_sst']   = (1,)
        lags_snc = dc['lags_snc']   = (1,)
    else:
        lags_sst = dc['lags_sst']   = (1,2,3,4,5)
        lags_snc = dc['lags_snc']   = (1,2)
    
 
    n_jobs = dc['n_jobs'] = np.min([28, int(0.9*(multiprocessing.cpu_count()))])
    seasons = dc['seasons'] = ('DJF', 'MAM' ,'JJA', 'SON')

    # Define a skeleton for naming output files
    basename = dc['basename'] = 'fittings_'+experiment+'_'+y_var+'_'+y_area+ \
                '_nsmpls'+str(n_smpls)+'_ntestyrs'+str(tst_len)+ \
                '_SST'+str(ncomps_sst)+'-'+str(lags_sst[-1])+ \
                '_SNC'+str(ncomps_snc)+'-'+str(lags_snc[-1])+'_'+yr1+'-'+yr2

    # Variables from ERA-20C and HadISST1, form: 'name_of_variable': ['domain', n_comps, lags, year_range]
    X_var_definitions = dc['X_var_definitions'] =       {
               'SST': ['global', ncomps_sst, lags_sst, (yr1, yr2)],
               'SNC': ['norhem', ncomps_snc, lags_snc, (yr1, yr2)],
                                                        }

    # Optional variables from https://climexp.knmi.nl/, form: 'name_of_index': [lags]
    X_clxp_definitions = dc['X_clxp_definitions'] =     {
                #'M1i':(1,), 'M2i':(1,), 'M3i':(1,), 
                #'M4i':(1,), 'M5i':(1,), 'M6i':(1,)
                                                        }

    # Read and preprocess the predictand data using xarray etc.
    y_eur, Y = dc['y_eur'], dc['Y'] = read_manipulate_Y_data(y_var, in__dir, (yr1, yr2), y_area)

    # Read and preprocess the raw predictor data using xarray etc.
    X_vars = dc['X_vars'] = read_manipulate_X_data(in__dir, (yr1, yr2), X_var_definitions)

    # Compress raw data with PCA, apply lagging, and create a Pandas dataframe
    if(experiment=='INCLPERSIS'):
        X = dc['X'] = prepare_X_array(Y, y_var, X_vars, X_var_definitions, X_clxp_definitions, include_persistence=True)
    else:
        X = dc['X'] = prepare_X_array(Y, y_var, X_vars, X_var_definitions, X_clxp_definitions, include_persistence=False)

    # Extract variable names
    vrbl_names = dc['vrbl_names'] = X.columns

    # Define training and test periods
    all_yrs = dc['all_yrs'] = list(np.arange(Y['time.year'][0],Y['time.year'][-1]).astype(int))
    tst_yrs = dc['tst_yrs'] = all_yrs[-tst_len:] 
    trn_yrs = dc['trn_yrs'] = list(np.array(all_yrs)[~np.isin(all_yrs,tst_yrs)])

    return dc 







# --- Model fitting ---

def bagging_LassoLarsCV(X, Y, vrbl_names, n_estimators, p_smpl, n_jobs):
    
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import LassoLarsCV
    
    max_n_estimators = int(3*n_estimators)
    cv = RepeatedKFold(n_splits=5, n_repeats=3)
    #cv = KFold(n_splits=5, shuffle=True)
    eps = 2e-12
    
    try: X = X.values
    except: pass
    try: Y = Y.values
    except: pass
    
    X = np.squeeze(X)
    Y = np.squeeze(Y)


    fitted_ensemble = BaggingRegressor(
                    base_estimator=LassoLarsCV(cv=cv, eps=eps, n_jobs=1),
                    n_estimators=max_n_estimators, 
                    max_samples=p_smpl, # Select 50% of training data per random sample
                    bootstrap=False,    # Sampling without replacement
                    oob_score=False,    # 
                    n_jobs=n_jobs,      #8,
                    random_state=70,
                    verbose=1).fit(X, Y) 
    
    
    # Definition of success in fitting: at least one predictor
    # needs to be found
    succesful_fittings = 0; i=0; final_ensemble = []
    while((succesful_fittings < n_estimators) & (i < max_n_estimators)):
        
        estimator = fitted_ensemble.estimators_[i]; i+=1
        predictor_indices = np.abs(estimator.coef_) > 0
        
        if(predictor_indices.sum() > 0):
            estimator_predictors = vrbl_names[predictor_indices]
            n_predictors = len(estimator_predictors)
            
            # Append results and fitted models to the result list
            final_ensemble.append([estimator, estimator_predictors, 
                                    predictor_indices, n_predictors])
            
            succesful_fittings += 1
    
    
    return final_ensemble


def boosting_LassoLarsCV(X, Y, vrbl_names, n_estimators, n_jobs):
    
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.linear_model import LassoLarsCV
    
    max_n_estimators = int(3*n_estimators)
    cv = RepeatedKFold(n_splits=5, n_repeats=3)
    #cv = KFold(n_splits=5, shuffle=True)
    eps = 2e-12
    
    try: X = X.values
    except: pass
    try: Y = Y.values
    except: pass
    
    X = np.squeeze(X)
    Y = np.squeeze(Y)

    fitted_ensemble = AdaBoostRegressor(
                    base_estimator=LassoLarsCV(cv=cv, eps=eps, n_jobs=n_jobs),
                    n_estimators=max_n_estimators, 
                    random_state=70).fit(X, Y) 
    
    final_ensemble = []
    for i, estimator in enumerate(fitted_ensemble.estimators_):
        predictor_indices = np.abs(estimator.coef_) > 0
        
        if(predictor_indices.sum() > 0):
            estimator_predictors = vrbl_names[predictor_indices]
            n_predictors = len(estimator_predictors)
            
            # Append results and fitted models to the result list
            final_ensemble.append([estimator, estimator_predictors, 
                                    predictor_indices, n_predictors])
    
    
    return final_ensemble







# --- Plotting ---

class LowerThresholdRobinson(ccrs.Robinson):   
    @property
    def threshold(self):
        return 1e3

class LowerThresholdOrtho(ccrs.Orthographic):  
    @property
    def threshold(self):
        return 1e3

class LowerThresholdLConf(ccrs.LambertConformal):
    @property
    def threshold(self):
        return 1e3

def plot_rectangle(ax,trans, lonmin,lonmax,latmin,latmax,clr,lw,alp):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    ax.plot(xs, ys,transform=trans,color=clr,linewidth=lw,alpha=alp)
    pass

def plot_scatter(ax,trans, lons,lats,clr1,clr2,sze,mrk,alp):
    ax.scatter(lons, lats,transform=trans,s=sze,marker=mrk,
                edgecolors=clr2,c=clr1,linewidth=0.3,alpha=alp)
    pass

def plot_text(ax,trans,lons,lats,txt,clr,fs,rot,box):
    
    font = {'family': 'sans-serif',
            'color':  clr,
            'weight': 'black',
            'style': 'italic',
            'size': fs,
            'rotation':rot,
            'ha':'center',
            'va':'center',
            }
    
    for i in range(len(txt)):
        if(box): 
            bbox_props = dict(boxstyle="square,pad=0.1", fc="w", ec="k", lw=1)
            ax.text(lons[i],lats[i],txt[i],transform=trans,fontdict=font,bbox=bbox_props)
        else:
            ax.text(lons[i],lats[i],txt[i],transform=trans,fontdict=font)





