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

def manipulate_data(ds, var, predef_clim, predef_trnd, trn_yrs, all_yrs, 
                    apply_latw=True, apply_detrending=True, dropna=True):
    """
    (1) Resample to seasonal resolution, 
    (2) remove seasonal cycles,
    (3) weight data so that values near 90N approach zero (if needed),
    (4) transform dimensions from (time, lat, lon) to (time, gridcell),
    (5) optionally reject grid cells containing only NaNs (land areas for SST etc.),
    (6) and remove linear trends from different seasons
    """

    
    if((var=='SD')|(var=='sd')|(var=='snowc')): 
        ds[var] = ds[var].where(ds[var]>=0, other=0.0)
        ds[var] = ds[var].where(ds[var]==0, other=1.0)
        #ds[var].values = Gauss_filter(ds[var].values, (0,3,3))
    
    """
    if((var=='hgt')|(var=='z')|(var=='GPT')):
        months = ds.time.to_index().month; ssn_ends = (months==2)|(months==5)|(months==8)|(months==11)
        ds = ds.sel(time=ssn_ends)
    else: 
        ds = ds.resample(time='3M').mean()
    """
        
    ds = ds.resample(time='3M').mean()

    ds = ds.sel(time=slice(str(all_yrs[0])+'-01-01', str(all_yrs[-1])+'-12-31')) 
    
    try: 
        clim = predef_clim
        ds = ds.groupby('time.season') - clim
        print('Predefined climatology used')
    except:
        clim = ds.sel(time=slice(str(trn_yrs[0])+'-01-01', str(trn_yrs[-1])+'-12-31')).groupby('time.season').mean('time')
        ds = ds.groupby('time.season') - clim
        print('Climatology calculated from data')
    
    if(apply_latw): ds[var].values = lat_weighting(ds[var].values, 
                                                       ds.lat, ds.lon)
    if(dropna):
        ds = ds.stack(gridcell=('lat', 'lon')).dropna(dim='gridcell',how='any')
    else: 
        ds = ds.stack(gridcell=('lat', 'lon')).fillna(0)
    
    if(apply_detrending): 
        ds = ds.load()
        trend_models = { }
        for ssn in ('DJF', 'MAM', 'JJA', 'SON'):
            #ssn_idx = ds['time.season'] == ssn
            
            trn_idx = bool_index_to_int_index(np.isin(ds['time.season'], ssn) & np.isin(ds['time.year'], trn_yrs))
            all_idx = bool_index_to_int_index(np.isin(ds['time.season'], ssn) & np.isin(ds['time.year'], all_yrs))
            
            trn_x = np.array(ds.time[trn_idx].values.tolist()).reshape(-1,1)
            all_x = np.array(ds.time[all_idx].values.tolist()).reshape(-1,1)
            try:
                trend = predef_trnd[ssn].predict(all_x)
                trend_models[ssn] = predef_trnd[ssn]
                print('Predefined trend model used')
            except:
                #_, trend_model = define_trends(ds[var][trn_idx], trn_x)
                _, trend_model = define_trends(ds[var][all_idx], all_x)
                trend = trend_model.predict(all_x)
                trend_models[ssn] = trend_model
                print('Trends calculated from data')
            
            ds[var][all_idx] = ds[var][all_idx] - trend
            

                
    return ds, clim, trend_models

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


def define_trends(data, x):
    """ Calculates linear trends.
        Assumes data to be of shape (time, gridcell). """
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression().fit(x, data)
    trend = model.predict(x)
    
    return trend, model

def apply_PCA(data, ncomp):
    """ Decomposition of data with principal component analysis. 
        Assumes data to be of shape (time, gridcell). """
    import sklearn.decomposition as dc
    
    pca = dc.PCA(n_components=ncomp, whiten=False, svd_solver='full')
    cps = pca.fit_transform(data)
    svl = pca.singular_values_
    return cps,pca,svl

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
                    'M6i':'http://climexp.knmi.nl/data/iM6.dat', # 1910 ->
                    'NAO':'http://climexp.knmi.nl/data/inao.dat', # 1821 ->
                    'NINO12':'http://climexp.knmi.nl/data/inino2.dat',
                    'NINO3': 'http://climexp.knmi.nl/data/inino3.dat',
                    'NINO34':'http://climexp.knmi.nl/data/inino5.dat',
                    'NINO4': 'http://climexp.knmi.nl/data/inino4.dat',
                    'AMO1':  'http://climexp.knmi.nl/data/iamo_hadsst.dat',
                    'AMO2':  'http://climexp.knmi.nl/data/iamo_hadsst_ts.dat',
                    'PDO1':  'http://climexp.knmi.nl/data/ipdo.dat',
                    'PDO2':  'http://climexp.knmi.nl/data/ipdo_hadsst3.dat',
                    'SOI':   'http://climexp.knmi.nl/data/isoi.dat',
                    } 

    url_string = name_to_url[name_of_index]
    try:
        fp2 = urllib.request.urlopen(url_string)
        data_extracted  = fp2.readlines()
    except: 
        pass
    
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




def prepare_X_array(Y, y_var, X_vars, predef_pcas, X_var_definitions, 
                    X_clxp_definitions, include_persistence=False):
    """
    (1) Read CLIMEXP variables, process them to seasonal anomalies, 
            lag them, and append to X
    (2) Extract principal components from preprocessed gridded datasets,
            lag them, and append to X
    """
    
    from sklearn.preprocessing import StandardScaler #, Imputer
    from sklearn.impute import SimpleImputer
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
    
    # Remove unnecessary columns
    drops = ['season','season_x','season_y','month_x','month_y','month',y_var]
    for dr in drops:
        try:    X = X.drop(columns=dr)
        except: pass

    # Variables from gridded datasets
    X_pcas, X_eigs, X_errs = {}, {}, {}
    for i,vr in enumerate(X_var_definitions):
        
        trn_yrs = X_var_definitions[vr][3]
        all_yrs = X_var_definitions[vr][4]

        trn_idx = bool_index_to_int_index(np.isin(X_vars[i]['time.year'], trn_yrs))
        all_idx = bool_index_to_int_index(np.isin(X_vars[i]['time.year'], all_yrs))
        
        try:
            cps = predef_pcas[vr].transform(X_vars[i].values)
            _, pca, svl = predef_pcas[vr], -99, -99
            print('Predefined PCA model used')
        except:
            #_, pca, svl = apply_PCA(X_vars[i].values[trn_idx],X_var_definitions[vr][1])
            cps, pca, svl = apply_PCA(X_vars[i].values,X_var_definitions[vr][1])
            #cps = pca.transform(X_vars[i].values)
            print('PCAs calculated from data')
        
        # Sampling uncertainty according to North et al. (1982)
        eig = svl**2        
        N = X_vars[i].values.shape[0]
        sampling_err = eig*(2/N)**(0.5)
        
        X_pcas[vr] = pca; X_eigs[vr] = eig; X_errs[vr] = sampling_err
        
        for cpn in range(X_var_definitions[vr][1]):
            vrb_name = vr+str(cpn+1)
            X.loc[:,vrb_name] = cps[:,cpn]
            
            
            for lag in X_var_definitions[vr][2]:
                X[vrb_name+'-'+str(lag)] = np.roll(X[vrb_name], lag)
                X[vrb_name+'-'+str(lag)][0:lag] = X[vrb_name][0]
            
            X = X.drop(columns=vrb_name)
    
    # Fill (potential) holes with mean values and normalize to N(0,1) 
    X[:] = SimpleImputer().fit_transform(X.values)
    X[:] = StandardScaler().fit_transform(X.values)

    return X, X_pcas, X_eigs, X_errs


def read_manipulate_Y_data(y_var, in__dir, predef_clim, predef_trnd, trn_yrs, all_yrs, y_area):
    
    name2code = {   
                    #'MSL':['msl', 'msl_era20c_monthly_1900-2010.nc'],
                    #'T2M':['t2m','te2m_era20c_monthly_1900-2010.nc']
                    #'T2M':['air','air.2m.mon.mean.nc'],
                    'T2M':['temperature_anomaly','HadCRUT.4.6.0.0.median.nc'],
                } 

        
    y_eur = read_and_select(in__dir+name2code[y_var][1], name2code[y_var][0], y_area)
    
    # Manipulate
    y_eur, clim, trend_models = manipulate_data(y_eur, name2code[y_var][0], 
                            predef_clim, predef_trnd, trn_yrs, all_yrs,  
                            apply_latw=False, apply_detrending=True, dropna=False)
    
    y_eur[y_var] = y_eur[name2code[y_var][0]]
    Y = y_eur.mean('gridcell')

    return y_eur, Y, clim, trend_models




def read_manipulate_X_data(in__dir, X_var_definitions, X_clim, X_trnd):
    
    X_vars = []
    name2code = {   
                    'ERA-20C':  {
                        #'SST':['sst',  'HadISST_sst.nc'                ],
                        'SST':['sst', 'sst_era20c_monthly_1900-2010.nc'],
                        'SNC':['sd',  'snw_era20c_monthly_1900-2010.nc'],
                        'GPT':['z',   'z15_era20c_monthly_1900-2010.nc'],
                        'MSL':['msl', 'msl_era20c_monthly_1900-2010.nc'],
                                },
                                
                    '20CRv2c':  {
                        'SST':['sst',   'sst.mon.anom.nc' ], 
                        'GPT':['hgt',   'hgt.mon.mean.nc'],
                        'SNC':['snowc', 'snowc.mon.mean.nc'],
                                },
                    
                    'ERA-Int':  {
                        'SST':['sst',   'sst_eraint_monthly_1979-2018.nc'], 
                        'GPT':['z',     'z15_eraint_monthly_1979-2018.nc'],
                                }
                } 
    
    for i, vrb in enumerate(X_var_definitions): 
        
        trn_yrs = X_var_definitions[vrb][3]
        all_yrs = X_var_definitions[vrb][4]
        
        X_src = X_var_definitions[vrb][5]
            
        data_raw = read_and_select(in__dir+name2code[X_src][vrb][1], name2code[X_src][vrb][0], 
                                        X_var_definitions[vrb][0])
        
        try:
            predef_clim, predef_trnd = X_clim[vrb], X_trnd[vrb]
        except:
            predef_clim, predef_trnd = {}, {}
        
        data_mnp, clim, trend_models = manipulate_data(data_raw, name2code[X_src][vrb][0], 
                                                predef_clim, predef_trnd, trn_yrs, all_yrs, 
                                                apply_latw=True, apply_detrending=True, dropna=False)
        
        X_vars.append(data_mnp[name2code[X_src][vrb][0]])
        X_clim[vrb] = clim
        X_trnd[vrb] = trend_models
    
    return X_vars, X_clim, X_trnd




def read_and_select(fles, var, area):
    """ 
    (1) Transform longitudes from [0,360] to [-180,180], 
    (2) reverse latitudes (if needed), 
    (3) and select area of interest
    """
    
    ds = xr.open_mfdataset(fles)
    
    # For 20CRv2c geopotential height   
    if(var=='hgt'): 
        ds = ds.sel(level=150.0)
    
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

    # For 20CRv2c snow cover
    if(var=='snowc'): 
        ds[var] = ds[var]/100.
        ds[var] = ds[var].where(ds[var]>=0.5, other=0.0)
        ds[var] = ds[var].where(ds[var] <0.5, other=1.0)
    
    # For HadISST1
    if((var=='sst')|(var=='sic')): 
        mask = ds[var].values == -1000.
        ds[var].values[mask] = np.nan
    
    if(  area=='europe'): ds = ds.squeeze().sel(lat=slice( 33,73), lon=slice(-12,40)) 
    elif(area=='westeu'): ds = ds.squeeze().sel(lat=slice(42,59),  lon=slice(-10,17))
    elif(area=='easeur'): ds = ds.squeeze().sel(lat=slice(38,56),  lon=slice(17,43))
    elif(area=='meditr'): ds = ds.squeeze().sel(lat=slice(30,45),  lon=slice(0,25))
    elif(area=='scandi'): ds = ds.squeeze().sel(lat=slice( 55,71), lon=slice(  4,34)) 
    elif(area=='norhem'): ds = ds.squeeze().sel(lat=slice(-10,87)) 
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

    experiment:     #'CONTROL', 'INCLPERSIS', ... 
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
        X_source    = dc['X_source']    = str(sys.argv[4]) 
    except:
        y_var       = dc['y_var']       = 'T2M'
        y_area      = dc['y_area']      = 'scandi'
        experiment  = dc['experiment']  = 'CONTROL'
        X_source    = dc['X_source']    = 'ERA-20C'
        
    # Define details    
    if(experiment=='CUTFIRSTYRS'):         
        yr1 = dc['yr1']             = '1945' #'1935' #'1940' #'1945'
    else:
        yr1 = dc['yr1']             = '1915' #'1900' #'1915' 
          
    yr2 = dc['yr2']                 = '2010'
    
    
    rstate = dc['rstate']           = 70               
    n_folds = dc['n_folds']         = 5
    p_smpl = dc['p_smpl']           = 0.5
    n_smpls = dc['n_smpls']         = 1000
    tst_len = dc['tst_len']         = 25

    ncomps_sst = dc['ncomps_sst']   = 5
    ncomps_snc = dc['ncomps_snc']   = 3
    ncomps_gpt = dc['ncomps_gpt']   = 3

    if(experiment=='NO_LAGS'):
        lags_sst = dc['lags_sst']   = (1,)
        lags_snc = dc['lags_snc']   = (1,)
        lags_gpt = dc['lags_gpt']   = (1,)
    else:
        lags_sst = dc['lags_sst']   = (1,2,3,4,5)
        lags_snc = dc['lags_snc']   = (1,2)
        lags_gpt = dc['lags_gpt']   = (1,2)
        
    
 
    n_jobs = dc['n_jobs'] = np.min([28, int(0.9*(multiprocessing.cpu_count()))])
    seasons = dc['seasons'] = ('DJF', 'MAM' ,'JJA', 'SON')

    # Define training and test periods
    all_yrs = dc['all_yrs'] = list(np.arange(int(yr1),int(yr2)+1))
    tst_yrs = dc['tst_yrs'] = all_yrs[-tst_len:] 
    trn_yrs = dc['trn_yrs'] = list(np.array(all_yrs)[~np.isin(all_yrs,tst_yrs)])

    # Define a skeleton for naming output files             
    basename = dc['basename'] = 'fittings_'+experiment+'_HadCRUT4-'+y_var+ \
                '_nsmpls'+str(n_smpls)+'_ntestyrs'+str(tst_len)+ \
                '_'+X_source+'-SST'+str(ncomps_sst)+'-'+str(lags_sst[-1])+ \
                '_'+X_source+'-GPT'+str(ncomps_gpt)+'-'+str(lags_gpt[-1])+ \
                '_'+yr1+'-'+yr2+'_'+y_area

    # Variables, form: 'name_of_variable': ['domain', n_comps, lags, year_range]
    X_var_definitions = dc['X_var_definitions'] =       {
               'SST': ['global', ncomps_sst, lags_sst, trn_yrs, all_yrs, X_source],
               'GPT': ['norhem', ncomps_gpt, lags_gpt, trn_yrs, all_yrs, X_source],
               #'SNC': ['norhem', ncomps_snc, lags_snc, trn_yrs, all_yrs, X_source],
                                                        }
    
    # Optional variables from https://climexp.knmi.nl/, form: 'name_of_index': [lags]
    X_clxp_definitions = dc['X_clxp_definitions'] =     {
                #'M1i':(1,), 'M2i':(1,), 'M3i':(1,), 
                #'M4i':(1,), 'M5i':(1,), 'M6i':(1,),
                #'NAO':(1,), 'NINO12':(1,), 'NINO3':(1,), 'NINO34':(1,), 'NINO4',:(1,),
                #'AMO1':(1,), 'AMO2':(1,), 'PDO1':(1,), 'PDO2':(1,), 'SOI',:(1,),
                                                        }

    # Read and preprocess the predictand data using xarray etc.
    y_eur, Y, cl, tr = dc['y_eur'], dc['Y'], dc['Y_clim'], dc['Y_trend'] = \
        read_manipulate_Y_data(y_var, in__dir, {}, {}, all_yrs, all_yrs, y_area)

    # Read and preprocess the raw predictor data using xarray etc.
    X_vars, cl, tr = dc['X_vars'], dc['X_clim'], dc['X_trnd'] = \
        read_manipulate_X_data(in__dir, X_var_definitions, {}, {})


    if(experiment=='INCLPERSIS'): 
        include_persistence=True
    else:
        include_persistence=False

    # Compress raw data with PCA, apply lagging, and create a Pandas dataframe        
    X,p,ei,er = dc['X'], dc['X_PCAs'], dc['X_EIGs'], dc['X_ERRs'] = prepare_X_array(Y, 
                y_var, X_vars, {}, X_var_definitions, X_clxp_definitions, include_persistence=include_persistence)

    if(experiment=='FOLLAND'):
        # Folland et al. 2012, Hall et al. 2017
        for i,vrb in enumerate(X.columns):
            if((vrb[0:4] == 'SST1')|(vrb[0:4] == 'sst1')):
                X[vrb] = StandardScaler().fit_transform(X[vrb][:,np.newaxis])
                X[vrb][ np.abs(X[vrb]) < 1 ] = 0
                X[vrb][ X[vrb] < -1 ] = -1
                X[vrb][ X[vrb] > 1.75 ] = 0
                X[vrb][ X[vrb] > 1 ] = 1
                print(X[vrb])

    # Extract variable names
    vrbl_names = dc['vrbl_names'] = X.columns
    
    return dc 







# --- Model fitting ---

def bagging_LassoLarsCV(X, Y, vrbl_names, n_estimators, p_smpl, n_jobs, max_n_estimators):
    
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import LassoLarsCV, LinearRegression
    
    cv = KFold(n_splits=5, shuffle=True)
    
    
    try: X = X.values
    except: pass
    try: Y = Y.values
    except: pass
    
    X = np.squeeze(X)
    Y = np.squeeze(Y)
    
    max_feats = int(X.shape[1]/3)
    eps = 2e-10

    fitted_ensemble = BaggingRegressor(
                    base_estimator=LassoLarsCV(cv=cv, eps=eps, max_iter=200, n_jobs=1),
                    #base_estimator=LinearRegression(n_jobs=1),
                    n_estimators=max_n_estimators, # Number of fittings
                    max_samples=0.5,   # Select 50% of training data per random sample
                    max_features=max_feats,   # Select N/3 variables randomly
                    bootstrap=False,   # 
                    bootstrap_features=False,
                    oob_score=False,
                    n_jobs=n_jobs,    #8,
                    random_state=70,
                    verbose=1).fit(X, Y) 
    
    
    all_sample_indices = np.arange(X.shape[0])
    feature_indices = fitted_ensemble.estimators_features_
    sample_indices  = fitted_ensemble.estimators_samples_
    outofs_indices  = []
    
    for i,smp in enumerate(sample_indices):
        out_sample = all_sample_indices[~np.isin(all_sample_indices, smp)]
        outofs_indices.append(out_sample)

    final_ensemble = []
    for i, estimator in enumerate(fitted_ensemble.estimators_):
        f_indices = feature_indices[i] 
        s_indices = sample_indices[i] 
        o_indices = outofs_indices[i] 
        a_indices = all_sample_indices
        true_indices = np.abs(estimator.coef_)>0
        
        # Definition of success in fitting: at least one predictor
        # needs to be found
        if(true_indices.sum() > 0):
            estimator_predictors = vrbl_names[f_indices][true_indices]
            n_predictors = true_indices.sum() 
            
            all_sample_score = calc_corr(Y[a_indices], estimator.predict(X[a_indices][:, f_indices]))
            
            # Append results and fitted models to the result list
            final_ensemble.append([estimator, estimator_predictors, 
                                    f_indices, n_predictors, all_sample_score])
            
    
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





