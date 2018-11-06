#!/usr/bin/env python




import sys, imp
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib; matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white")
import cartopy.crs as ccrs


# Directories
basedir = str(sys.argv[4])
in__dir = str(sys.argv[5])
out_dir = str(sys.argv[6])

# Own functions
sys.path.append(basedir)
try:     fcts = imp.reload(fcts)
except:  import functions as fcts



# Read and process all input data, define variables etc.
data = fcts.READ_DEFINE_AND_PROCESS_EVERYTHING(basedir, in__dir)

# Read results from fittings
results = pd.read_pickle(out_dir+data['basename']+'.pkl')
observations = data['y_eur'][data['y_var']].mean(axis=1).values


print('Plotting results for',data['basename'])


# Analyse numbers of predictors in the ensemble members,
# and the occurrence of individual predictors

sns.set(style="whitegrid")
fig1, axes = plt.subplots(2, len(data['seasons']), figsize=(len(data['seasons'])*3,7), sharey='row')
for s,ssn in enumerate(data['seasons']):
    
    ssn_idx = results['Season']==ssn 
    
    to_plot = fcts.count_prdctr_freqs(results[ssn_idx]['Optimal predictors'], 'Predictor'); #print(to_plot)
    to_plot['Coefficient'] = np.nan
    coefs = {}
    for i,pdctr in enumerate(to_plot['Predictor']):
        coefs[pdctr] = [] #np.full(100,np.nan)
        for row in results[ssn_idx].iterrows():
            idx = np.array(row[1]['Optimal predictors']) == pdctr
            coef = row[1]['Regression coefficients'][idx]
            if(len(coef>0)): coefs[pdctr].append(coef[0])
    
    coefs = pd.DataFrame.from_dict(coefs, orient='index').transpose()
    
    g = sns.barplot(x='Predictor',y='Frequency (%)',data=to_plot[:15],ax=axes[0,s],palette='Reds_r'); 
    g.set_xticklabels(g.get_xticklabels(),rotation=90); 
    g.set_title(ssn)
    
    to_plot = fcts.count_prdctr_freqs(results[ssn_idx]['N of optimal predictors'], 'N of predictors').sort_index(); #print(to_plot)
    h = sns.barplot(x='N of predictors', y= 'Frequency (%)', data=to_plot[:15],palette='Reds_r', ax=axes[1,s])
    h.set_xticklabels(h.get_xticklabels(),rotation=90); 
    h.set_title(ssn)
    

plt.tight_layout()
fig1.savefig(out_dir+'fig_prd_counts_mxcps_'+data['basename']+'.png',bbox_inches='tight',dpi=120)
fig1.savefig(out_dir+'fig_prd_counts_mxcps_'+data['basename']+'.pdf',bbox_inches='tight'); #plt.show()#











# Calculate and save skill scores,
# and plot time series graphs 

skill_scores =  pd.DataFrame(data = {   
                    'ACC': [np.nan,  np.nan,  np.nan,  np.nan], 
                    'MSSS_clim': [np.nan,  np.nan,  np.nan,  np.nan],
                    'MSSS_pers': [np.nan,  np.nan,  np.nan,  np.nan],
                }, index = data['seasons']) 



sns.set(style="whitegrid")
fig1, axes1 = plt.subplots(len(data['seasons']),1, figsize=(7,2.2*len(data['seasons'])))


for s,ssn in enumerate(data['seasons']): 
    
    n_models = int(data['n_smpls'])
    
    ssn_idx = results['Season']==ssn 
    all_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['all_yrs']))
    trn_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['trn_yrs']))
    tst_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['tst_yrs']))
    
    forecast = np.full((data['Y'][data['y_var']].shape[0], n_models), np.nan)
    
    # Apply models to all years (including training and test periods)
    # Apply quantile mapping also
    for m,model in enumerate(results[ssn_idx]['Fitted model'][:n_models]):
        try:
            forecast[all_idx,m] = model.predict(data['X'].values[all_idx]) 
            
            if(data['experiment']=='SKIPQMAP'):
                pass
            else:
                forecast[all_idx,m] = fcts.q_mapping(observations[trn_idx], forecast[trn_idx,m], forecast[all_idx,m], 100)
            
        except: pass
    
    persistence = np.full(observations.shape[0], np.nan)
    
    persistence[all_idx] = fcts.q_mapping(observations[trn_idx], observations[trn_idx-1], observations[all_idx-1], 100)
    obs = observations[tst_idx] 
    cli = np.zeros(observations[tst_idx].shape)
    per = persistence[tst_idx]
    fcs = np.nanmean(forecast[tst_idx], axis=1)
    #bst = np.nanmean([cli,per,fcs],axis=0)
    
    if(data['experiment']=='DETREND'):
        # Remove linear trends from the test years
        import scipy.signal as sgn
        obs = sgn.detrend(obs, type='linear')
        per = sgn.detrend(per, type='linear')
        fcs = sgn.detrend(fcs, type='linear')
    else:
        pass
    
    L = 5; B = 10000
    corf = fcts.calc_bootstrap(fcs, obs, cli, fcts.calc_corr, [5,95], L, B)
    corp = fcts.calc_bootstrap(per, obs, cli, fcts.calc_corr, [5,95], L, B)
    cora = fcts.calc_bootstrap(np.nanmean(forecast[all_idx], axis=1), 
                                observations[all_idx], observations[all_idx], 
                                fcts.calc_corr, [5,95], L, B)
    
    msss_clim = fcts.calc_bootstrap(fcs,obs,cli, fcts.calc_msss,[5,95], L, B)
    msss_pers = fcts.calc_bootstrap(fcs,obs,per, fcts.calc_msss,[5,95], L, B)
    
    skill_scores['ACC'][ssn]        = str(round(corf[1],2))+corf[3]
    skill_scores['MSSS_clim'][ssn]  = str(round(msss_clim[1],2))+msss_clim[3]
    skill_scores['MSSS_pers'][ssn]  = str(round(msss_pers[1],2))+msss_pers[3]
    
    print('Bootstrap for '+data['y_var']+' correlation in '+data['y_area']+' '+ssn+' all years:',cora)
    print('Bootstrap for '+data['y_var']+' correlation in '+data['y_area']+' '+ssn+' test years:',corf)
    print('N of models for '+data['y_var']+' in '+data['y_area']+' '+ssn+':',str(forecast.shape[1]))
    
    dtaxis = pd.to_datetime(data['Y']['time'][all_idx].values).values
    p_95 = np.nanpercentile(forecast[all_idx], q=[2.5, 97.5], axis=1)
    
    axes1[s].plot(dtaxis, observations[all_idx],c='k',linewidth=0.8, label='Observations')
    axes1[s].plot(dtaxis, persistence[all_idx],c='k',linestyle='--',linewidth=0.8,label='Persistence, ACC='+str(corp[1])[0:4])
    axes1[s].plot(dtaxis, np.nanmean(forecast[all_idx], axis=1),c='r',label='Ensemble mean')
    axes1[s].fill_between(x=dtaxis, y1=p_95[0], y2=p_95[1], color="r", alpha=0.2, label='Ensemble 95% range')
    
    axes1[s].axvline(x=data['Y']['time'][trn_idx][-1].values, c='k', linewidth=2.0)
    axes1[s].legend(loc='lower right',ncol=2,frameon=True,fontsize=6)
    axes1[s].set_title(ssn+',  ACC: '+str(corf[1])[0:5]+corf[3]+', MSSS (climatology): '+
                        str(msss_clim[1])[0:5]+msss_clim[3]+', MSSS (persistence): '+
                        str(msss_pers[1])[0:5]+msss_pers[3]) #+', a='+str(np.mean(alp))[0:5])
    axes1[s].set_xlim(['1940-10-31 00:00:00', '2012-10-31 00:00:00'])
    
    
    plt.tight_layout(); 
    fig1.savefig(out_dir+'fig_probab_nmodels'+str(n_models)+'_'+data['basename']+'.png',dpi=120)
    fig1.savefig(out_dir+'fig_probab_nmodels'+str(n_models)+'_'+data['basename']+'.pdf'); #plt.show()


skill_scores.to_csv(out_dir+'skill_scores_'+data['basename']+'.csv')









# Apply a moving ACC analysis window for different seasons

sns.set(style="whitegrid")
fig1, axes1 = plt.subplots(1,1, figsize=(7,3))

window=12; ssn_c = ('r', 'b', 'k', 'c')
for s,ssn in enumerate(data['seasons']): 
    
    n_models = int(data['n_smpls'])
    
    ssn_idx = results['Season']==ssn 
    all_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['all_yrs']))
    trn_idx = fcts.bool_index_to_int_index(np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['trn_yrs']))
    
    forecast = np.full((data['Y'][data['y_var']].shape[0], n_models), np.nan)
    
    for m,model in enumerate(results[ssn_idx]['Fitted model'][:n_models]):
        try:
            forecast[all_idx,m] = model.predict(data['X'].values[all_idx]) 
            forecast[all_idx,m] = fcts.q_mapping(observations[trn_idx], forecast[trn_idx,m], forecast[all_idx,m], 100)
        except: pass 
    
    fcs = np.nanmean(forecast[all_idx], axis=1)
    obs = observations[all_idx] 
    
    acc_mov = np.full(observations[all_idx].shape, np.nan); 
    for i,tstep in enumerate(all_idx):
        try: 
            acc_mov[i] = fcts.calc_corr(fcs[i-window:i+window], obs[i-window:i+window])
        except: pass
    
    acc_mov[-window:] = np.nan
    dtaxis = pd.to_datetime(data['Y']['time'][all_idx].values).values
    
    axes1.plot(dtaxis, acc_mov, c=ssn_c[s], linewidth=1.8, label=ssn)
    
    if(ssn=='JJA'): axes1.axvline(x=data['Y']['time'][trn_idx][-1].values, c='k', linewidth=2.0)


axes1.set_ylabel('ACC');plt.tight_layout();plt.legend(loc='best')
fig1.savefig(out_dir+'fig_movingACC_nmodels'+str(n_models)+'_'+data['basename']+'.png',dpi=120);
fig1.savefig(out_dir+'fig_movingACC_nmodels'+str(n_models)+'_'+data['basename']+'.pdf'); #plt.show()



print("Finished plot_results_for_regions.py!")




