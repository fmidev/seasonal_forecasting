#!/usr/bin/env python



import sys, imp, glob
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib; matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white")


from sklearn.preprocessing import StandardScaler

# Directories
basedir = str(sys.argv[5])
in__dir = str(sys.argv[6])
out_dir = str(sys.argv[7])

# Own functions
sys.path.append(basedir)
try:     fcts = imp.reload(fcts)
except:  import functions as fcts




# Read and process all input data, define variables etc.
data = fcts.READ_DEFINE_AND_PROCESS_EVERYTHING(basedir, in__dir)



print('Plotting results for',data['basename'])









def str_n_float(data_frame):
    out_str = data_frame.copy(deep=True)
    out_flt = data_frame.copy(deep=True)
    try:
        out_flt = out_flt.replace({'\*':''}, regex=True).astype(float)
    except:
        out_flt = out_flt.astype(float)
    
    return out_str, out_flt




# Collect validation metrics for all domains and seasons,
# and draw them as a matrix

cols = {'scandi':' SC', 'westeu':' WE', 
        #'easeur':' EA', 'meditr':' MD', 
        'europe':' EU'}


correlations =  pd.DataFrame(data = {   
                    'DJF': np.full(len(cols), np.nan), 
                    'MAM': np.full(len(cols), np.nan),
                    'JJA': np.full(len(cols), np.nan),
                    'SON': np.full(len(cols), np.nan),
                }, index = list(cols.values())) 

msss_clim =  correlations.copy(deep=True) 
msss_pers =  correlations.copy(deep=True)




skill_scr_files = glob.glob(out_dir+'skill_scores_'+data['basename'][:-6]+'*csv')

for i,fle in enumerate(skill_scr_files):
    dta = pd.read_csv(fle, index_col=0)
    
    import re
    p = list(cols.keys())
    for ptn in p:
        result = re.search(ptn,fle)
        if result: 
            y_area = ptn
    
    correlations.loc[cols[y_area]] = dta['ACC'].values
    msss_clim.loc[cols[y_area]] = dta['MSSS_clim'].values
    msss_pers.loc[cols[y_area]] = dta['MSSS_pers'].values
    

corr_str, corr_flt = str_n_float(correlations)
mscl_str, mscl_flt = str_n_float(msss_clim)
mspr_str, mspr_flt = str_n_float(msss_pers)



print('Mean ACC: ',np.nanmean(correlations.replace({'\*':''}, regex=True).astype(float)))

mean_corr = round(np.mean(corr_flt.values), 2)
mean_mscl = round(np.mean(mscl_flt.values), 2)
mean_mspr = round(np.mean(mspr_flt.values), 2)


sns.set(style="white", font_scale=1.6) 
fig, axes = plt.subplots(1,3,figsize=(14,4),sharey=True)
cmap = plt.cm.RdBu_r #plt.cm.hot_r 


spl = sns.heatmap(corr_flt,  
                cmap=cmap, vmax=0.7, vmin=-0.7, ax=axes[0],
                annot=corr_str, fmt='', annot_kws={'fontsize':18}, cbar=False,
            ); spl.set_title('ACC, mean: '+str(mean_corr)[0:5])

spl.set_yticklabels(spl.get_yticklabels(), rotation = 0)

spl = sns.heatmap(mscl_flt,  
                cmap=cmap, vmax=0.7, vmin=-0.7, ax=axes[1],
                annot=mscl_str, fmt='', annot_kws={'fontsize':18}, cbar=False,
            ); spl.set_title('MSSS$_{clim}$, mean: '+str(mean_mscl))

spl = sns.heatmap(mspr_flt,
                cmap=cmap, vmax=0.7, vmin=-0.7, ax=axes[2],
                annot=mspr_str, fmt='', annot_kws={'fontsize':18}, cbar=False,
            ); spl.set_title('MSSS$_{pers}$, mean: '+str(mean_mspr))

plt.tight_layout()
fig.savefig(out_dir+'fig_acc_msss_'+data['y_var']+'_'+data['basename']+'.png',dpi=100)
fig.savefig(out_dir+'fig_acc_msss_'+data['y_var']+'_'+data['basename']+'.pdf'); #plt.show()








# Stop execution here if not CONTROL
if(data['experiment'] == 'CONTROL'):
    pass
else:
    sys.exit()




# Read results from all areas
files = glob.glob(out_dir+data['basename'][:-6]+'*pkl')
for fle in files:
    dta = pd.read_pickle(fle)
    try:
        results = results.append(dta)
    except:
        results = dta


    
counts = fcts.count_prdctr_freqs(results['Optimal predictors'], 'Predictor')

print('Frequency of predictors in all domains and seasons:')
print(counts)





# Plot PCs and EOFs for each gridded input parameter separately
for i,var in enumerate(data['X_vars']):
        
    nme = list(data['X_var_definitions'].keys())[i]
    ncomps = data['X_var_definitions'][nme][1]
    
    cps_full, pca_full, svl = fcts.apply_PCA(var.values, ncomps)
    cps_full[0] = cps_full[1]
    cps_full = StandardScaler().fit_transform(cps_full)
    
    patterns = StandardScaler().fit_transform(pca_full.components_.T).T
    
    print('Tot evr:', np.sum(pca_full.explained_variance_ratio_*100))
    pattern_ds = xr.full_like(var[0:ncomps],   np.nan).rename({'time': 'Comp'})
    compnnt_ds = xr.full_like(var[:,0:ncomps], np.nan).rename({'gridcell': 'Comp'})
    pattern_ds['Comp'] = np.arange(1,ncomps+1)
    compnnt_ds['Comp'] = np.arange(1,ncomps+1)
    
    sns.set(style="white", font_scale=1.0) 
    fig, axes = plt.subplots(ncomps, 2, figsize=(9,ncomps*2), gridspec_kw={'width_ratios':[2,1]})
    
    levels = [-2, -1, -0.5, 0, 0.5, 1, 2]   
    
    for j,cp in enumerate(np.arange(ncomps)):
        
        pattern = patterns[j] 
        compnnt = cps_full[:,j] 
        evr = pca_full.explained_variance_ratio_[j]*100
        
        pattern_ds[j]   = np.squeeze(pattern)
        compnnt_ds[:,j] = np.squeeze(compnnt)
        #
        
        fgrd = pattern_ds[j].unstack('gridcell').plot.contourf(ax=axes[j,1], 
                levels=levels,center=0, add_colorbar=True, cbar_kwargs={'label': ''})
        
        lsmask = fcts.read_and_select(in__dir+'lsmask_era20c_monthly_1900-2010.nc', 'lsm', -99)
        if(nme=='SST'):
            lsmask['lsm'] = lsmask['lsm'].where(lsmask['lsm']>0.5, other=np.nan)
            lsmask['lsm'].plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        if(nme=='GPT'):
            lsmask['lsm'] = lsmask['lsm'].where(lsmask['lsm']>0.5, other=np.nan)
            lsmask['lsm'].plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1], alpha=0.3)
        
        if(nme=='SNC'):
            lsmask['lsm'] = lsmask['lsm'].where(lsmask['lsm']<0.5, other=np.nan)
            lsmask['lsm'].sel(lat=slice(-10,90)).plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        if(nme=='SIC'):
            lsmask['lsm'] = lsmask['lsm'].where(lsmask['lsm']>0.5, other=np.nan)
            lsmask['lsm'].sel(lat=slice(0,90)).plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        
        tser = compnnt_ds[:,j].to_dataframe().drop(columns=['season','Comp'])
        tser.plot(ax=axes[j,0], color='darkred', legend=False)
        #
        
        axes[j,0].set_title(nme+' PC'+str(j+1)+', expl. variance: '+str(evr)[0:4]+'%'); 
        axes[j,1].set_title(''); axes[j,1].set_xlabel(''); axes[j,1].set_ylabel(''); 
        axes[j,0].set_xlabel(''); axes[j,0].set_ylabel(''); axes[j,0].set_ylim([-3.1, 3.1])
        #axes[j,0].axvline(x=tser.iloc[tser.index=='1986-01-31'].index.values[0], c='k', linewidth=2.0)
        
    plt.tight_layout(); #plt.show()
    plt.savefig(out_dir+'fig_pca_tsers_patterns_'+nme+'_'+data['basename']+'.png',dpi=120)
    plt.savefig(out_dir+'fig_pca_tsers_patterns_'+nme+'_'+data['basename']+'.pdf'); #plt.show()





print("Finished plot_collected_results.py!")



    








