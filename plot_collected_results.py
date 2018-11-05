#!/usr/bin/env python



# Data directories
basedir = '/lustre/tmp/kamarain/seasonal_prediction/'
in__dir = '/lustre/tmp/kamarain/netcdf_input/'
out_dir = basedir+'results/'


import sys, imp, glob
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib; matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white")
import cartopy.crs as ccrs
#import cartopy.feature as cfeature

from sklearn.preprocessing import StandardScaler

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

cols = {'scandi':' FS', 'wsteur':' WE', 
        'easeur':' EA', 'meditr':' MD', 
        'europe':' EU'}


correlations =  pd.DataFrame(data = {   
                    'DJF': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan], 
                    'MAM': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'JJA': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'SON': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                }, index = list(cols.values())) 

msss_clim =  pd.DataFrame(data = {   
                    'DJF': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan], 
                    'MAM': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'JJA': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'SON': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                }, index = list(cols.values())) 

msss_pers =  pd.DataFrame(data = {   
                    'DJF': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan], 
                    'MAM': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'JJA': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                    'SON': [np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
                }, index = list(cols.values())) 




skill_scr_files = glob.glob(out_dir+'skill_scores*'+data['experiment']+'*'+data['y_var']+'*csv')

for i,fle in enumerate(skill_scr_files):
    dta = pd.read_csv(fle, index_col=0)
    
    import re
    p = ('scandi', 'wsteur', 'easeur', 'meditr', 'europe')
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
files = glob.glob(out_dir+'*'+data['experiment']+'*'+data['y_var']+'*pkl')
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
    
    cps_full, pca_full = fcts.apply_PCA(var.values, ncomps)
    cps_full[0] = cps_full[1]
    cps_full = StandardScaler().fit_transform(cps_full)
    
    patterns = StandardScaler().fit_transform(pca_full.components_.T).T
    
    print('Tot evr:', np.sum(pca_full.explained_variance_ratio_*100))
    pattern_ds = xr.full_like(var[0:ncomps],   np.nan).rename({'time': 'Comp'})
    compnnt_ds = xr.full_like(var[:,0:ncomps], np.nan).rename({'gridcell': 'Comp'})
    pattern_ds['Comp'] = np.arange(1,ncomps+1)
    compnnt_ds['Comp'] = np.arange(1,ncomps+1)
    
    sns.set(style="white", font_scale=1.0) 
    fig, axes = plt.subplots(6, 2, figsize=(10,12), gridspec_kw={'width_ratios':[2,1]})
    
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
        
        lsmask = fcts.read_and_select(in__dir+'lsmask_era20c_monthly_1900-2010.nc', 'LSM', (-99, -99), -99)
        if(nme=='SST'):
            lsmask['LSM'] = lsmask['LSM'].where(lsmask['LSM']>0.5, other=np.nan)
            lsmask['LSM'].plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        if(nme=='SNC'):
            lsmask['LSM'] = lsmask['LSM'].where(lsmask['LSM']<0.5, other=np.nan)
            lsmask['LSM'].sel(lat=slice(0,90)).plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        if(nme=='SIC'):
            lsmask['LSM'] = lsmask['LSM'].where(lsmask['LSM']>0.5, other=np.nan)
            lsmask['LSM'].sel(lat=slice(0,90)).plot.contourf(ax=axes[j,1], add_colorbar=False, colors='k', levels=[-0.1,1.1])
        
        
        tser = compnnt_ds[:,j].to_dataframe().drop(columns=['season','Comp']).plot(ax=axes[j,0], legend=False)
        #
        
        axes[j,0].set_title(nme+' PC'+str(j+1)+', expl. variance: '+str(evr)[0:4]+'%'); 
        axes[j,1].set_title(''); axes[j,1].set_xlabel(''); axes[j,1].set_ylabel(''); 
        axes[j,0].set_xlabel(''); axes[j,0].set_ylabel(''); axes[j,0].set_ylim([-3.1, 3.1])
    
    plt.tight_layout(); #plt.show()
    plt.savefig(out_dir+'fig_pca_tsers_patterns_'+nme+'_'+data['basename']+'.png',dpi=120)
    plt.savefig(out_dir+'fig_pca_tsers_patterns_'+nme+'_'+data['basename']+'.pdf'); #plt.show()







# Plot domain boundaries on a map
fig, ax = plt.subplots(1,1,figsize=(6,6), 
    subplot_kw={'projection':fcts.LowerThresholdLConf(central_longitude=15, central_latitude=30)})

ax.set_extent([-12, 41, 28.5, 73.8], ccrs.PlateCarree())

# Boxes
fcts.plot_rectangle(ax, ccrs.PlateCarree(), 4,34,55,71, 'r', 3.5,1)   # Scandi
fcts.plot_rectangle(ax, ccrs.PlateCarree(), -10.5,17,42,59, 'k', 3.5,1)  # West
fcts.plot_rectangle(ax, ccrs.PlateCarree(), 17.7,43,38,56.5, 'c', 3.5,1)  # East
fcts.plot_rectangle(ax, ccrs.PlateCarree(), 0,25,30,45, 'Orange', 3.5,1) # Meditr
fcts.plot_rectangle(ax, ccrs.PlateCarree(), -12,40,33,73, 'Purple', 3.5,1) # Europe

# Texts
fcts.plot_text(ax,ccrs.PlateCarree(),[12,],[69,],['SC',],'r',25,0,False)
fcts.plot_text(ax,ccrs.PlateCarree(),[-5,],[55.5,],['WE',],'k',25,0,False)
fcts.plot_text(ax,ccrs.PlateCarree(),[22,],[51.5,],['EA',],'c',25,0,False)
fcts.plot_text(ax,ccrs.PlateCarree(),[5,],[35,],['MD',],'Orange',25,0,False)
fcts.plot_text(ax,ccrs.PlateCarree(),[-4,],[70.5,],['EU',],'Purple',25,0,False)
#ax.gridlines(linewidth=0.5)

#ax.#add_feature(cfeature.LAND, zorder=0)
ax.coastlines(); #add_feature(cfeature.COASTLINE, zorder=0)

plt.tight_layout(); 
fig.savefig(out_dir+'fig_map_areas.png',dpi=100)
fig.savefig(out_dir+'fig_map_areas.pdf'); #plt.show()



print("Finished plot_collected_results.py!")



    








