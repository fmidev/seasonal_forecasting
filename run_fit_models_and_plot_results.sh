#!/bin/bash
# To execute the script in UNIX environment, type:
# $ sh run_fit_models_and_plot_results.sh
#



# Prepare the Anaconda Python environment 
export PATH="/lustre/tmp/kamarain/miniconda/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/tmp/kamarain/miniconda/lib"


# Target variable
y_var='T2M'

# Target domains
#declare -a areas=('europe' 'scandi' 'easeur' 'westeu' 'meditr')
declare -a areas=('europe' 'scandi' 'westeu')

# Experiment 
# Possible values: 'CONTROL', 'INCLPERSIS', 'SKIPQMAP', 'DETREND', 'NO_LAGS', 'CUTFIRSTYRS', 'SINGLE', 'WEIGHTING', 'FOLLAND'
exp='CONTROL'

# Source for predictor data
# Possible values: 'ERA-20C', '20CRv2c'
src='ERA-20C'

# Directories
basedir='/home/users/kamarain/seasonal_forecasting/'
in__dir='/lustre/tmp/kamarain/netcdf_input/'
out_dir='/lustre/tmp/kamarain/seasonal_prediction/results/'


# Fit models for each area one by one
for area in "${areas[@]}"
do
   echo $area
   #python fit_models.py $y_var $area $exp $src $basedir $in__dir $out_dir 
done


# Plot results for each area one by one
for area in "${areas[@]}"
do
   echo $area
   #python plot_results_for_regions.py $y_var $area $exp $src $basedir $in__dir $out_dir 
done


# Plot collected results 
area='europe'
python plot_collected_results.py $y_var $area $exp $src $basedir $in__dir $out_dir 

echo "Finished run_fit_models_and_plot_results.sh!"
