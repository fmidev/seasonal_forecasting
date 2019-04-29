#!/bin/bash
#PBS -N fit_n_plot
#PBS -j oe
#PBS -k oe
#PBS -l walltime=1:30:00
#PBS -l nodes=3:ppn=28
#
# To execute the script in FMI's Voima XC40 HPC environment, type:
# $ qsub run_fit_models_and_plot_results.sh
#
# Prior to running, make sure that you 
#   - reserved enough nodes: one per target domain
#   - reserved enough computing time: 0.5-1 hours in Voima


# Prepare the Anaconda Python environment 
export PATH="/lustre/tmp/kamarain/miniconda/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/tmp/kamarain/miniconda/lib"

export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080
export ftp_proxy=http://wwwproxy.fmi.fi:8080 

# Copy scripts to Lustre and cd there
cp /home/users/kamarain/seasonal_forecasting/fit_models.py                  /lustre/tmp/kamarain/seasonal_prediction/ 
cp /home/users/kamarain/seasonal_forecasting/fit_retrospective_models.py    /lustre/tmp/kamarain/seasonal_prediction/ 
cp /home/users/kamarain/seasonal_forecasting/plot_results_for_regions.py    /lustre/tmp/kamarain/seasonal_prediction/ 
cp /home/users/kamarain/seasonal_forecasting/plot_results_for_ERA-Int.py    /lustre/tmp/kamarain/seasonal_prediction/ 
cp /home/users/kamarain/seasonal_forecasting/plot_collected_results.py      /lustre/tmp/kamarain/seasonal_prediction/ 
cp /home/users/kamarain/seasonal_forecasting/functions.py                   /lustre/tmp/kamarain/seasonal_prediction/ 

cd /lustre/tmp/kamarain/seasonal_prediction/



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
basedir='/lustre/tmp/kamarain/seasonal_prediction/'
in__dir='/lustre/tmp/kamarain/netcdf_input/'
out_dir='/lustre/tmp/kamarain/seasonal_prediction/results/'


# Fit models using different nodes
for area in "${areas[@]}"
do
   echo $area
   aprun -n1 -N1 -d28 python fit_models.py $y_var $area $exp $src $basedir $in__dir $out_dir &
done
wait

for area in "${areas[@]}"
do
   echo $area
   #aprun -n1 -N1 -d28 python fit_retrospective_models.py $y_var $area $exp $src $basedir $in__dir $out_dir &
done
wait


# Plot results using different nodes
for area in "${areas[@]}"
do
   echo $area
   aprun -n1 -N1 -d28 python plot_results_for_regions.py $y_var $area $exp $src $basedir $in__dir $out_dir &
done
wait

# Apply models to ERA-Interim 
for area in "${areas[@]}"
do
   echo $area
   #aprun -n1 -N1 -d28 python plot_results_for_ERA-Int.py $y_var $area $exp $src $basedir $in__dir $out_dir &
done
wait

# Plot collected results using one node
area='europe'
aprun -n1 -N1 -d28 python plot_collected_results.py $y_var $area $exp $src $basedir $in__dir $out_dir &
wait

echo "Finished run_fit_models_and_plot_results.sh!"
