#!/bin/bash
#
# To execute the script in FMI's Voima XC40 HPC environment, type:
# $ sh run_download_and_preprocess_data.sh



# Prepare the Anaconda Python environment 
export PATH="/lustre/tmp/kamarain/miniconda/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/tmp/kamarain/miniconda/lib"

export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080
export ftp_proxy=http://wwwproxy.fmi.fi:8080 


mkdir -p /lustre/tmp/kamarain/netcdf_input/
cp /home/users/kamarain/seasonal_forecasting/download_era20c_from_ecmwf.py   /lustre/tmp/kamarain/netcdf_input/
cd /lustre/tmp/kamarain/netcdf_input/

# Download and uncompress HadISST1
wget https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz && gunzip -f HadISST_sst.nc.gz



# Modify this list to select ERA-20C variables to download
#declare -a vars=('vo850' 'sst' 'pmsl' 'z70' 'snw' 'aice' 'te2m' 'smo' 'tcw' 'z500')
declare -a vars=('snw' 'te2m' 'lsmask' 'prec')

# Download and preprocess ERA-20C
for var in "${vars[@]}"
do
   echo $var
   python download_era20c_from_ecmwf.py $var &
done
wait

echo "Finished run_download_and_preprocess_data.sh!"
