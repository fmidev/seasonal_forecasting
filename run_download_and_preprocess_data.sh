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

# Download data
#wget https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz && gunzip -f HadISST_sst.nc.gz
#wget ftp://ftp.cdc.noaa.gov/Datasets/gistemp/combined/250km/air.2x2.250.mon.anom.comb.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/gpcc/full_v7/precip.mon.total.1x1.v7.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/sst.mon.anom.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/gaussian/monolevel/air.2m.mon.mean.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/gaussian/monolevel/prate.mon.mean.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/gaussian/monolevel/snowc.mon.mean.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/gaussian/monolevel/icec.mon.mean.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/pressure/hgt.mon.mean.nc


# Modify this list to select ERA-20C variables to download
#declare -a vars=('vo850' 'sst' 'pmsl' 'z70' 'snw' 'aice' 'te2m' 'smo' 'tcw' 'z500')
#declare -a vars=('snw' 'te2m' 'lsmask' 'prec')
declare -a vars=('z15' 'sst')

# Download and preprocess ERA-20C
for var in "${vars[@]}"
do
   echo $var
   python download_era20c_from_ecmwf.py $var &
done
wait

echo "Finished run_download_and_preprocess_data.sh!"
