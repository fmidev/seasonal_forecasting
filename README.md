# Seasonal forecasting
A Python code for producing statistical seasonal forecasts in Europe.

## Dependencies
The code was developed using Python 3.6 and several external libraries,
installed with the Miniconda installer, available here:
https://conda.io/miniconda.html.

The conda-forge repository was used to install the libraries:
conda install -c conda-forge numpy scipy matplotlib cartopy xarray seaborn pandas scikit-learn

At the moment, CDO is needed to process the grib files. The code could be modified such that
using CDO is avoided, but it has not been implemented yet.

CDO is available here:
https://code.mpimet.mpg.de/projects/cdo

## Downloading the input data  
For downloading the data, one Unix shell file, and one Python script are used.
The shell file, run_download_and_preprocess_data.sh, downloads directly some input data 
needed for modeling, and uses the download_era20c_from_ecmwf.py script to download the rest.

Prior to running the file, make sure that the folder structures are correctly defined, and all
Python dependencies are installed. One additional library, ecmwfapi, is also needed:
https://confluence.ecmwf.int/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch

Running the file as a regular Unix shell file:
sh run_download_and_preprocess_data.sh 

## Running the forecasting experiments
For the actual code, one Unix shell file and three Python scritps are used.
The shell file, run_fit_models_and_plot_results.sh, defines command line arguments 
for Python scripts, and launches the scripts to computing nodes of the Voima PBS
system of FMI. The shell file can be simplified such that it can be run outside the PBS
system: for that, use the run_download_and_preprocess_data.sh as a template.

Prior to running the file, make sure that the folder structures are correctly defined, and all
Python dependencies are installed.

Running the file in Voima PBS:
qsub run_fit_models_and_plot_results.sh 

Running the file as a regular Unix shell file:
sh run_fit_models_and_plot_results.sh 

