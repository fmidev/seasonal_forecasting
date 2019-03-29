# Statistical seasonal forecasts
A Python code for producing statistical seasonal forecast experiments in Europe.

## Dependencies
The code was developed using Python 3.6 and several external libraries,
which were installed with the Miniconda installer, available here:
https://conda.io/miniconda.html

The conda-forge repository was used to install the libraries:
`conda install -c conda-forge numpy scipy matplotlib cartopy xarray seaborn pandas scikit-learn`

## Downloading the input data  
For downloading the data, one Unix shell file, and one Python script are used.
The shell file, run_download_and_preprocess_data.sh, downloads directly some input data 
needed for modeling, and uses the download_era20c_from_ecmwf.py script to download the rest.

Prior to running the file, make sure that the folder structures are correctly defined inside 
the files, and all Python dependencies are installed. One additional library, ecmwfapi, 
is also needed:
https://confluence.ecmwf.int/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch

Running the file as a regular Unix shell file:
`bash run_download_and_preprocess_data.sh`

## Running the forecasting experiments
For the actual code, one Unix shell file and three Python scritps are used.
The shell file, run_fit_models_and_plot_results.sh, defines command line arguments 
for Python scripts. An alternative run file, run_fit_models_and_plot_results_VOIMA.sh, 
launches the scripts to computing nodes of the VOIMA PBS system of FMI. 

Prior to running the scripts, make sure that the folder structures are correctly defined, and all
Python dependencies are installed (and visible to computing nodes if running in VOIMA). 
In VOIMA the computation takes typically 2-3 hours, so be sure to request enough wall time.

Running the experiments in a regular Unix environment:
`bash run_fit_models_and_plot_results.sh`

Running the experiments in VOIMA PBS:
`qsub run_fit_models_and_plot_results_VOIMA.sh`
