#!/usr/bin/env python
#
# To run this script, you need an API key
# available from https://api.ecmwf.int/v1/key/
#
# Make sure that you have CDO installed prior to running the script


import numpy as np
import os, sys, itertools
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

# Read the command line argument
varname = str(sys.argv[1]) 

# Create a datestring for monthly retrieval
years = np.arange(1900,2011).astype(int)
months = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')
dates = ''
for year, mon in itertools.product(years, months): 
    dates = dates+'/'+str(year)+mon+'01'

dates = dates[1:]



name2code = {
             'pmsl':    ["151.128",                 "an",   "sfc",  "0",    dates,          "00/03/06/09/12/15/18/21",  "0"],
             'psfc':    ["134.128",                 "an",   "sfc",  "0",    dates,          "00/03/06/09/12/15/18/21",  "0"],
             'z500':    ["129.128",                 "an",   "pl",   "500",  dates,          "00/03/06/09/12/15/18/21",  "0"],
             'z70':     ["129.128",                 "an",   "pl",   "70",   dates,          "00/03/06/09/12/15/18/21",  "0"],
             'vo850':   ["138.128",                 "an",   "pl",   "850",  dates,          "00/03/06/09/12/15/18/21",  "0"],
             'vo500':   ["138.128",                 "an",   "pl",   "500",  dates,          "00/03/06/09/12/15/18/21",  "0"],
             'tcw':     ["136.128",                 "an",   "sfc",  "0",    dates,          "00/03/06/09/12/15/18/21",  "0"],
             'te2m':    ["167.128",                 "an",   "sfc",  "0",    dates,          "00/06/12/18",              "0"],
             'sst':     ["34.128",                  "an",   "sfc",  "0",    dates,          "00/06/12/18",              "0"],
             'snw':     ["141.128",                 "an",   "sfc",  "0",    dates,          "00/06/12/18",              "0"],
             'aice':    ["31.128",                  "an",   "sfc",  "0",    dates,          "00/06/12/18",              "0"],
             'smo':     ["39.128/40.128/41.128",    "an",   "sfc",  "0",    dates,          "00/06/12/18",              "0"],
             'prec':    ["31.228",                  "fc",   "sfc",  "0",    dates,          "00",                       "0"],
             'lsmask':  ["172.128",                 "an",   "sfc",  "0",    "1901-01-01",   "00",                       "0"]
            }




basename = "%s_era20c_monthly_1900-2010" % (varname)
ncfile   = "%s.nc" % (basename)

if os.path.exists(ncfile):
    pass

else:
    opts = {
            "stream"    : "moda", 
            "dataset"   : "era20c",
            "grid"      : "1.25/1.25",
            "param"     : name2code[varname][0],
            "type"      : name2code[varname][1],
            "levtype"   : name2code[varname][2],
            "levelist"  : name2code[varname][3],
            "date"      : name2code[varname][4], 
            "time"      : name2code[varname][5],
            "format"    : "netcdf",
            "target"    : ncfile
           }
    server.retrieve(opts)



print("Finished download_era20c_from_ecmwf.py!")


