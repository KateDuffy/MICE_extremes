
import os
import rpy2.robjects as ro
from glob import glob
import xarray as xr 
import platform
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pickle as pkl
import sys
from netCDF4 import Dataset
import netCDF4
import pandas as pd
import multiprocessing
from sklearn.externals.joblib import Parallel,delayed


num_cores = multiprocessing.cpu_count()
import seaborn as sns
homedir = os.environ['HOME']
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
R = ro.r
import warnings
warnings.filterwarnings("ignore")
ismev = importr('ismev')
gof = importr('gnFit')
base = importr('base')
statis = importr('stats')
eva = importr('eva')
extremes = importr('extRemes')


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):

    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

def generate_maximas(nc_file):
    maximas = nc_file.groupby('time.year').max('time')
    return maximas


def EVT_moving(maxima_nc, conversion = 86400, attr = 'pr'):

    lats = maxima_nc.lat.values
    lons = maxima_nc.lon.values
    lat_mesh,lon_mesh = np.meshgrid(lats,lons) 
    start = (maxima_nc.year.values[0]) 
    end = (maxima_nc.year.values[-1]) 
    leap = 30
    D3 = end-start-leap+1
    year = list(range(start,end-leap+1))

    RL_100 = np.zeros((len(lats),len(lons),D3))
    RL_100[:] = np.nan

    RL_30 = np.zeros((len(lats),len(lons),D3))
    RL_30[:] = np.nan


    for i,lati in enumerate(lats):
        for j,loni in enumerate(lons):
            

            results = {}

            maxima = maxima_nc.sel(lat = lati,lon = loni)
            attr_name =  "maxima."+"pr"+".values[0]"

            start = 0
            end = start+leap
            val = []

            k = 0

            if (eval(attr_name)>0):
                start = 0
                end = start+30
                val = []

                while end<len(maxima_nc.year.values):
                             
                    window = (maxima.isel(year = range(start,end)))
                    block_maxima = "window."+attr+".values"
                    block_maxima = eval(block_maxima)
                    block_maxima = block_maxima*conversion
                    block_maxima = block_maxima[block_maxima < 900]
                    block_maxima = block_maxima[~np.isnan(block_maxima)]
                    fevd_object =  extremes.fevd(block_maxima , verbose = False)

                    results['location'] = fevd_object.rx2('results')[0][0]
                    results['scale'] = fevd_object.rx2('results')[0][1]
                    results['shape'] = fevd_object.rx2('results')[0][2]

                    y = eva.rgevr(1000, 1, loc = results['location'], 
                            scale = results['scale'], shape = results['shape'])

                    sig_test = statis.ks_test(block_maxima, y)
                    results['p_val'] = sig_test.rx2('p.value')[0]


                    return_periods = [30,100]

                    for r in return_periods:

                        try:
                            temp = extremes.ci_fevd(fevd_object, alpha = 0.05, 
                            return_period = r, verbose = False)

                            RL = temp[1]

                        except:
                            RL = np.nan

                        if r == 30:
                            RL_30[i,j,k] = RL
                        else:
                            RL_100[i,j,k] = RL

                        print(RL)
                    k = k+1
                    start = start +1
                    end = end +1

    ds = xr.Dataset({'RL_100': (['lat', 'lon', 'year'], RL_100),
                'RL_30': (['lat', 'lon', 'year'], RL_30)},
                    coords={'lon': lons,
                            'lat':  lats,
                            'year': year,})
    return ds

flatten = lambda l: [item for sublist in l for item in sublist]


def GEV_combined(model_list):

    lats = model_list[0].lat.values
    lons = model_list[0].lon.values


    parameters = []
    return_value = []
    bounds = []
    lati = []
    loni = []

    for i,lat1 in enumerate(lats):
        for j,lon1 in enumerate(lons):
            params = {}
            rl_bound = {}
            return_levels ={}
            params = {}


            maxima = model_list[0].sel(lat = lat1,lon = lon1)
            var_keys = maxima.data_vars.keys()
            attr_name =  "maxima."+"pr"+".values[0]"


            if np.isnan(eval(attr_name)):
                continue
            appended_maxima =[]
            lati.append(lat1)
            loni.append(lon1)

            for i, maximum in enumerate(model_list):

                maxima1 = maximum.sel(lat = lat1,lon = lon1)
                maxima_vals = (eval("maxima1."+"pr"+".values"))
                maxima_vals = maxima_vals[~np.isnan(maxima_vals)]
                maxima_vals = maxima_vals[maxima_vals<700]
                maxima_vals = list(maxima_vals)



                appended_maxima.append(maxima_vals)

            flat_max =  flatten(appended_maxima)
            
            block_maxima = np.asarray(flat_max)
            

            gev_fit = eva.gevrFit(block_maxima)
            MLE_est = list(gev_fit.rx2('par.ests'))

            params['location'] = MLE_est[0]

            if MLE_est[1]<0:
                MLE_est[1] = 0.01


            params['scale'] =  MLE_est[1]

            params['shape'] = MLE_est[2]

            y = eva.rgevr(10000, 1, loc = MLE_est[0], scale = MLE_est[1], shape = MLE_est[2])

            sig_test = statis.ks_test(block_maxima, y)

            params['p_val'] = sig_test.rx2('p.value')[0]

            parameters.append(params)



            return_periods = [5,10,30,50,100,200,500]

            for r in return_periods:
                temp = eva.gevrRl(gev_fit, r, method = "delta")
                RL = temp.rx2('Estimate')[0]
                if RL <0:
                    RL = np.nan
                name =  'years_' + str(r)
                return_levels[name] = RL

                LB = temp.rx2('CI')[0]
                UB = temp.rx2('CI')[1]

                name2 = name + '_LB'
                name3 = name + '_UB'

                rl_bound[name2] = LB
                rl_bound[name3] = UB


            return_value.append(return_levels)
            bounds.append(rl_bound)

    return_value =  pd.DataFrame(return_value).set_index([lati,loni])
    return_value = return_value.to_xarray()
    return_value = return_value.rename({'level_0': 'lat', 'level_1': 'lon'}, inplace=True)

    parameters   =  pd.DataFrame(parameters).set_index([lati,loni])
    parameters   = parameters.to_xarray()
    parameters = parameters.rename({'level_0': 'lat', 'level_1': 'lon'}, inplace=True)

    rl_bound     =  pd.DataFrame(bounds).set_index([lati,loni])
    rl_bound     = rl_bound.to_xarray()
    rl_bound = rl_bound.rename({'level_0': 'lat', 'level_1': 'lon'}, inplace=True)


    return {'return values': return_value, 'parameters' : parameters, 'rl_bound' : rl_bound}






if __name__=="__main__":
    pass








