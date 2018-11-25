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
from tqdm import tqdm


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



def masking(observation, model):
    masker = (observation.isel(time=0))
    mask = (xr.where(masker >= 0, 1, np.nan))
    start_lat = mask['lat'][0].values
    end_lat = mask['lat'][-1].values
    start_lon = mask['lon'][0].values
    end_lon = mask['lon'][-1].values
    mask = mask.rename({'observation':'pr'})
    temp = model['PRECT'].sel(lat = slice(start_lat,end_lat),lon = slice(start_lon,end_lon))
    masked_model = mask*temp
    return masked_model
                           
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



def summed_extremes(model,ndays):
        summed = model.rolling(time=ndays).sum()
        maximas = summed.groupby('time.year').max('time')
        return maximas


def EVT(maxima_nc, var_name = "pr",conversion = 86400*1000):
    
    lats = maxima_nc.lat.values
    lons = maxima_nc.lon.values

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
        

            maxima = maxima_nc.sel(lat = lat1,lon = lon1)

            var_keys = maxima.data_vars.keys()
            attr_name =  "maxima."+"pr"+".values[0]"

            if abs(eval(attr_name))>= 0:
                lati.append(lat1)
                loni.append(lon1)

                block_maxima = "maxima."+"pr"+".values"
                block_maxima = eval(block_maxima)
                block_maxima = block_maxima*conversion
                block_maxima = block_maxima[block_maxima < 900]
                block_maxima = block_maxima[~np.isnan(block_maxima)]

                fevd_object = extremes.fevd(block_maxima , verbose = False)


                params['location'] = fevd_object.rx2('results')[0][0]
                params['scale'] =  fevd_object.rx2('results')[0][1]
                params['shape'] = fevd_object.rx2('results')[0][2]



              

                y = eva.rgevr(1000, 1, loc = params['location'], scale = params['scale'], shape = params['shape'])

                sig_test = statis.ks_test(block_maxima, y)
                params['p_val'] = sig_test.rx2('p.value')[0]

                parameters.append(params)

                

                return_periods = [5,10,30,50,100]

                for r in return_periods:
                    
                    name =  'years_' + str(r)
                    name2 = name + '_LB'
                    name3 = name + '_UB'


                    try:
                        temp = extremes.ci_fevd(fevd_object, alpha = 0.05, return_period = r, verbose = False)
                        RL = temp[1]      
                        LB = temp[0]
                        UB = temp[2]  
                        if UB>RL and RL >0:
                            return_levels[name] = RL
                            rl_bound[name2] = LB
                            rl_bound[name3] = UB
                        else:
                            return_levels[name] = np.nan
                            rl_bound[name2] = np.nan
                            rl_bound[name3] = np.nan

                    except:
                        return_levels[name] = np.nan
                        rl_bound[name2] = np.nan
                        rl_bound[name3] = np.nan            
                    
                    print((LB, RL, UB))
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


def GEV_combined(model_list,varname = "pr" ,conversion = 86400*1000):

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


            maxima = model_list[0].sel(lat = lat1,lon = lon1)
            var_keys = maxima.data_vars.keys()
            attr_name =  "maxima."+varname+".values[0]"


            if np.isnan(eval(attr_name)):
                continue

            appended_maxima =[]
            lati.append(lat1)
            loni.append(lon1)

            for i, maximum in enumerate(model_list):

                maxima1 = maximum.sel(lat = lat1,lon = lon1)
                maxima_vals = (eval("maxima1."+varname+".values"))
                maxima_vals = maxima_vals*conversion
                maxima_vals = maxima_vals[~np.isnan(maxima_vals)]
                maxima_vals = maxima_vals[maxima_vals<1600]
                maxima_vals = list(maxima_vals)

                appended_maxima.append(maxima_vals)

            flat_max =  flatten(appended_maxima)
            
            block_maxima = np.asarray(flat_max)
            print(type(block_maxima))
            fevd_object = extremes.fevd(block_maxima , verbose = False)


            params['location'] = fevd_object.rx2('results')[0][0]
            params['scale'] =  fevd_object.rx2('results')[0][1]
            params['shape'] = fevd_object.rx2('results')[0][2]



              

            y = eva.rgevr(1000, 1, loc = params['location'], scale = params['scale'], shape = params['shape'])

            sig_test = statis.ks_test(block_maxima, y)
            params['p_val'] = sig_test.rx2('p.value')[0]

            parameters.append(params)

                

            return_periods = [5,10,30,50,100]

            for r in return_periods:
                
                name =  'years_' + str(r)
                name2 = name + '_LB'
                name3 = name + '_UB'


                try:
                    temp = extremes.ci_fevd(fevd_object, alpha = 0.05, return_period = r, verbose = False)
                    RL = temp[1]      
                    LB = temp[0]
                    UB = temp[2]  
                    if UB>RL and RL >0:
                        return_levels[name] = RL
                        rl_bound[name2] = LB
                        rl_bound[name3] = UB
                    else:
                        return_levels[name] = np.nan
                        rl_bound[name2] = np.nan
                        rl_bound[name3] = np.nan

                except:
                    return_levels[name] = np.nan
                    rl_bound[name2] = np.nan
                    rl_bound[name3] = np.nan            
                
                print((LB, RL, UB))
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


flatten = lambda l: [item for sublist in l for item in sublist]

