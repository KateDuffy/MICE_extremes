import os
import xarray as xr

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import pickle
import sys


def max_mice(filenum,mask):

    start_lat = mask['lat'][0].values
    end_lat = mask['lat'][-1].values
    start_lon = mask['lon'][0].values
    end_lon = mask['lon'][-1].values
    file_name = "*"+filenum+".cam"+"*" + ".nc"
    nc_lists = glob.glob(file_name)
    IC = []
    for file in nc_lists:
        temp = (xr.open_dataset(file))
        temp = temp["PRECT"].sel(lat=slice(start_lat, end_lat),
                                 lon=slice(start_lon, end_lon))
        temp = temp*mask
        temp = temp.groupby('time.year').max('time')
        IC.append(temp)
    merged= xr.merge(IC)
# print("here")
    return(merged)

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



path = '/Volumes/GCM/MICE_data/MICE'
os.chdir(path)


   
    #path = "/Users/udit/Documents/MICE_paper/MICE_Data"


obs = r"Users/uditbhatia/Documents/dev_codes/"
obs_file = r"interpolated_prcp.nc"
path = os.path.join(obs,obs_file)

xr.open_dataset(path)





# masker = (ref_obs.isel(time=0))
# mask = (xr.where(masker >= 0, 1, np.nan))

# mask = mask.rename({'observation': 'pr'})

# file_nums = range(1, 31)
# file_nums = [str('%03d' % i) for i in file_nums]

# pickle_path = "Users/uditbhatia/Documents/dev_codes/"
# pickle_file = "MICE_max_85.pkl"
# pickle_object =  pickle_path+pickle_file

# MICE_extremes = []

# if os.path.exists(pickle_object)==True:

#     for i in file_nums:
        
#         try:
#             temp = (max_mice)(i, mask=mask)
#             MICE_extremes.append(temp)
#             print(i)
#         except:
#             print("cannot open %s" %i)


#     save_as_pickled_object(MICE_extremes, pickle_object)







# for num in file_nums[0:1]:
#     print(num)
#     file_name = "*"+num+".cam"+"*" + ".nc"
#     nc_lists = glob.glob(file_name)
#     IC = []
#     for file in nc_lists:
#         print(file)
#         temp = (xr.open_dataset(file))
#         print(temp)
#         temp = temp["PRECT"].sel(lat=slice(start_lat, end_lat),
#                                  lon=slice(start_lon, end_lon))
#         temp = temp*mask
#         temp = temp.groupby('time.year').max('time')
#         IC.append(temp)
# merged= xr.merge(IC)
# # print("here")
# print(merged) 
        
        

  
        
       
