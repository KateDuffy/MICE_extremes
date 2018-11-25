import os
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
from joblib import Parallel, delayed
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()
import seaborn as sns
from master_func import *

homedir = os.environ['HOME']
 
reference_grid_path = []
mme_paths = []


path = homedir + "/Documents/dev_codes/"
file = "MME_cleaned.pkl"
filepath = path + file

path = homedir + "/Documents/dev_codes/"
file = "Directories.pkl"
directorynames = path + file


	

if os.path.exists(filepath) == False:
	if platform.system() == 'Linux':
	    mme_path = homedir + "/Documents/cmip5_download/cmip5_download/cmip5_pr/rcp85/"
	    ref_obs_path = homedir + "/Documents/dev_codes/"


	else:
	    mme_path = homedir + "/Documents/cmip5_download/cmip5_pr/rcp85/"
	    ref_obs_path = homedir + "/Documents/dev_codes/"


	ref_file_name = "interpolated_prcp.nc"

	comp_ref  = ref_obs_path + ref_file_name

	ref_obs = xr.open_dataset(comp_ref)

	masker = (ref_obs.isel(time=0))
	mask = (xr.where(masker >= 0, 1, np.nan))


	mask = mask.rename({'observation':'pr'})

	start_lat = mask['lat'][0].values
	end_lat = mask['lat'][-1].values



	start_lon = mask['lon'][0].values
	end_lon = mask['lon'][-1].values
	conversion = 1

	MME_directories = mme_path + "*/"
	MME_directories = (glob(MME_directories))
	

	all_ensembles =  []
	test = []
	names = []

	for num,i in enumerate(MME_directories):
		
		nc_paths = i +"*.nc"
		a = (glob(nc_paths))
		opened_ensemble = []
		
		vname = 'pr'

		for num2,j in enumerate(a):
			nc = netCDF4.Dataset(j)
			h = nc.variables[vname]
			times = nc.variables['time']
			t_cal = times.calendar
			jd = netCDF4.num2date(times[:],times.units,calendar=times.calendar)
			

			if t_cal != "360_day" :

				if num2 ==0:
					print(i)

					names.append(i)
				
				jd = [np.datetime64(x) for x in jd]
				nc.close()
				temp = xr.open_dataset(j)
				temp.time.values = jd
				
				temp = temp['pr'].sel(lat = slice(start_lat,end_lat),lon = slice(start_lon,end_lon), time = slice('2006-01-02','2099-12-31'))
				interpolated = temp.interp_like(mask)
				masked = interpolated*mask
				masked = masked*conversion
				opened_ensemble.append(masked)
				
			#masked_max = masked.groupby('time.year').max('time')
		if len(opened_ensemble)>0:
			merged_ensemble = xr.merge(opened_ensemble)
			all_ensembles.append(merged_ensemble)
		


	save_as_pickled_object(all_ensembles, filepath)
	save_as_pickled_object(names, directorynames)



path = homedir + "/Documents/dev_codes/"
file = "MME_extremes_new.pkl"
filepath2 = path + file	


if os.path.exists(filepath2) == False:
	MMEs = try_to_load_as_pickled_object_or_None(filepath)
	print(MMEs)
	print("loaded")
	extremes = []
	for num,i in enumerate(MMEs):
		extremes.append(generate_maximas(i))


	#extremes = Parallel(n_jobs=num_cores)(delayed(generate_maximas)(i) for i in MMEs)
	save_as_pickled_object(extremes, filepath2)

		#all_ensembles.append(merged_ensemble)
else:
	a = try_to_load_as_pickled_object_or_None(filepath2)




#a = try_to_load_as_pickled_object_or_None(filepath)




		# 	jd = netCDF4.num2date(times[:],times.units,calendar=times.calendar)
	# 	jd = [str(x) for x in jd]
	# 	#print(netCDF4. DateFromJulianDay(jd[0], calendar='standard'))
	# 	nc.close()
	# 	jd = [np.datetime64(x) for x in jd]
	# a = (str(jd[0]))
		
	# print(a)
	# print(np.datetime64(a))



		# temp = xr.open_dataset(j)
		# print(j)
		# df = (temp.to_dataframe())
		# #df.time = pd.to_datetime(df.time,error = 'coerce')
		# print(df)
		# temp = xr.decode_cf(temp)		
		# print(temp['time'])

		# 
		# print(temp)
		# #temp = temp['pr'].sel(lat = slice(start_lat,end_lat),lon = slice(start_lon,end_lon))
		# #print(temp['time'].values)
# 		interpolated = temp.interp_like(mask)
# 		masked = interpolated*mask
# 		print(masked)
# 		masked_max = masked.groupby('time.year').max('time')
# 		opened_ensemble.append(masked_max)	
	#merged_ensemble = xr.merge(opened_ensemble)	
# 	all_ensembles.append(merged_ensemble)

# print(all_ensembles)


# plt.show()
# path = homedir + "/Documents/MME_extremes/"

# file = "MME_extremes_extracted"

# filepath = path + file

# save_as_pickled_object(all_ensembles, filepath)



# # print("merging")

