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
import multiprocessing
import seaborn as sns
from master_func import *
from joblib import Parallel,delayed
num_cores = multiprocessing.cpu_count()
from tqdm import tqdm


if __name__ == '__main__':

	print(num_cores)
	homedir = os.environ['HOME']
	if platform.system() == 'Linux':
		path = homedir + "/Documents/dev_codes/"

	else:
		path = homedir + "/Documents/dev_codes/"
	file = "MICE_max_85.pkl"
	filepath = path + file

	print(filepath)
	ams = try_to_load_as_pickled_object_or_None(filepath)
	print("loaded")

	file = "MICE_RLs.pkl"
	filepath2 = path + file

	if os.path.exists(filepath2) == False:

		RL_MICE = Parallel(n_jobs = 8, verbose = 50)(delayed(EVT_moving)(i, conversion = 86400*1000) for i in tqdm(ams))

		save_as_pickled_object(RL_MICE, filepath2)
