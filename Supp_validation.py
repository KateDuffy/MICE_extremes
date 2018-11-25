import os
from glob import glob
import xarray as xr 
import platform
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pickle as pkl
import sys
import multiprocessing
from sklearn.externals.joblib import Parallel,delayed
from netCDF4 import Dataset
import netCDF4
import pandas as pd
import multiprocessing
import seaborn as sns
from master_func import *
from tqdm import tqdm


maxima_MICE = try_to_load_as_pickled_object_or_None("MICE_max_85.pkl")
print(maxima_MICE)