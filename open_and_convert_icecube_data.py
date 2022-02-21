import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import glob


data_files = glob.glob("./data/IC*.csv")

data_day = np.array([])
data_sigmas = np.array([])
data_ra = np.array([])
data_dec = np.array([])

for data_file_name in data_files:
    print("Loading filename: %s" % data_file_name)
    f = open(data_file_name)

    data = np.loadtxt(data_file_name, dtype='float')
    data_day = np.append(data_day, data[:, 0])
    data_sigmas = np.append(data_sigmas, data[:, 2])
    data_ra = np.append(data_ra, data[:, 3])
    data_dec = np.append(data_dec, data[:, 4])

data_sigmas = np.deg2rad(data_sigmas)
data_ra = np.deg2rad(data_ra)
data_dec = np.deg2rad(data_dec)

selection = np.abs(data_dec) < np.deg2rad(87.0)
data_day = data_day[selection]
data_sigmas = data_sigmas[selection]
data_ra = data_ra[selection]
data_dec = data_dec[selection]

np.savez("processed_data/output_icecube_data.npz",
         data_day=data_day,
         data_sigmas=data_sigmas,
         data_ra=data_ra,
         data_dec=data_dec)
