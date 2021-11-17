import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import glob

data_files = glob.glob("./data/IC*-events.txt")
data_sigmas = []
data_ra = []
data_dec = []
data_day = []

for data_file_name in data_files:
    f = open(data_file_name)

    for i, line in enumerate(f):
        if(i == 0):
            #print(line)
            continue
        
        if(i % 10000 == 0):
            print("Line", i,"of",data_file_name)
        
        line_ = line.split(" ")
        line_ = np.array(line_)[np.array(line_) != '']
        data_day += [float(line_[0])]
        data_sigmas += [float(line_[2])]
        data_ra += [float(line_[3])]
        data_dec += [float(line_[4])]

#plt.hist(data_day, 1085)
#plt.hist(data_day, 2000)
#plt.show()
#exit()
        
data_sigmas = np.deg2rad(np.array(data_sigmas))
data_ra     = np.deg2rad(np.array(data_ra))
data_dec    = np.deg2rad(np.array(data_dec))
    
data_sigmas = data_sigmas[np.abs(data_dec) < np.deg2rad(87.0)]
data_ra     = data_ra[np.abs(data_dec) < np.deg2rad(87.0)]
data_dec    = data_dec[np.abs(data_dec) < np.deg2rad(87.0)]
data_sin_dec = np.sin(data_dec)

np.savez("processed_data/output_icecube_data.npz", data_sigmas=data_sigmas, data_ra=data_ra, data_dec=data_dec, data_sin_dec=data_sin_dec)
