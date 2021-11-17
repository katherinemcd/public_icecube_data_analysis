import numpy as np
import matplotlib.pyplot as plt
import scipy 
import scipy.signal

# Load up the IceCube data
icecube_data = np.load("processed_data/output_icecube_data.npz", allow_pickle=True)
data_sigmas = np.array(icecube_data["data_sigmas"])
data_ra = np.array(icecube_data["data_ra"])
data_dec = np.array(icecube_data["data_dec"])

data_ra = data_ra[data_sigmas != 0.0]
data_dec = data_dec[data_sigmas != 0.0]
data_sigmas = data_sigmas[data_sigmas != 0.0]

data_sin_dec = np.sin(data_dec)

plt.figure()
plt.title("3 Year IceCube Data, $\sin(\delta)$, N="+str(len(data_dec))+" Events")
plt.hist(data_sin_dec, range=(-1, 1), bins=50)
plt.xlabel("$\sin(\delta)$")
plt.xlim(-1.0, 1.0)
plt.plot([np.sin(np.deg2rad(-87.0)), np.sin(np.deg2rad(-87.0))], [0.0, 10000.0], color="red")
plt.plot([np.sin(np.deg2rad(87.0)), np.sin(np.deg2rad(87.0))], [0.0, 10000.0], color="red")

print(np.min(np.deg2rad(data_dec)), np.max(np.deg2rad(data_dec)))

convert_data_dec = np.deg2rad(data_dec)

thetas_r = np.linspace(-np.pi/2.0, np.pi/2.0, 100)

sin_dec_o_interest = np.linspace(np.sin(np.deg2rad(-87.0)), np.sin(np.deg2rad(87.0)), 1000)

sin_dec_o_interest_counts = np.zeros(len(sin_dec_o_interest))
for i in range(len(sin_dec_o_interest)):
    #sin_dec_o_interest_counts[i] += np.sum(np.abs(np.arcsin(data_sin_dec) - np.arcsin(sin_dec_o_interest[i])) < np.deg2rad(3.0)) / (2.0 * np.sin(np.deg2rad(3.0)) * np.cos(np.arcsin(sin_dec_o_interest[i])))
    #sin_dec_o_interest_counts[i] += np.sum(np.abs(np.arcsin(data_sin_dec) - np.arcsin(sin_dec_o_interest[i])) < np.deg2rad(1.0)) / (2.0 * np.sin(np.deg2rad(1.0)) * np.cos(np.arcsin(sin_dec_o_interest[i])))
    sin_dec_o_interest_counts[i] += np.sum(np.abs(np.arcsin(data_sin_dec) - np.arcsin(sin_dec_o_interest[i])) < np.deg2rad(0.1)) / (2.0 * np.sin(np.deg2rad(0.1)) * np.cos(np.arcsin(sin_dec_o_interest[i])))
    #sin_dec_o_interest_counts[i] += np.sum(np.abs(np.arcsin(data_sin_dec) - np.arcsin(sin_dec_o_interest[i])) < np.deg2rad(10.0)) / (2.0 * np.sin(np.deg2rad(10.0)) * np.cos(np.arcsin(sin_dec_o_interest[i])))
    
plt.figure()
counts, bins, stuff = plt.hist(data_sin_dec, range=(-np.pi/2.0, np.pi/2.0), bins=180, density=True)

smooth_counts = scipy.signal.savgol_filter(counts, 29, 2, mode='mirror')
bin_half_width = (bins[1] - bins[0]) / 2.0
x_counts = bins[1:] - bin_half_width

# Now, gotta normalize this bad boy 
f_smooth_counts = scipy.interpolate.interp1d(x_counts, smooth_counts, kind='cubic', bounds_error=False, fill_value='extrapolate')

count_norm, count_err = scipy.integrate.quad(f_smooth_counts, -1.0, 1.0)

smooth_counts /= count_norm
smooth_counts /= 2.0 * np.pi
f_smooth_counts = scipy.interpolate.interp1d(x_counts, smooth_counts, kind='cubic', bounds_error=False, fill_value=0.0)

x = np.linspace(-1.0, 1.0, 1000)
y = f_smooth_counts(x)

plt.figure()
plt.plot(x, y, label="Smooth")

#np.savez("processed_data/output_icecube_background_count.npz", x = x, y = y)

f_o_interest = scipy.interpolate.interp1d(sin_dec_o_interest, sin_dec_o_interest_counts, kind='cubic', bounds_error=False, fill_value=0.0)
count_norm_oi, count_err_oi = scipy.integrate.quad(f_o_interest, -1.0, 1.0)
plt.plot(sin_dec_o_interest, sin_dec_o_interest_counts / count_norm_oi / 2.0 / np.pi, label="6 deg thing")
plt.ylim(0.0, 0.2)
plt.legend()

np.savez("processed_data/output_icecube_background_count.npz", x = sin_dec_o_interest, y = sin_dec_o_interest_counts / count_norm_oi / 2.0 / np.pi)


plt.show()
