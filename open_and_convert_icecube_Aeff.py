import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.integrate
import scipy.interpolate
import glob

#alpha = 2.2
alpha = 2.5

# And now, effective area
data_files = glob.glob("./data/IC86-2012-TabulatedAeff.txt")

data_E = []
data_cos_zenith = []
data_Aeff = []
for data_file_name in data_files:
    f = open(data_file_name)

    for i, line in enumerate(f):
        if(i == 0):
            continue

        if(i % 10000 == 0):
            print("Line", i,"of",data_file_name)

        line_ = line.split(" ")
        line_ = np.array(line_)[np.array(line_) != '']

        data_E          += [(float(line_[0]) + float(line_[1])) / 2.0]
        data_cos_zenith += [(float(line_[2]) + float(line_[3])) / 2.0]
        data_Aeff += [float(line_[4][:-1])]
        
data_E = np.array(data_E) / 1000.0 # convert to TeV
data_cos_zenith = np.array(data_cos_zenith)
data_Aeff = 10000.0 * np.array(data_Aeff) # convert to cm^2

data_cos_zenith *= -1.0 # cos(theta) = - sin(Declination)


# So, now have to integrate Aeff as function of declination
unique_cos_zenith = np.unique(data_cos_zenith)
x_cos_dec_steps = []
y_integrate_steps = []

for iUnique_cos_zenith in unique_cos_zenith:
    cur_E_max = data_E[data_cos_zenith == iUnique_cos_zenith]
    cur_Aeff = data_Aeff[data_cos_zenith == iUnique_cos_zenith]

    cur_E_max = cur_E_max[cur_Aeff != 0.0]
    cur_Aeff  = cur_Aeff[cur_Aeff != 0.0]
        
    # functional time
    f = scipy.interpolate.interp1d(cur_E_max, np.power(cur_E_max, -alpha) * cur_Aeff, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    integrated_Aeff, int_Aeff_error = scipy.integrate.quad(f, np.min(data_E), np.max(data_E), limit=5000)

    x_cos_dec_steps += [iUnique_cos_zenith]
    y_integrate_steps += [integrated_Aeff]

    print(np.min(cur_E_max), np.max(cur_E_max), iUnique_cos_zenith, integrated_Aeff, int_Aeff_error)

# sort steps, just in case
x_cos_dec_steps, y_integrate_steps = zip(*sorted(zip(x_cos_dec_steps, y_integrate_steps)))
x_cos_dec_steps = np.array(x_cos_dec_steps)
y_integrate_steps = np.array(y_integrate_steps)

# Interpolate evenly
x_cos_dec_steps = x_cos_dec_steps[y_integrate_steps > 0.0]
y_integrate_steps = y_integrate_steps[y_integrate_steps > 0.0]
f_Aeff_dec_integration = scipy.interpolate.interp1d(x_cos_dec_steps, y_integrate_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")

np.savez("processed_data/output_icecube_AffIntegrated.npz", cos_dec = x_cos_dec_steps, Aeffintegrated=y_integrate_steps)

# Now just save one slice, in case its useful
select_array = np.logical_and((data_E < 32.0).astype(bool), (data_E > 28.0).astype(bool))
data_cos_zenith = data_cos_zenith[select_array]
data_Aeff = data_Aeff[select_array]
data_E = data_E[select_array]

# Sort 
data_cos_zenith, data_Aeff = zip(*sorted(zip(data_cos_zenith, data_Aeff)))
data_cos_zenith = np.array(data_cos_zenith)
data_Aeff = np.array(data_Aeff)

f_Aeff = scipy.interpolate.interp1d(data_cos_zenith, data_Aeff, kind='quadratic', bounds_error=False, fill_value="extrapolate")

new_cos_zenith = np.linspace(-1.0, 1.0, 1000)
new_Aeff = f_Aeff(new_cos_zenith)

np.savez("processed_data/output_icecube_Aff_30Tev.npz", cos_dec = new_cos_zenith, Aeff=new_Aeff)

new_cos_dec_steps = np.linspace(-1.0, 1.0, 1000)
new_integrated = f_Aeff_dec_integration(new_cos_dec_steps)

new_integrated = savgol_filter(new_integrated, 101, 3) # Smooth it out

plt.figure()
plt.title("Aeff Integrated Over Energy")
plt.semilogy(x_cos_dec_steps, y_integrate_steps)
plt.semilogy(new_cos_dec_steps, new_integrated)
plt.ylabel("$A_{eff}$ [m$^2$]")
plt.xlabel("$\sin \delta $")

plt.figure()
plt.title("Aeff at 30 TeV")
plt.semilogy(data_cos_zenith, data_Aeff)
plt.semilogy(new_cos_zenith, new_Aeff)
plt.ylabel("$A_{eff}$ [m$^2$]")
plt.xlabel("$\sin \delta $")
plt.show()

exit()
