import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import glob
#import healpy as hp
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.optimize import curve_fit
#import ROOT 
import copy 

matplotlib.rcParams.update({'font.size': 14})

#likelihood_map = np.load("calculated_likelihood_map.npy", allow_pickle=True)
#likelihood_map = np.load("../result_data/calculated_fit_likelihood_map_allsky_with_HESE.npy", allow_pickle=True)
likelihood_map = np.load("../result_data/calculated_fit_likelihood_map_allsky_with_INJECTED.npy", allow_pickle=True)

fill_value = 0.0

data_map_pos = copy.deepcopy(likelihood_map)
data_map_pos[data_map_pos <= 0.0] = fill_value
data_map_neg = copy.deepcopy(likelihood_map)
data_map_neg[data_map_pos > 0.0] = fill_value

likelihood_map = data_map_pos

#likelihood_map *= np.log(10.0) # Made a mistake, this fixes it
#step_size = 0.2
#step_size = 0.5
#step_size = 0.2
step_size = 1.0
every_pt_dec = np.arange(-90.0, 90, step_size)
every_pt_ra  = np.arange(0.0, 360, step_size)
dec_num = len(every_pt_dec)
ra_num = len(every_pt_ra)

every_pt = np.ones((ra_num, dec_num, 2))
for iX in range(ra_num):
    for iY in range(dec_num):
        every_pt[iX][iY] = [every_pt_ra[iX], every_pt_dec[iY]]

def gauss_function(x, a, mean, sigma):
    return a*np.exp(-0.5 * np.power((x - mean)/ sigma, 2.0))

stuff_to_plot = np.sqrt(2.0 * likelihood_map)
stuff_to_plot[np.abs(every_pt[:,:,1]) < -87.0] = fill_value

plt.figure()
#counts, bins, stuff = plt.hist(stuff_to_plot.flatten(), range=(-0.0, 6.0), bins=60, log=True)
print("Biggest thing is:", np.max(stuff_to_plot.flatten()[np.isnan(stuff_to_plot.flatten()) == False]))
counts, bins, stuff = plt.hist(stuff_to_plot.flatten(), range=(-0.0, 6.0), bins=60, log=True)
xs = np.array(bins)[1:] - (bins[1] - bins[0]) / 2.0
ys = np.array(counts)[:]

# Going to let ROOT handle this fitting 
#hist = ROOT.TH1F("hist", "hist", 60, 0.0, 6.0)
#for i in range(1, hist.GetNbinsX()):
#    hist.SetBinContent(i, counts[i-1])

#f1 = ROOT.TF1("f1","gaus",0.0, 10.0)
#f1.SetParameter(0, 50000.0);
#f1.SetParameter(1, 0.0);
#f1.SetParameter(2, 1.0);

#f1.FixParameter(1, 0.0);

#hist.Fit("gaus", "L", "", xs[0], xs[-1]);
#hist.Fit("f1", "LB", "", xs[0], 2.5);
#hist.Fit("f1", "LB", "", 1.0, 2.0);

#g = hist.GetListOfFunctions().FindObject("gaus");
#rfit_constant = f1.GetParameter(0);
#rfit_mean = f1.GetParameter(1);
#rfit_sigma = f1.GetParameter(2);

#print("Fit paramas, c, mean, sig =", rfit_constant, rfit_mean, rfit_sigma)

#hist.Draw()
#input("Press enter to kill plots.")
#exit()

#popt, pcov = curve_fit(gauss_function, xs, ys, p0 = [np.max(ys), 1.0])
#print("Fit done, popt=", popt)
#plt.close()

x_new = np.linspace(0.0, 6.0, 1000)

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.errorbar(np.array(bins)[1:] - (bins[1] - bins[0]) / 2.0, counts, xerr=6.0/2.0/float(len(counts)), yerr=np.sqrt(counts), color="black", label="Observed (All Sky)", fmt='.')
#ax.plot(x_new, gauss_function(x_new, rfit_constant, rfit_mean, rfit_sigma), color="blue", label="Normal Distribution")
#ax.plot(x_new, gauss_function(x_new, *popt), color="blue", label="Gaussian Fit")
#ax.plot(x_new, 2.5 * gauss_function(x_new, *popt), color="blue", label="Gaussian Fit")

ax.set_ylim(0.5, 2.0 * counts[1])
ax.set_xlim(0.0, 6.0)
ax.set_xlabel("$\sqrt{2 \Delta \ln \mathcal{L}}$", labelpad=-1)
#ax.set_title("All Sky Point Search in 0.2$^\circ$ Steps of 3 year IceCube Data")
ax.grid()
ax.legend()
#plt.savefig("output_fit_allsky_histogram.pdf")
#plt.savefig("output_fit_allsky_histogram.png")

# Residual time
plt.figure()
fig, ax = plt.subplots()
#ax.plot(x_new, , color="blue", label="Gaussian Fit")
#ax.plot(x_new, gauss_function(x_new, *popt), color="blue", label="Gaussian Fit")
#ax.plot(x_new, 2.5 * gauss_function(x_new, *popt), color="blue", label="Gaussian Fit")
x_new = np.array(bins)[1:] - (bins[1] - bins[0]) / 2.0
#ax.errorbar(x_new, np.array(counts) - gauss_function(x_new, rfit_constant, rfit_mean, rfit_sigma), xerr=6.0/2.0/float(len(counts)), yerr=np.sqrt(counts), color="black", label="Observation", fmt='.')
ax.set_ylim(-50.0, 50.0)
ax.set_xlim(0.0, 6.0)
ax.set_xlabel("$\sqrt{2 \Delta \ln \mathcal{L}}$")
ax.set_ylabel("Fit Residual: Data - Fit")
#ax.set_title("Fit Residual - All Sky Point Search in 0.2$^\circ$ Steps of 3 year IceCube Data")
ax.legend()
ax.grid()
#plt.savefig("output_fit_allsky_histogram_res.pdf")
#plt.savefig("output_fit_allsky_histogram_res.png")

#plt.show()
#exit()


N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, 0/256, N)
vals[:, 1] = np.linspace(1, 0/256, N)
vals[:, 2] = np.linspace(1, 100/256, N)
newcmp = ListedColormap(vals)

#stuff_to_plot = np.sqrt(2.0 * likelihood_map)
#stuff_to_plot = likelihood_map
#stuff_to_plot[np.abs(every_pt[:,:,1]) > 87.0] = 0.0

find_the_best = stuff_to_plot.flatten()

find_the_best, index_of_the_best = zip(*sorted(zip(find_the_best, range(len(find_the_best)))));
find_the_best = np.flip(find_the_best, 0)
index_of_the_best = np.flip(index_of_the_best, 0)
def cheat_round(stuff, factor=1):
    return float(int(stuff * 10.0 * factor))/(10.0 * factor)
for i in range(20):
    print(cheat_round(find_the_best[i], 10.0),"&",
          cheat_round(every_pt[:,:,0].flatten()[index_of_the_best[i]]), "&",
          cheat_round(every_pt[:,:,1].flatten()[index_of_the_best[i]]), "\\\\")
    print("\hline")


plt.figure()
ax = plt.subplot(111, projection="aitoff")
#ax.set_xticklabels([''])
plot_o_interest = ax.pcolormesh(np.deg2rad(every_pt[:,:,0])-np.pi, np.deg2rad(every_pt[:,:,1]), np.power(stuff_to_plot, 2.0), cmap=newcmp, vmin=0.0, vmax=5.0)
#plot_o_interest = ax.pcolormesh(np.deg2rad(every_pt[:,:,0])-np.pi, np.deg2rad(every_pt[:,:,1]), stuff_to_plot, cmap='gnuplot2', vmin=0.0, vmax=5.0)
cbar0 = plt.colorbar(plot_o_interest, orientation="horizontal")
#cbar0.set_label("$\sqrt{2 \Delta \ln \mathcal{L}}$")
cbar0.set_label("$2 \Delta \ln \mathcal{L}$")
#plt.savefig("output_fit_allsky_map_aitoff.pdf")
#plt.savefig("output_fit_allsky_map_aitoff.png", dpi=600, quality=95)

plt.figure()
plt.imshow(np.flip(np.power(stuff_to_plot, 2.0).transpose(), axis=0), cmap=newcmp, extent=(-180.0, 180.0, -90.0, 90.0), vmin=0.0, vmax=5.0)
plt.xlabel("RA [$^\circ$]")
plt.ylabel("$\delta$ [$^\circ$]")
cbar1 = plt.colorbar(orientation="horizontal")
cbar1.set_label("$\sqrt{2 \Delta \ln \mathcal{L}}$")
#plt.savefig("output_fit_allsky_map_cart.pdf")
#plt.savefig("output_fit_allsky_map_cart.png", dpi=600, quality=95)


plt.show()
