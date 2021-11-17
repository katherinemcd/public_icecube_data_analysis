import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import multiprocessing as mp
#from multiprocessing import Pool
import copy
from iminuit import Minuit
from pprint import pprint # we use this to pretty print some stuff later
import datetime 
import cProfile
import pstats


def load_icecube_data(file_name):
    
    # Load up the IceCube data
    icecube_data = np.load(file_name, allow_pickle=True)

    data_sigmas = np.array(icecube_data["data_sigmas"])
    data_ra = np.array(icecube_data["data_ra"])
    data_dec = np.array(icecube_data["data_dec"])

    # get rid of entries that will cause trouble in probability calculations
    # and things too close to the poles
    selection_region = np.logical_and(data_sigmas != 0.0, np.abs(data_dec) < np.deg2rad(87.0))
    data_ra = data_ra[selection_region]
    data_dec = data_dec[selection_region]
    data_sigmas = data_sigmas[selection_region]

    return data_ra, data_dec, data_sigmas

def prepare_skymap_coordinates(step_size):
    every_pt_dec = np.arange(-np.pi/2.0, np.pi/2.0, step_size)
    every_pt_ra  = np.arange(0.0, 2.0 * np.pi, step_size)

    dec_num = len(every_pt_dec)
    ra_num = len(every_pt_ra)
    
    every_pt = np.ones((ra_num, dec_num, 2))
    
    for iX in range(len(every_pt[:,0,0])):
        for iY in range(len(every_pt[0,:,0])):
            every_pt[iX][iY] = [every_pt_ra[iX], every_pt_dec[iY]]

    return every_pt
            
def load_background_map(file_name, every_pt):
    data_bg = np.load(file_name, allow_pickle=True)
    data_bg_x = data_bg['x']
    data_bg_y = data_bg['y']
    f_bg = scipy.interpolate.interp1d(data_bg_x, data_bg_y, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    B_i = f_bg(every_pt[:,:,1])
    return B_i
    
def Si_likelihood(N, cart_x_i_, cart_y_i_, cart_z_i_, cart_x_s_, cart_y_s_, cart_z_s_, sigma_i):

    norm_i = np.sqrt(np.power(cart_x_i_, 2.0) + np.power(cart_y_i_, 2.0) + np.power(cart_z_i_, 2.0))
    norm_s = np.sqrt(np.power(cart_x_s_, 2.0) + np.power(cart_y_s_, 2.0) + np.power(cart_z_s_, 2.0))

    dists_great = cart_x_i_ * cart_x_s_ + cart_y_i_ * cart_y_s_ + cart_z_i_ * cart_z_s_
    dists_great /= norm_i * norm_s

    dists_great = np.arccos(dists_great)

    S_i = 1.0 / (2.0 * np.pi * sigma_i * sigma_i)
    S_i *= np.exp(-0.5* np.power(dists_great / sigma_i, 2.0))

    return S_i

def calculate_likelihood(N, S_i, B_i):
    def _calculate_likelihood(n_s):
        return -calculate_likelihood_given_parameters(n_s, N, S_i, B_i)
    return _calculate_likelihood

def calculate_likelihood_given_parameters(n_s, N, S_i, B_i):
    return np.sum(np.log(n_s/N * S_i + (1.0 - n_s / N) * B_i))

def big_job_submission(N, B_i, cart_x_i, cart_y_i, cart_z_i, data_sigmas, cart_x_s, cart_y_s, cart_z_s, iJob, pixel_x, pixel_y):
    
    pixels_Si = Si_likelihood(N, cart_x_i, cart_y_i, cart_z_i, cart_x_s[pixel_x][pixel_y], cart_y_s[pixel_x][pixel_y], cart_z_s[pixel_x][pixel_y], data_sigmas)
    pixels_Bi = B_i[pixel_x][pixel_y]

    m = Minuit(calculate_likelihood(N, pixels_Si, pixels_Bi),
               n_s = 0.0)
    m.limits = [(0, N)]
    m.migrad() # finds minimum of least_squares function
    
    n_s = m.values[0] # fitted value of n_s for this spot
    del_ln_L = (calculate_likelihood_given_parameters(n_s, N, pixels_Si, pixels_Bi) - calculate_likelihood_given_parameters(0.0, N, pixels_Si, pixels_Bi))

    #if(iJob % 5 == 0):
    print("Done with", iJob, pixel_x, pixel_y, "n_s=", n_s)

    return n_s, del_ln_L


def main():

    step_size = np.deg2rad(20.0) # 0.2 deg
    
    data_ra, data_dec, data_sigmas = load_icecube_data("./processed_data/output_icecube_data.npz")    
    
    N = len(data_sigmas) # number of total events

    every_pt = prepare_skymap_coordinates(step_size) # this is the coordinate of each point on the sky we are checking

    # loading up the background probability
    B_i = load_background_map("processed_data/output_icecube_background_count.npz", every_pt)
    
    # Preprocess the data to speed up calls to Si_likelihood
    # In the equations in the paper, these are i indices, the index of data
    x_i_ = np.array(tuple(zip(data_ra, data_dec)))

    # In the equations in the paper, these are s indices, the index of source direction
    x_s_ = copy.deepcopy(every_pt)

    # convert to cartesian from angular to make distance calculations easier (but less accurate)
    cart_x_i = np.sin(np.pi/2.0 - x_i_[:,1]) * np.cos(x_i_[:,0])
    cart_y_i = np.sin(np.pi/2.0 - x_i_[:,1]) * np.sin(x_i_[:,0])
    cart_z_i = np.cos(np.pi/2.0 - x_i_[:,1])

    cart_x_s = np.sin(np.pi/2.0 - x_s_[:,:,1]) * np.cos(x_s_[:,:,0])
    cart_y_s = np.sin(np.pi/2.0 - x_s_[:,:,1]) * np.sin(x_s_[:,:,0])
    cart_z_s = np.cos(np.pi/2.0 - x_s_[:,:,1])

    #pool = Pool(10)
    parallel_results = []

    print(every_pt[:,:,0].shape)
    
    data_map = np.zeros(every_pt[:,:,0].shape)
    n_s_map = np.zeros(every_pt[:,:,0].shape)

    i_to_ix_map = np.zeros(len(data_map.flatten()))
    i_to_iy_map = np.zeros(len(data_map.flatten()))
    running_i = 0
    for iX in range(len(every_pt[:,0,0])):
        for iY in range(len(every_pt[0,:,0])):

            if(np.abs(np.rad2deg(every_pt[0,:,0][iY])) > 87.0):
                continue

            i_to_ix_map[running_i] = iX
            i_to_iy_map[running_i] = iY
            #parallel_results += [pool.apply_async(big_job_submission, [N, B_i, cart_x_i, cart_y_i, cart_z_i, data_sigmas, cart_x_s, cart_y_s, cart_z_s, running_i, iX, iY])]
            parallel_results += [big_job_submission(N, B_i, cart_x_i, cart_y_i, cart_z_i, data_sigmas, cart_x_s, cart_y_s, cart_z_s, running_i, iX, iY)]
            running_i += 1

    for i in range(len(parallel_results)):
        #ns, del_ln_L = parallel_results[i].get()
        ns, del_ln_L = parallel_results[i]#.get()
        n_s_map[int(i_to_ix_map[i])][int(i_to_iy_map[i])] = ns
        if(ns >= 0.0):
            data_map[int(i_to_ix_map[i])][int(i_to_iy_map[i])] = del_ln_L
        else:
            data_map[int(i_to_ix_map[i])][int(i_to_iy_map[i])] = -del_ln_L

    #pool.close()

    #if(save):
    #    np.save("result_data/calculated_fit_likelihood_map_allsky.npy", data_map)
    #    np.save("result_data/calculated_fit_ns_map_allsky.npy", n_s_map)
    plt.figure()
    plt.title("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    data_map_pos = data_map[data_map > 0.0]
    data_map_neg = data_map[data_map < 0.0]
    data_map_zero = data_map[data_map == 0.0]
    #plt.hist(np.append(np.sqrt(2.0 * data_map_pos.flatten()), -np.sqrt(2.0 * -data_map_neg.flatten())), range=(-6.0, 6.0), bins=120, log=True)
    plt.hist(np.sqrt(2.0 * data_map_pos.flatten()), range=(-6.0, 6.0), bins=120, log=True)
    plt.hist(-np.sqrt(2.0 * -data_map_neg.flatten()), range=(-6.0, 6.0), bins=120, log=True)
    plt.hist(np.sqrt(2.0 * data_map_zero.flatten()), range=(-6.0, 6.0), bins=120, log=True)         
    
    plt.figure()
    plt.title("$n_s$")
    plt.hist(n_s_map.flatten(), range=(-30.0, 30.0), bins=60, log=True)
    
    plt.figure()
    plt.title("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    #plt.imshow(np.sqrt(2.0 * np.abs(data_map)).transpose())
    plt.imshow(np.sqrt(2.0 * np.abs(data_map)).transpose())
    
    # Where are the zeros at.
    data_map_zero = copy.deepcopy(data_map)
    data_map_zero[data_map_zero == 0] = 1e10
    plt.figure()
    plt.imshow(data_map_zero.transpose())
    
    plt.figure()
    plt.title("$n_s$")
    plt.imshow(n_s_map.transpose())
    
    plt.show()

    
if __name__ == "__main__":

    '''
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    #cProfile.run('main()')
    '''    

    main()
