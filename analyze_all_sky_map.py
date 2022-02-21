import time
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from iminuit import Minuit


def Si_likelihood(N, cart_s):
    """
    Calculates the signal PDF at a given
    point in the sky.

    Parameters
    ----------
    N : int
        The total number of data events.
    cart_s : array_like
        The cartesian position on sky that is being tested.
    """

    norm_i = np.sqrt(np.sum(np.power(cart_i, 2.0), axis=0))
    norm_s = np.sqrt(np.sum(np.power(cart_s, 2.0)))
    great_dists = np.dot(cart_s, cart_i)
    great_dists /= norm_i*norm_s
    great_dists = np.arccos(great_dists)

    S_i = 1.0 / (2.0 * np.pi * data_sigmas * data_sigmas)
    S_i *= np.exp(-0.5 * np.power(great_dists / data_sigmas, 2.0))

    return S_i


def calculate_likelihood(n_s, N, S_i, B_i):
    """
    Calculates the test statistic for a given
    number of clustered neutrinos (n_s) and
    given signal pdf (S_i), background pdf (B_i),
    and the total number of events (N).
    """
    if(n_s < 0):
        return 0.0
    else:
        return np.sum(np.log(n_s/N * S_i + (1.0 - n_s / N) * B_i))


def load_background(file_name, zenith_angles):
    data_bg = np.load(file_name, allow_pickle=True)
    f_bg = scipy.interpolate.interp1d(data_bg['x'],
                                      data_bg['y'],
                                      kind='cubic',
                                      bounds_error=False,
                                      fill_value="extrapolate")
    B_i = f_bg(zenith_angles)
    return B_i


def load_icecube_data(file_name):
    icecube_data = np.load(file_name, allow_pickle=True)

    data_sigmas = np.array(icecube_data["data_sigmas"])
    data_ra = np.array(icecube_data["data_ra"])
    data_dec = np.array(icecube_data["data_dec"])

    allowed_entries = data_sigmas != 0.0
    data_ra = data_ra[allowed_entries]
    data_dec = data_dec[allowed_entries]
    data_sigmas = data_sigmas[allowed_entries]

    return data_ra, data_dec, data_sigmas


def big_job_submission(N, B_i, cart_s, i_source):

    S_i = Si_likelihood(N, cart_s)

    def _calculate_likelihood(n_s):
        return -calculate_likelihood(n_s, N, S_i, B_i)

    m = Minuit(_calculate_likelihood,
               n_s=0.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits = [(0, N)]

    m.migrad()  # finds minimum of least_squares function

    n_s = m.values[0]  # fitted value of n_s for this spot
    del_ln_L = (calculate_likelihood(n_s, N, S_i, B_i) - calculate_likelihood(0.0, N, S_i, B_i))

    if(i_source % 1000 == 0):
        print("%i) \t n_s = \t %f" % (i_source, n_s))

    return n_s, del_ln_L


def parallel_with_start_stop(N, B_i, cart_s, i_job_start, i_job_stop):
    results = np.zeros((i_job_stop - i_job_start, 2))
    for i in range(len(B_i)):
        results[i] = big_job_submission(N,
                                        B_i[i],
                                        cart_s[:, i],
                                        i_job_start + i)
    return results


def prepare_skymap_coordinates(step_size):
    """
    Returns the RA and Dec for each point, and a map with the index
    """

    ra_sweep = np.arange(0.0, 2.0 * np.pi, step_size)
    dec_sweep = np.arange(-np.pi/2.0, np.pi/2.0, step_size)

    ra_len = len(ra_sweep)
    dec_len = len(dec_sweep)

    total_pts = dec_len * ra_len

    index_map = np.zeros((total_pts, 2), dtype='int')

    ras = np.zeros(total_pts)
    decs = np.zeros(total_pts)

    i_source = 0
    for iX in range(ra_len):
        for iY in range(dec_len):
            index_map[i_source] = [iX, iY]
            ras[i_source] = ra_sweep[iX]
            decs[i_source] = dec_sweep[iY]
            i_source += 1

    return ras, decs, index_map, ra_len, dec_len


def main():

    use_parallel = True
    n_cpu = 20
    # chunk_size = 250
    step_size = np.deg2rad(10.0)  # Degrees step on the sky

    global data_sigmas
    data_ra, data_dec, data_sigmas = load_icecube_data("./output_icecube_data.npz")
    data_sin_dec = np.sin(data_dec)

    #  This is the coordinate of each point on the sky we are checking.
    cat_ra, cat_dec, index_map, ra_len, dec_len = prepare_skymap_coordinates(step_size)
    cat_sin_dec = np.sin(cat_dec)

    N = len(data_sigmas)  # Number of total events
    N_sky_pts = len(cat_ra)

    print("Number of IceCube events: \t %i" % N)
    print("Number of skypoints to calc: \t %i" % N_sky_pts)

    # loading up the background probability
    B_i = load_background("./output_icecube_background_count.npz",
                          cat_sin_dec)

    # Preprocess the data to speed up calls to Si_likelihood
    # In the equations in the paper, these are i indices, the index of data
    cart_x_i = np.sin(np.pi/2.0 - data_dec) * np.cos(data_ra)
    cart_y_i = np.sin(np.pi/2.0 - data_dec) * np.sin(data_ra)
    cart_z_i = np.cos(np.pi/2.0 - data_dec)

    global cart_i
    cart_i = np.array([cart_x_i, cart_y_i, cart_z_i])

    # In the equations in the paper, these are s indices, the index of source direction
    cart_x_s = np.sin(np.pi/2.0 - cat_dec) * np.cos(cat_ra)
    cart_y_s = np.sin(np.pi/2.0 - cat_dec) * np.sin(cat_ra)
    cart_z_s = np.cos(np.pi/2.0 - cat_dec)
    cart_s = np.array([cart_x_s, cart_y_s, cart_z_s])

    results = []

    start_time = time.time()

    if(use_parallel):
        pool = Pool(n_cpu)

        args_for_multiprocessing = [(N, B_i[i_source], np.array(cart_s[:, i_source]), i_source) for i_source in range(N_sky_pts)]
        results = pool.starmap(big_job_submission,
                               args_for_multiprocessing)

        '''
        # so now I have to figure out a way to make it start and stop and pass things along
        args_for_multiprocessing = []
        for start_i_source in range(0, N_sky_pts - chunk_size, chunk_size):
            args_for_multiprocessing += [(N,
                                          B_i[start_i_source: start_i_source + chunk_size],
                                          cart_s[:, start_i_source: start_i_source + chunk_size],
                                          start_i_source,
                                          start_i_source + chunk_size)]
        args_for_multiprocessing += [(N,
                                      B_i[start_i_source + chunk_size: -1],
                                      cart_s[:, start_i_source + chunk_size: -1],
                                      start_i_source + chunk_size,
                                      len(B_i)-1)]

        results = pool.starmap(parallel_with_start_stop,
                               args_for_multiprocessing)

        results = np.array(results)
        results_reshape = np.zeros((N_sky_pts, 2))
        i_source = 0
        for results_ in results:
            for j in range(len(results_)):
                results_reshape[i_source] = results_[j]
                i_source += 1
        results = results_reshape
        '''

        pool.close()
    else:
        for i_source in range(N_sky_pts):
            results += [big_job_submission(N,
                                           B_i[i_source],
                                           cart_s[:, i_source],
                                           i_source)]

    end_time = time.time()

    if(use_parallel):
        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))

    data_map = np.zeros((ra_len, dec_len))
    n_s_map = np.zeros((ra_len, dec_len))

    for i_source in range(N_sky_pts):
        if(np.abs(np.rad2deg(cat_dec[i_source])) > 87.0):
            continue

        ns, del_ln_L = results[i_source]
        i_ra, i_dec = index_map[i_source]
        n_s_map[i_ra, i_dec] = ns
        data_map[i_ra, i_dec] = del_ln_L

    np.save("./calculated_fit_likelihood_map_allsky.npy", data_map)
    np.save("./calculated_fit_ns_map_allsky.npy", n_s_map)

    data_map_pos = data_map[data_map > 0.0]
    data_map_neg = data_map[data_map < 0.0]
    data_map_zero = data_map[data_map == 0.0]

    plt.figure()
    plt.hist(np.sqrt(2.0 * data_map_pos.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Positive TS")
    plt.hist(-np.sqrt(2.0 * -data_map_neg.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Negative TS")
    plt.hist(np.sqrt(2.0 * data_map_zero.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Zero TS")
    plt.xlabel("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.hist(n_s_map.flatten(),
             range=(-30.0, 30.0), bins=60, log=True)
    plt.xlabel("$n_s$")
    plt.grid()

    plt.figure()
    plt.title("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    plt.imshow(np.sqrt(2.0 * np.abs(data_map)).transpose())
    plt.xlabel("RA Index")
    plt.ylabel("Dec Index")

    plt.figure()
    plt.title("$n_s$")
    plt.imshow(n_s_map.transpose())
    plt.xlabel("RA Index")
    plt.ylabel("Dec Index")

    plt.show()


if(__name__ == "__main__"):

    main()
