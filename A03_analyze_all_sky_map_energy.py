import time
import numpy as np
import matplotlib.pyplot as plt
import IceCubeAnalysis_energy as IceCubeAnalysis
from multiprocessing import Pool


def main(icecube_file_name, background_file_name, output_file_names,
         step_size=15, n_cpu=None):
    """
    Performs the all-sky source search. The script breaks the sky into
    a grid, with step between points defined by `step_size`. For each 
point,
    we find the most likely value of astrophysical neutrinos from the
    source at the given point. Creates a map of the max-likelihood and
    most-likely number of neutrinos from each point.
    Parameters
    ----------
    icecube_file_name : str
        IceCube pickle file location.
    background_file_name : str
        File location of pre-processed background PDF.
    output_file_names : array_like
        Outpnp.savez("./processed_data/background_energies_muon.npz",
        data_eng = data_eng,
        data_dec = data_dec,
        data_ra = data_ra)
ut file names for fitted values of likelihood
        (0th entry) and n_s (1st entry).
    step_size : float
        The degrees step size to perform the all-sky search.
    n_cpu : int
        The number of CPUs to use in the parallelization.
        If n_cpu is None, the computation is not parallelized.
    """

    use_parallel = (n_cpu is not None)

    sourcesearch_ = IceCubeAnalysis.SourceSearch(icecube_file_name)
    sourcesearch_.load_background(background_file_name)

    #  This is the coordinate of each point on the sky we are checking.
    cord_s, ra_len, dec_len = IceCubeAnalysis.prepare_skymap_coordinates(step_size)

    N_sky_pts = len(cord_s)

    print("Number of IceCube events: \t %i" % sourcesearch_.N)
    print("Number of skypoints to calc: \t %i" % N_sky_pts)

    start_time = time.time()

    if(use_parallel):
        pool = Pool(n_cpu)

        args_for_multiprocessing = [(np.array(cord_s[i_source]), i_source) for i_source in range(N_sky_pts)]
        results = pool.starmap(sourcesearch_.job_submission,
                               args_for_multiprocessing)

        pool.close()
    else:
        results = []
        for i_source in range(N_sky_pts):
            results += [sourcesearch_.job_submission(cord_s[i_source],
                                                     i_source)]

    end_time = time.time()

    if(use_parallel):
        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))

    results_ = [list(t) for t in zip(*results)]
    ns = results_[0]
    del_ln_L = results_[1]
    #eng_prob = np.array(results_[2])
    #print(np.average(eng_prob).flatten())
    n_s_map = np.reshape(ns, (ra_len, dec_len))
    data_map = np.reshape(del_ln_L, (ra_len, dec_len))
    np.save(output_file_names[0], data_map)
    np.save(output_file_names[1], n_s_map)


if(__name__ == "__main__"):
    icecube_file_name = "./processed_data/output_icecube_data_energy.npz"
    background_file_name = "./processed_data/output_icecube_background_energy.npz"
    output_file_names = ["./processed_data/calculated_fit_likelihood_map_allsky_energy.npy",
                         "./processed_data/calculated_fit_ns_map_allsky_energy.npy"]
    main(icecube_file_name, background_file_name, output_file_names, step_size=0.2, n_cpu=20)
