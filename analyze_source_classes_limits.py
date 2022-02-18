import copy
import glob
import scipy.integrate
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool


def Si_likelihood(N, cart_i, cart_s, sigma_i):
    """
    Calculates the signal PDF at a given
    point in the sky.

    Parameters
    ----------
    N : int
        The total number of data events.
    cart_i : array_like
        The cartesian coordinates of data events.
    cart_s : array_like
        The cartesian position on sky that is being tested.
    sigma_i : array_like
        The spatial standard deviation, in degrees
    """

    norm_i = np.sqrt(np.sum(np.power(cart_i, 2.0), axis=0))
    norm_s = np.sqrt(np.sum(np.power(cart_s, 2.0)))
    great_dists = np.dot(cart_s, cart_i)
    great_dists /= norm_i*norm_s
    great_dists = np.arccos(great_dists)

    S_i = 1.0 / (2.0 * np.pi * sigma_i * sigma_i)
    S_i *= np.exp(-0.5 * np.power(great_dists / sigma_i, 2.0))

    return S_i


def calculate_likelihood(n_s, N, S_i, B_i):
    """
    Calculates the test statistic for a given
    number of clustered neutrinos (n_s) and
    given signal pdf (S_i), background pdf (B_i),
    and the total number of events (N).
    """
    return np.sum(np.log(n_s/N * S_i + (1.0 - n_s / N) * B_i))


def test_statistic_at_point(cart_i, cart_s, B_i, n_s):
    """
    Calculates the test statistic at point

    Parameters
    ----------
    cart_i : array_like
        The cartesian coordinates of data events.
    cart_s : array_like
        The cartesian position on sky that is being tested.
    B_i : float
        The background number of events at that point.
    n_s : int
        The number of neutrinos being tested for that point
    """

    Si = Si_likelihood(N, cart_i, cart_s, data_sigmas)

    del_ln_L_n_s = calculate_likelihood(n_s, N, Si, B_i)
    del_ln_L_0 = calculate_likelihood(0.0, N, Si, B_i)

    return 2.0 * (del_ln_L_n_s - del_ln_L_0)


def load_catalog(file_name, allowed_names):

    catelog_data = np.load(file_name, allow_pickle=True)
    cat_ra = catelog_data["cat_RA"]
    cat_dec = catelog_data["cat_Dec"]
    cat_names = catelog_data["cat_names"]
    cat_type = catelog_data["cat_type"]
    cat_flux1000 = catelog_data["cat_flux1000"]

    allowed_names_mask = np.zeros(len(cat_dec))
    for i in range(len(allowed_names)):
        allowed_names_mask[cat_type == allowed_names[i]] = 1

    allowed = np.logical_and(np.abs(cat_dec) < 87.0,
                             allowed_names_mask)

    cat_ra = cat_ra[allowed]
    cat_dec = cat_dec[allowed]
    cat_names = cat_names[allowed]
    cat_flux1000 = cat_flux1000[allowed]

    cat_ra = np.deg2rad(cat_ra)
    cat_dec = np.deg2rad(cat_dec)

    return cat_ra, cat_dec, cat_names, cat_flux1000


def load_background(file_name, zenith_angles):
    data_bg = np.load(file_name, allow_pickle=True)
    f_bg = scipy.interpolate.interp1d(data_bg['x'],
                                      data_bg['y'],
                                      kind='cubic',
                                      bounds_error=False,
                                      fill_value="extrapolate")
    B_i = f_bg(zenith_angles)
    return B_i


def load_weights(weights_type, cat_flux1000):
    if(weights_type == 'flat'):
        cat_flux_weights = 1e-9 * np.ones(len(cat_flux1000))
    elif(weights_type == 'flux'):
        cat_flux_weights = cat_flux1000
    elif(weights_type == 'dist'):
        cat_flux_weights = np.zeros(len(cat_flux1000))
        cat_flux_weights = 1e-2 / np.power(cat_DL, 2.0)
        cat_flux_weights[cat_DL[i] == -10.] = 0.0
    else:
        print("Weights not known: %s" % weights)
        exit()
    return cat_flux_weights


def load_Aeff(file_name):

    icecube_Aeff_integrated = np.load(file_name, allow_pickle=True)

    f_Aeff_dec_integration = scipy.interpolate.interp1d(icecube_Aeff_integrated['cos_dec'],
                                                        icecube_Aeff_integrated['Aeffintegrated'],
                                                        kind='cubic',
                                                        bounds_error=False,
                                                        fill_value="extrapolate")

    return f_Aeff_dec_integration


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


def calculate_span(E1, E2, alpha, cat_flux_weights, n_entries=30):
    sum_of_interest = 0
    for i_source in range(len(cat_flux_weights)):
        sum_of_interest += (np.power(E1 / E2, alpha)
                            * np.power(E2, 2.0)
                            * cat_flux_weights[i_source]
                            / (4.0 * np.pi))

    para_min = 1e-13 / sum_of_interest
    para_max = 1e-9 / sum_of_interest

    para_span = np.power(10.0, np.linspace(np.log10(para_min), np.log10(para_max), n_entries))
    return para_span


if(__name__ == "__main__"):

    # Parameters of the problem
    alpha = 2.5
    allowed_names = ['BLL', 'FSRQ']
    weights_type = 'flat'

    # The time used in integration, in seconds
    T = (1.0 * 365.25 * 24.0 * 3600.0)

    # The energy bounds used in integration.
    E1 = 100.0
    E2 = 30.0

    # Load IceCube data
    data_ra, data_dec, data_sigmas = load_icecube_data("processed_data/output_icecube_data_2010.npz")
    data_sin_dec = np.sin(data_dec)
    
    # Load integrate detector affective volume, a function of zenith angle
    f_Aeff_dec_integration = load_Aeff("processed_data/output_icecube_AffIntegrated_2010_%s.npz" % alpha)

    # Load catalog data
    cat_ra, cat_dec, cat_names, cat_flux1000 = load_catalog("./processed_data/4LAC_catelogy.npz", allowed_names)
    cat_sin_dec = np.sin(cat_dec)

    # Load background PDF
    B_i = load_background("./processed_data/output_icecube_background_count_2010.npz",
                          cat_sin_dec)

    # Load flux weights used when setting limits.
    cat_flux_weights = load_weights(weights_type, cat_flux1000)

    N = len(data_sigmas)
    N_sources = len(cat_ra)

    print("Number of Sources:\t %i" % len(cat_ra))
    print("Number of Events:\t %i" % N)

    # The cartesian positions of all data events.
    # Easier for great circle distance calculations.
    cart_x_i = np.sin(np.pi/2.0 - data_dec) * np.cos(data_ra)
    cart_y_i = np.sin(np.pi/2.0 - data_dec) * np.sin(data_ra)
    cart_z_i = np.cos(np.pi/2.0 - data_dec)
    cart_i = np.array([cart_x_i, cart_y_i, cart_z_i])

    # The cartesian positions of all sources in the catalog.
    # Easier for great circle distance calculations
    cart_x_s = np.sin(np.pi/2.0 - cat_dec) * np.cos(cat_ra)
    cart_y_s = np.sin(np.pi/2.0 - cat_dec) * np.sin(cat_ra)
    cart_z_s = np.cos(np.pi/2.0 - cat_dec)
    cart_s = np.array([cart_x_s, cart_y_s, cart_z_s])

    # Calculate the points that we will then loop over
    parameterized_span = calculate_span(E1, E2, alpha, cat_flux_weights)

    sweep_test_stats = np.zeros(len(parameterized_span))
    sweep_flux = np.zeros(len(parameterized_span))

    test_stat_each_source = np.zeros((N_sources, len(parameterized_span)))

    for i_given_para, given_para in enumerate(parameterized_span):

        current_flux = 0.0
        current_test_stat = 0.0

        parallel_results = np.zeros(N_sources)
        for i_source in range(N_sources):
            given_ns = given_para * cat_flux_weights[i_source] * T * np.power(E1, alpha) * f_Aeff_dec_integration(cat_sin_dec[i_source])
            current_flux += given_para * cat_flux_weights[i_source] * np.power(E1 / E2, alpha)

            parallel_results[i_source] = test_statistic_at_point(cart_i,
                                                                 cart_s[:, i_source],
                                                                 B_i[i_source],
                                                                 given_ns)

        for i_source in range(N_sources):
            current_test_stat_ = parallel_results[i_source]

            test_stat_each_source[i_source, i_given_para] = current_test_stat_

            current_test_stat += current_test_stat_

        sweep_test_stats[i_given_para] = current_test_stat
        sweep_flux[i_given_para] = np.power(E2, 2.0) * current_flux / (4.0 * np.pi)

        print(given_para, sweep_flux[i_given_para], sweep_test_stats[i_given_para])

    calculated_values = sweep_flux != 0

    plt.semilogx(sweep_flux[calculated_values],
                 sweep_test_stats[calculated_values],
                 color="black",
                 label="3 yr. IceCube Data")

    for i_source in range(N_sources):
        plt.semilogx(sweep_flux[calculated_values],
                     test_stat_each_source[i_source, calculated_values],
                     color="red",
                     alpha=0.1)

    plt.axhline(-3.85,
                color="red",
                linestyle="--",
                label="95% Confidence Level")

    plt.xlabel("F_v, [TeV / cm^2 / s / sr] at 30 TeV")
    plt.ylabel("$2 \Delta \ln \mathcal{L}$")

    plt.xlim(np.min(sweep_flux[calculated_values]),
             np.max(sweep_flux[calculated_values]))
    plt.ylim(-10.0, 10.0)

    plt.grid()
    plt.legend()

    plt.show()
