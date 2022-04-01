import glob
import scipy.interpolate
import scipy.integrate
import numpy as np


def load_Aeff(input_file_names, output_file_name, alpha=2.5):
    
    data_E = np.array([])
    data_zenith = np.array([])
    data_Aeff = np.array([])

    for data_file in input_file_names:
        data = np.loadtxt(data_file)
        
        data_E = np.append(data_E, (data[:, 0] + data[:, 1]) / 2.0)
        data_zenith = np.append(data_zenith, (data[:, 2] + data[:, 3]) / 2.0)
        data_Aeff = np.append(data_Aeff, data[:, 4])
        

    data_E = np.power(10.0, np.array(data_E)) / 1000.0  # convert to TeV
    data_cos_zenith = np.cos(np.deg2rad(np.array(data_zenith)))
    data_Aeff = 10000.0 * np.array(data_Aeff) # convert to cm^2

    data_cos_zenith *= -1.0 # cos(theta) = - sin(Declination)

    # So, now have to integrate Aeff as function of declination
    unique_cos_zeniths = np.unique(data_cos_zenith)
    x_cos_dec_steps = np.zeros(len(unique_cos_zeniths))
    y_integrate_steps = np.zeros(len(unique_cos_zeniths))

    for i_unique_cos_zenith, unique_cos_zenith in enumerate(unique_cos_zeniths):

        cur_E_max = data_E[data_cos_zenith == unique_cos_zenith]
        cur_Aeff = data_Aeff[data_cos_zenith == unique_cos_zenith]

        cur_E_max = cur_E_max[cur_Aeff != 0.0]
        cur_Aeff  = cur_Aeff[cur_Aeff != 0.0]
              
        # functional time
        f_integrand = scipy.interpolate.interp1d(cur_E_max,
                                                 np.power(cur_E_max, -alpha) * cur_Aeff,
                                                 kind='linear',
                                                 bounds_error=False,
                                                 fill_value="extrapolate")
    
        integrated_Aeff, int_Aeff_error = scipy.integrate.quad(f_integrand,
                                                               np.min(data_E),
                                                               np.max(data_E),
                                                               limit=5000)
        
        x_cos_dec_steps[i_unique_cos_zenith] = unique_cos_zenith
        y_integrate_steps[i_unique_cos_zenith] = integrated_Aeff

        print("%i \t %.2f \t %.2f \t %.2f \t %.2f " % (i_unique_cos_zenith,
                                                       unique_cos_zenith,
                                                       np.min(cur_E_max),
                                                       np.max(cur_E_max),
                                                       integrated_Aeff))

    # sort steps, just in case
    x_cos_dec_steps, y_integrate_steps = zip(*sorted(zip(x_cos_dec_steps, y_integrate_steps)))
    x_cos_dec_steps = np.array(x_cos_dec_steps)
    y_integrate_steps = np.array(y_integrate_steps)

    x_cos_dec_steps = x_cos_dec_steps[y_integrate_steps > 0.0]
    y_integrate_steps = y_integrate_steps[y_integrate_steps > 0.0]

    x_cos_dec_steps = x_cos_dec_steps[np.logical_not(np.isinf(y_integrate_steps))]
    y_integrate_steps = y_integrate_steps[np.logical_not(np.isinf(y_integrate_steps))]

    np.savez(output_file_name,
             cos_dec=x_cos_dec_steps,
             Aeffintegrated=y_integrate_steps)
        
    return True

if(__name__ == "__main__"):

    input_file_names = glob.glob("./data/icecube_10year_ps/irfs/*effectiveArea.csv")
    alpha = 2.5

    output_file_name = "./processed_data/output_icecube_AffIntegrated_%s.npz" % alpha
    
    f_Aeff_dec_integration = load_Aeff(input_file_names, output_file_name, alpha)
