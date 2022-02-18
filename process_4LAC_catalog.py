import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units


def open_and_convert_catalog(file_name, output_file_name):
    hdul = fits.open(file_name)

    print(hdul.info())
    print(hdul[1].columns.info())

    # Its possible this isn't right, but I think it is
    cat_names = hdul[1].data.field(0).tolist()
    cat_RA = hdul[1].data.field(1).tolist()
    cat_Dec = hdul[1].data.field(2).tolist()
    cat_flux1000 = hdul[1].data.field(6).tolist()
    cat_z = hdul[1].data.field(30).tolist()
    cat_type = hdul[1].data.field(19).tolist()
    cat_var_index = hdul[1].data.field(34).tolist()

    np.savez(output_file_name,
             cat_names=cat_names,
             cat_RA=cat_RA,
             cat_Dec=cat_Dec,
             cat_type=cat_type,
             cat_flux1000=cat_flux1000,
             cat_z=cat_z,
             cat_var_index=cat_var_index)

    plt.figure()
    plt.hist(cat_var_index,
             log=True,
             range=(0.0, 100.0),
             bins=100)
    plt.xlabel("Measured Variability Index of Objects in Catalog")

    plt.figure()
    plt.scatter(cat_RA, cat_Dec)
    plt.xlabel("RA")
    plt.ylabel("Dec.")

    coords = SkyCoord(ra=cat_RA,
                      dec=cat_Dec,
                      unit='degree')
    ra = coords.ra.wrap_at(180 * units.deg).radian
    dec = coords.dec.radian
    color_map = plt.cm.Spectral_r

    fig = plt.figure(figsize=(6, 4))
    fig.add_subplot(111, projection='aitoff')
    image = plt.hexbin(ra, dec,
                       cmap=color_map,
                       gridsize=512,
                       mincnt=1,
                       bins='log')
    plt.xlabel('R.A.')
    plt.ylabel('Decl.')
    plt.grid(True)
    plt.colorbar(image, spacing='uniform', extend='max')

    plt.figure()
    plt.hist(cat_flux1000,
             log=True,
             range=(-1.0e-9, 1.0e-9),
             bins=1000)
    plt.xlabel("Measured Flux of Objects in Catalog")

    plt.show()


if(__name__ == "__main__"):

    open_and_convert_catalog("./data/table_4LAC.fits",
                             "./processed_data/4LAC_catelogy.npz")
