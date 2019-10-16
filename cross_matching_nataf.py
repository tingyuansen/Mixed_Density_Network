import numpy as np
import pandas as pd
import sys
import os
import subprocess
import copy
import astropy.io.fits as pyfits
import time

# load catalog
catalog = pyfits.getdata("allStar-r12-l33.fits")
catalog_id = catalog['APOGEE_ID'].astype("str")

# load cross-match catalog
df = pd.read_csv("Temp_APOGEE_IDs.txt")
df = np.array(df).ravel()

# loop over all spectra
for i in range(df.shape):

    # search for ID
    ind = np.where(df[i] == catalog_id)
    # download spectra
    apogee_id = catalog_id[ind]
    field = catalog['FIELD'][ind]
    loc_id = catalog['LOCATION_ID'][ind]
    telescope = catalog['TELESCOPE'][ind]

    if (telescope == 'lco25m'):
        filename = 'asStar-r12-%s.fits' % apogee_id.strip()
    else:
        filename = 'apStar-r12-%s.fits' % apogee_id.strip()

    # copy files
    os.system("cp apogee_dr16/" + filename + " apogee_dr16_nataf")
