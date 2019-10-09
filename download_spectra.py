import numpy as np
import sys
import os
import subprocess
import copy
import astropy.io.fits as pyfits
import time

# dr16
master_path = "data.sdss.org/sas/apogeework/apogee/spectro/redux/r12/stars/"

# load catalog
catalog = pyfits.getdata("../allStar-r12-l33.fits")
catalog_id = catalog['APOGEE_ID'].astype("str")

print(catalog_id.shape)

# initiate a batch
batch = int(sys.argv[1])

# restore missing index
temp = np.load("ind_missing.npz")
ind_missing = temp["ind_missing"]

# loop over all spectra
for i in range(int(4e3)):

    # download spectra
    apogee_id = catalog_id[ind_missing[i+int(batch*4e3)]]
    field = catalog['FIELD'][ind_missing[i+int(batch*4e3)]]
    loc_id = catalog['LOCATION_ID'][ind_missing[i+int(batch*4e3)]]
    telescope = catalog['TELESCOPE'][ind_missing[i+int(batch*4e3)]]

    if (telescope == 'lco25m'):
        filename = 'asStar-r12-%s.fits' % apogee_id.strip()
    else:
        filename = 'apStar-r12-%s.fits' % apogee_id.strip()

    if (telescope == 'apo1m'):
        filepath = os.path.join(master_path,'apo1m', field.strip(), filename)
    elif (telescope == 'apo25m'):
        filepath = os.path.join(master_path,'apo25m', field.strip(), filename)
    elif (telescope == 'lco25m'):
        filepath = os.path.join(master_path,'lco25m', field.strip(), filename)

#    # download spectrum
    os.system("wget --user=<username> --password=<password> " + filepath)
