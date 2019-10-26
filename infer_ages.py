# import packages
import numpy as np
import time

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter


# load models
model = torch.load("models.pt",  map_location=lambda storage, loc: storage) # load in cpu
model.eval()


#==============================================================================
# loop over all apogee data
temp = np.load("../2apogee_rc_region_spectra_0.npz")
apogee_spectra = temp["spectra"]

for i in range(12):
    temp = np.load("../2apogee_rc_region_spectra_" + str(i+1) + ".npz")
    apogee_spectra = np.vstack([apogee_spectra,temp["spectra"]])

# restore catalog id
temp = np.load("../apogee_unique.npz")
ind_unique = temp["ind_unique"]
apogee_spectra = apogee_spectra[ind_unique,:].T
print(apogee_spectra.shape)


#==============================================================================
# extract input and output arrays
input_array = np.copy(apogee_spectra).T
x_valid = input_array[:,:]
x_valid_count = x_valid.shape[0]


#==============================================================================
# scale the labels
x_valid = x_valid-0. # convert to the write float format
x_valid = torch.Tensor(x_valid)

#-----------------------------------------------------------------------------
# loop over all batches
y_category = []
y_normal_loc = []
y_normal_std = []

batch_size = 10000
num_batch = x_valid.shape[0]//batch_size
print(num_batch)

for i in range(num_batch+1):
    print(i)
    y_category_temp, y_normal = model(x_valid[i*batch_size:(i+1)*batch_size,:])
    y_category.extend(np.exp(y_category_temp._param.detach().numpy()))
    y_normal_loc.extend(y_normal.mean.detach().numpy()[:,:,0] -3.)
    y_normal_std.extend(y_normal.stddev.detach().numpy()[:,:,0])
y_category = np.array(y_category)
y_normal_loc = np.array(y_normal_loc)
y_normal_std = np.array(y_normal_std)
print(y_category.shape)
print(y_normal_loc.shape)
print(y_normal_std.shape)

#-----------------------------------------------------------------------------
# the dominant mode
y_category_mode = np.argmax(y_category, axis=1)
y_pred = np.array([y_normal_loc[i,y_category_mode[i]] \
                   for i in range(y_normal_loc.shape[0])])
y_error = np.array([y_normal_std[i,y_category_mode[i]] \
                   for i in range(y_normal_std.shape[0])])

print(y_pred.shape)
np.savez("results.npz", y_pred = y_pred, y_error = y_error)
