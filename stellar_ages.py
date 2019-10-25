# import package
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

from models import MixtureDensityNetwork

# set cuda
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')


#==============================================================================
# restore catalog
temp = np.load("../source_apogee_rc_spectra.npz")
apogee_spectra = temp["apogee_spectra"].T
apogee_spectra_err = temp["apogee_spectra_err"].T
apogee_log_age = temp["log_age"]
#apogee_id = temp["apogee_id"]
apogee_payne_teff = temp["teff"]

#-----------------------------------------------------------------------------
# extract input and output arrays
input_array = np.copy(apogee_spectra).T
output_array = np.vstack([apogee_log_age]).T

x = input_array[:2000,:]
y = output_array[:2000,:]
y_train = np.copy(y)

x_valid = input_array[2000:,:]
y_valid = output_array[2000:,:]
x_valid_count = x_valid.shape[0]


#==============================================================================
# scale the labels
x = x-0.
x_valid = x_valid-0. # convert to the write float format

#mu_y = np.mean(y, axis=0)
#std_y = np.std(y, axis=0)
#y = (y-mu_y)/std_y
#y_valid = (y_valid-mu_y)/std_y

y = y-0.
y_valid = y_valid - 0.

#-----------------------------------------------------------------------------
# make pytorch variables
x = torch.Tensor(x)
y = torch.Tensor(y)
x.cuda()
y.cuda()

x_valid = torch.Tensor(x_valid)
y_valid = torch.Tensor(y_valid)
x_valid.cuda()
y_valid.cuda()

#==============================================================================
# define model
dim_in = 7214
dim_out = 1
model = MixtureDensityNetwork(dim_in, dim_out, n_components=1)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# number of iterations to run
n_iterations = 2000

# batch size
batch_size = 512

#-----------------------------------------------------------------------------
# train batches
nsamples = x.shape[0]
nbatches = nsamples // batch_size

# initiate counter
current_loss = np.inf
training_loss =[]
validation_loss = []

#-----------------------------------------------------------------------------
# train the network
for e in range(int(n_iterations)):

    # randomly permute the data
    perm = torch.randperm(nsamples)
    perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        loss = model.loss(x[idx], y[idx]).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

#--------------------------------------------------------------------------------------------
    # check the validation loss
    if e % 100 == 0:
        loss_valid = model.loss(x_valid, y_valid).mean()
        print('iter %s:' % e, 'training loss = %.3f' % loss,\
              'validation loss = %.3f' % loss_valid)

        loss_data = loss.detach().data.item()
        loss_valid_data = loss_valid.detach().data.item()
        training_loss.append(loss_data)
        validation_loss.append(loss_valid_data)

        # record the weights and biases if the validation loss improves
        if loss_valid_data < current_loss:
            current_loss = loss_valid_data
            torch.save(model, "../models.pt")

#--------------------------------------------------------------------------------------------
# save the final training loss
np.savez("training_loss.npz",\
         training_loss = training_loss,\
         validation_loss = validation_loss)
