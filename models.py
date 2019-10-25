# code adapted from : https://github.com/tonyduan/mdn
# MIT license
# please cite accordingly

import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

#---------------------------------------------------------------------------------------
class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


#---------------------------------------------------------------------------------------
class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, num_channel=16, hidden_dim=50,\
                  mask_size=11):
        super().__init__()
        self.n_components = n_components

        self.deconv1 = torch.nn.Conv1d(1, num_channel, mask_size, stride=2)
        self.deconv2 = torch.nn.Conv1d(num_channel, num_channel, mask_size, stride=2)
        self.deconv3 = torch.nn.Conv1d(num_channel, num_channel, mask_size, stride=2)
        self.deconv4 = torch.nn.Conv1d(num_channel, 1, mask_size, stride=2)

        self.batch_norm1 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm2 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm3 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm4 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(1),
                            torch.nn.LeakyReLU()
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(442, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        x = x[:,None,:]
        x = self.deconv1(x)
        x = self.batch_norm1(x)
        x = self.deconv2(x)
        x = self.batch_norm2(x)
        x = self.deconv3(x)
        x = self.batch_norm3(x)
        x = self.deconv4(x)
        x = self.batch_norm4(x)[:,0,:]
        params = self.mlp(x)

        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))


#---------------------------------------------------------------------------------------
class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, num_channel=16, hidden_dim=50, mask_size=11):
        super().__init__()

        self.deconv1 = torch.nn.Conv1d(1, num_channel, mask_size, stride=2)
        self.deconv2 = torch.nn.Conv1d(num_channel, num_channel, mask_size, stride=2)
        self.deconv3 = torch.nn.Conv1d(num_channel, num_channel, mask_size, stride=2)
        self.deconv4 = torch.nn.Conv1d(num_channel, 1, mask_size, stride=2)

        self.batch_norm1 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm2 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm3 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_channel),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm4 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(1),
                            torch.nn.LeakyReLU()
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(442, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = x[:,None,:]
        x = self.deconv1(x)
        x = self.batch_norm1(x)
        x = self.deconv2(x)
        x = self.batch_norm2(x)
        x = self.deconv3(x)
        x = self.batch_norm3(x)
        x = self.deconv4(x)
        x = self.batch_norm4(x)[:,0,:]
        params = self.mlp(x)
        return OneHotCategorical(logits=params)
