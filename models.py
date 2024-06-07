import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

EPS = 1e-8

def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


class LatentGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, latent_dim, hidden=(256, 256), state_dependet_std=False, device_id=-1):
        super(LatentGaussianActor, self).__init__()

        if device_id == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

        self.l1_latent = nn.Linear(latent_dim, state_dim)
        self.l1 = nn.Linear(state_dim*2, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3_mu = nn.Linear(hidden[1], action_dim)

        if state_dependet_std:
            self.l3_log_std = nn.Linear(hidden[1], action_dim)
        else:
            self.log_std = torch.zeros(action_dim).to(self.device)

        self.log_std_bounds = (-5., 0.)

        self.max_action = max_action
        self.state_dependet_std = state_dependet_std

    def forward(self, state, latent):
        z = F.relu(self.l1_latent(latent))
        sz = torch.cat([state, z], 1)
        a = F.relu(self.l1(sz))
        a = F.relu(self.l2(a))
        mu = self.max_action * torch.tanh(self.l3_mu(a))
        if self.state_dependet_std:
            log_std = self.max_action * torch.tanh(self.l3_log_std(a))
            log_std = soft_clamp(log_std, *self.log_std_bounds)
        else:
            log_std = self.log_std
        std = log_std.exp()

        dist = D.Normal(mu, std)
        action = dist.rsample()
        log_pi = dist.log_prob(action).sum(dim=-1)
        # log_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=1)
        #
        # action = torch.tanh(action)
        # mu = torch.tanh(mu)

        return mu, log_pi.view(-1, 1), action

    def get_log_prob(self, state, action, latent):
        z = F.relu(self.l1_latent(latent))
        sz = torch.cat([state, z], 1)
        a = F.relu(self.l1(sz))
        a = F.relu(self.l2(a))
        mu = self.max_action * torch.tanh(self.l3_mu(a))
        if self.state_dependet_std:
            log_std = self.max_action * torch.tanh(self.l3_log_std(a))
            log_std = soft_clamp(log_std, *self.log_std_bounds)
        else:
            log_std = self.log_std

        log_std = soft_clamp(log_std, *self.log_std_bounds)
        std = log_std.exp()

        dist = D.Normal(mu, std)

        log_prob = dist.log_prob(action)
        if len(log_prob.shape) == 1:
            return log_prob
        else:
            return log_prob.sum(-1, keepdim=True)


class LatentCritic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(LatentCritic, self).__init__()

        # Q1 architecture
        self.l1_latent = nn.Linear(latent_dim, state_dim)
        self.l1 = nn.Linear(state_dim*2 + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4_latent = nn.Linear(latent_dim, state_dim)
        self.l4 = nn.Linear(state_dim*2 + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action, latent):
        z1 = F.relu(self.l1_latent(latent))

        saz1 = torch.cat([state, action, z1], 1)

        q1 = F.relu(self.l1(saz1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        z2 = F.relu(self.l4_latent(latent))
        saz2 = torch.cat([state, action, z2], 1)

        q2 = F.relu(self.l4(saz2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, latent):
        z1 = F.relu(self.l1_latent(latent))
        saz1 = torch.cat([state, action, z1], 1)
        q1 = F.relu(self.l1(saz1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_cont_dim, latent_disc_dim, temperature=.67, hidden=(256, 256), device_id=-1):
        super(VAE, self).__init__()

        # Encoder architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        if latent_cont_dim > 0:
            self.l3_cont_mu = nn.Linear(hidden[1], latent_cont_dim)
            self.l3_cont_log_std = nn.Linear(hidden[1], latent_cont_dim)
        if latent_disc_dim > 0:
            self.l3_disc = nn.Linear(hidden[1], latent_disc_dim)
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim

        self.l4 = nn.Linear(latent_disc_dim + latent_cont_dim, hidden[0])
        self.l5 = nn.Linear(hidden[0], hidden[1])
        self.l6 = nn.Linear(hidden[1], state_dim + action_dim)

        self.temperature = temperature
        self.log_std_bounds = (-5., 0.)

        if device_id == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        ----------
        adopted from https://github.com/Schlumberger/joint-vae/
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).to(self.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1).to(self.device)
            return one_hot_samples

    def sample_normal(self, mean, log_std):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        ----------
        adopted from https://github.com/Schlumberger/joint-vae/
        """
        if self.training:
            std = torch.exp(log_std)
            eps = torch.zeros(std.size()).normal_()
            eps = eps.to(self.device)
            return mean + std * eps
        else:
            # Reconstruction mode
            return


    def forward(self, state, action):
        z, z_cont, z_disc, z_mu, z_log_std, alpha = self.encode(state, action)

        h_dec = F.relu(self.l4(z))
        h_dec = F.relu(self.l5(h_dec))
        sa_pred = self.l6(h_dec)

        return z, z_cont, z_disc, z_mu, z_log_std, alpha, sa_pred

    def encode(self, state, action):
        z = None
        z_cont = None
        z_disc = None
        alpha = None
        z_mu = None
        z_log_std = None
        log_p = None
        sa = torch.cat([state, action], 1)
        h = F.relu(self.l1(sa))
        h = F.relu(self.l2(h))
        if self.latent_cont_dim > 0:
            z_mu= self.l3_cont_mu(h)
            z_log_std = self.l3_cont_log_std(h)
            z_cont = self.sample_normal(z_mu, z_log_std)

            log_std = soft_clamp(z_log_std, *self.log_std_bounds)
            std = log_std.exp()
            dist = D.Normal(z_mu, std)

            log_prob = dist.log_prob(z_cont)
            if len(log_prob.shape) == 1:
                log_p = log_prob
            else:
                log_p = log_prob.sum(-1, keepdim=True)

        if self.latent_disc_dim > 0:
            alpha = F.softmax(self.l3_disc(h), dim=1)
            z_disc = self.sample_gumbel_softmax(alpha)

        if self.latent_disc_dim < 1:
            z = z_cont
        elif self.latent_cont_dim < 1:
            z = z_disc
        else:
            z = torch.cat([z_cont, z_disc], 1)

        return z, z_cont, z_disc, z_mu, z_log_std, log_p

    def kl_discrete_loss(self, alpha, non_reduction=False):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)]).to(self.device)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)

        if non_reduction:
            return torch.sum(neg_entropy, dim=1, keepdim=True) + log_dim

        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy

        return kl_loss.mean()

    def kl_normal_loss(self, mean, log_std, non_reduction=False):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + 2*log_std - mean.pow(2) - torch.exp(2*log_std))

        if non_reduction:
            return torch.sum(kl_values, dim=1, keepdim=True)

        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)

        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        return kl_loss
    
    def get_latent_log_prob(self, state, action, latent):
        sa = torch.cat([state, action], 1)
        h = F.relu(self.l1(sa))
        h = F.relu(self.l2(h))

        if self.latent_cont_dim > 0:
            z_mu= self.l3_cont_mu(h)
            z_log_std = self.l3_cont_log_std(h)

            log_std = soft_clamp(z_log_std, *self.log_std_bounds)
            std = log_std.exp()

            dist = D.Normal(z_mu, std)

            log_prob = dist.log_prob(latent)
            if len(log_prob.shape) == 1:
                return log_prob
            else:
                return log_prob.sum(-1, keepdim=True)
        else:
            return None


