import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

from models import VAE, LatentGaussianActor, LatentCritic

EPS = 1e-12

def to_one_hot(y, class_num):
	one_hot = np.zeros((y.shape[0], class_num))
	one_hot[range(y.shape[0]), y] = 1.0

	return one_hot

class DiveOff(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_freq=2,
		scale=2.0,
		weight_type='clamp',
		hidden=(256, 256),
		device_id=-1,
		lr=3e-4,
		latent_cont_dim=2,
		latent_disc_dim=0,
		vae_steps=1e5,
		z_width =2.0,
		info_lr_rate=0.2,
		info_alpha=1.0,
		info_weight=True,
		scale_schedule=False,
		T_scale_max=5e5,
		schedule=None,
		Tmax=1e6,
		weighted_q=True,
		v_scale=1.0,
		beta=50,
		wml_alpha=1.0
	):
		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.latent_dim = latent_cont_dim + latent_disc_dim

		self.actor = LatentGaussianActor(state_dim, action_dim, max_action, self.latent_dim, hidden=hidden).to(
			self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		if schedule is not None:
			self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer,
																			  T_max=int(Tmax / policy_freq))

		self.critic = LatentCritic(state_dim, action_dim, self.latent_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.vae = VAE(state_dim, action_dim, latent_cont_dim, latent_disc_dim).to(self.device)

		self.vae_init_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
		self.vae_init_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.vae_init_optimizer,
																			 T_max=int(vae_steps))

		self.info_optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters(),
															   self.vae.parameters()), lr=lr * info_lr_rate)
		if schedule is not None:
			self.info_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.info_optimizer,
																			 T_max=int(Tmax / policy_freq))

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq

		self.scale = scale
		self.weight_type = weight_type

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.latent_cont_dim = latent_cont_dim
		self.latent_disc_dim = latent_disc_dim

		self.z_width = z_width
		self.info_alpha = info_alpha
		self.info_weight = info_weight
		self.scale_schedule = scale_schedule
		self.T_scale_max = T_scale_max
		self.schedule = schedule
		self.weighted_q = weighted_q
		self.beta=beta
		self.v_scale = v_scale
		self.wml_alpha = wml_alpha

		self.total_it = 0
		self.vae_it = 0
		self.actor_it = 0

		print('DiveOff')
		print('scale', self.scale)
		print('info_alpha', info_alpha)
		print('info_lr_rate', info_lr_rate)
		print('latent_cont_dim', latent_cont_dim)
		print('info_weight', info_weight)
		print('scale_schedule', scale_schedule)
		print('schedule', schedule)
		print('weighted_q', weighted_q)
		print('v_scale', v_scale)
		print('wml_alpha', wml_alpha)

	def select_latent_action(self, state, latent, stochastic=False):

		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		latent = torch.FloatTensor(latent.reshape(1, -1)).to(self.device)

		mu, log_std, action = self.actor(state, latent)
		if stochastic:
			return action.cpu().data.numpy().flatten()

		return mu.cpu().data.numpy().flatten()

	def select_action_batch(self, state_batch, z, state_dim, action_dim):
		z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)

		z_rep = z.repeat(state_batch.shape[0], 1)
		action_batch, _, _ = self.actor(state_batch, z_rep)

		print('action', action_batch.cpu().data.numpy().shape)

		return action_batch.cpu().data.numpy()

	def sample_latent(self, replay_buffer, sample_num=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(sample_num)

		z_sample, _, _, z_mu, _, _ = self.vae.encode(state,action)

		return z_sample.cpu().data.numpy(), z_mu.cpu().data.numpy(), state.cpu().data.numpy()

	def compute_vae_loss(self, replay_buffer, batch_size=256, beta=50):
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		z, z_cont, z_disc, z_mu, z_log_std, alpha, sa_pred = self.vae(state, action)
		sa = torch.cat([state, action], 1)
		recon_loss = F.mse_loss(sa_pred, sa.detach())
		kl_cont_loss = 0
		kl_disc_loss = 0
		vae_loss = recon_loss
		if self.latent_cont_dim > 0:
			kl_cont_loss = self.vae.kl_normal_loss(z_mu, z_log_std)
			vae_loss += beta * kl_cont_loss
		if self.latent_disc_dim > 0:
			kl_disc_loss = self.vae.kl_discrete_loss(alpha)
			vae_loss += beta * kl_disc_loss

		return vae_loss, kl_cont_loss
	
	def compute_weighted_vae_loss(self, replay_buffer, weight, batch_size=256, beta=50):
		# Sample replay buffer
		state, action, _, _, _ = replay_buffer.sample(batch_size)

		z, z_cont, z_disc, z_mu, z_log_std, alpha, sa_pred = self.vae(state, action)
		sa = torch.cat([state, action], 1)
		recon_loss = self.weighted_mse_loss(weight.detach(), sa_pred, sa.detach())
		kl_cont_loss = 0
		kl_disc_loss = 0
		vae_loss = recon_loss
		if self.latent_cont_dim > 0:
			kl_cont_loss = self.vae.kl_normal_loss(z_mu, z_log_std, non_reduction=True)
			vae_loss += beta * torch.mean( weight.detach() * kl_cont_loss)
		if self.latent_disc_dim > 0:
			kl_disc_loss = self.vae.kl_discrete_loss(alpha, non_reduction=True)
			vae_loss += beta * torch.mean( weight.detach() * kl_disc_loss )

		return vae_loss, kl_cont_loss

	def train_vae(self, replay_buffer, batch_size=256, beta=50):
		self.vae_it += 1

		vae_loss, kl_cont_loss = self.compute_vae_loss(replay_buffer, batch_size=batch_size, beta=beta)

		self.vae_init_optimizer.zero_grad()
		vae_loss.backward()
		self.vae_init_optimizer.step()
		self.vae_init_scheduler.step()

		if self.vae_it % 1000 == 0:
			print(self.vae_it, ": vae_loss loss", vae_loss.item(), 'kl_cont:', kl_cont_loss.item())

	def train(self, replay_buffer, batch_size=256, envname=''):
		self.total_it += 1
		self.critic.train()

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if 'antmaze' in envname:
			reward = reward - 1

		latent, _, _, _, _, _ = self.vae.encode(state, action)

		with torch.no_grad():
			_, _, next_action = self.actor_target(next_state, latent.detach())
			current_Q1, current_Q2 = self.critic_target(next_state, next_action, latent.detach())
			minQ = torch.min(current_Q1, current_Q2)
			target_Q = reward + not_done * self.discount * minQ

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, latent.detach())
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it %1000 ==0:
			print(self.total_it, ": critic loss", critic_loss.item())

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			with torch.no_grad():

				Q1, Q2 = self.critic_target(state, action, latent.detach())
				Qmin = torch.min(Q1, Q2)

				_, _, action_pred = self.actor_target(state, latent.detach())
				Q1_pred, Q2_pred = self.critic_target(state, action_pred, latent.detach())
				v = torch.min(Q1_pred, Q2_pred)

				adv = Qmin - v
				width = torch.max(adv).detach() - torch.min(adv).detach()

				if self.total_it < self.T_scale_max and self.scale_schedule:
					vae_scale = self.v_scale * ( 1 - np.cos( self.total_it / self.T_scale_max * np.pi ) ) * 0.5
				else:
					vae_scale = self.v_scale

				vae_weight = None
				weight = None
				if self.scale == 0.0:
					weight = adv
					vae_weight = adv
				elif self.weight_type == 'clamp':
					weight = torch.exp(self.scale * (adv)).clamp(0, 100)
					vae_weight = torch.exp(vae_scale * (adv)).clamp(0, 100)
				else:
					weight = torch.exp(self.scale * (adv - adv.max()) / width)
					vae_weight = torch.exp(vae_scale * (adv - adv.max()) / width)
				
				if self.total_it % 10000 == 0:
					print('vae_scale', vae_scale)
					print('vae_weight', vae_weight[:10])

			# Compute actor loss
			log_p = self.actor.get_log_prob(state, action, latent.detach())
			actor_loss = - (log_p * weight.detach() ).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			if self.schedule is not None and self.schedule != 'none':
				self.actor_scheduler.step()

			# latent_pred = self.posterior(state)
			# info_loss = F.mse_loss(latent_pred, latent.detach())
			#
			# self.posterior_optimizer.zero_grad()
			# info_loss.backward()
			# self.posterior_optimizer.step()

			vae_loss, _ = self.compute_weighted_vae_loss(replay_buffer, weight.detach(), batch_size=batch_size)

			if self.weighted_q:
				z_current, _, _, z_mu, z_log_std, log_p_z = self.vae.encode(state, action.detach())

				wml_loss = - (log_p_z * vae_weight.detach() ).mean()

				vae_loss += self.wml_alpha * wml_loss

			z_rand = torch.rand(size=(batch_size, self.latent_dim)).to(self.device) * self.z_width * 2.0 - self.z_width
			action_rand, _, _ = self.actor(state, z_rand.detach())
			_, _, _, z_rand_pred, _, _ = self.vae.encode(state, action_rand)

			if self.info_weight:
				info_loss = self.info_alpha * self.weighted_mse_loss(vae_weight.detach(), z_rand_pred, z_rand.detach())
			else:
				info_loss = self.info_alpha * F.mse_loss(z_rand_pred, z_rand.detach())

			vae_loss += info_loss

			# Optimize the info loss
			self.info_optimizer.zero_grad()
			vae_loss.backward()
			self.info_optimizer.step()
			if self.schedule is not None and self.schedule != 'none':
				self.info_scheduler.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return critic_loss.item()

	def weighted_mse_loss(self, weight, input, output):
		diff = (input - output) ** 2
		mean_diff = torch.mean(diff, dim=1, keepdim=True)
		return torch.mean(weight * mean_diff)

	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")

	def load(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor"))
