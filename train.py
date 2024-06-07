import numpy as np
import torch
import torch.nn as nn

import gym
import argparse
import os
import d4rl
import d4rl.gym_mujoco

from d4rl.gym_mujoco.gym_envs import OfflineWalker2dEnv, OfflineHopperEnv

import h5py
from tqdm import tqdm

import utils
import DiveOff
import velEnv


def eval_policy(policy, env_name, seed, mean, std, a_mean, a_std, args, seed_offset=100, eval_episodes=10, render=False):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	episode_return_set = []
	episode_d4rl_set = []
	latent = None

	for _ in range(eval_episodes):

		if 'VAE' in args.policy or 'DiveOff' in args.policy :
			latent = latent_uniform_sampling(args.latent_cont_dim, args.latent_disc_dim, sample_num=1)

		episode_return = 0
		state, done = eval_env.reset(), False
		while not done:
			if render:
				eval_env.render()
			state = (np.array(state).reshape(1,-1) - mean)/std
			if 'VAE' in args.policy or 'DiveOff' in args.policy:
				action = policy.select_latent_action(state, latent)
			else:
				action = policy.select_action(state)
				action = (np.array(action).reshape(1, -1) * a_std + a_mean).flatten()
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			episode_return += reward

		episode_return_set.append(episode_return)
		episode_d4rl_set.append(eval_env.get_normalized_score(episode_return))

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward)

	print("----------------------------------------------")
	print('Evaluation over ', eval_episodes,' episodes d4rl_score: ',d4rl_score, 'return: ', avg_reward)
	print("----------------------------------------------")
	return d4rl_score, avg_reward, np.asarray(episode_d4rl_set), np.asarray(episode_return_set)


def latent_uniform_sampling(latent_cont_dim, latent_disc_dim, sample_num):
	latent_dim = latent_cont_dim + latent_disc_dim
	z = None
	z_cont = None
	if not latent_cont_dim == 0:
		z_cont = np.random.uniform(-1, 1, size=(sample_num, latent_cont_dim))
		if latent_disc_dim == 0:
			z = z_cont
	if not latent_disc_dim == 0:
		z_disc = np.random.randint(0, latent_disc_dim, sample_num)
		z_disc = to_one_hot(z_disc, latent_disc_dim)
		if latent_cont_dim == 0:
			z = z_disc
		else:
			z = np.hstack((z_cont, z_disc))
	return z


def to_one_hot(y, num_columns):
	"""Returns one-hot encoded Variable"""
	y_one_hot = np.zeros((y.shape[0], num_columns))
	y_one_hot[range(y.shape[0]), y] = 1.0
	return y_one_hot


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="DiveOff")               # Policy name,walker2d-expert-v2
	parser.add_argument("--env", default="walker2dvel-diverse-expert-medium-v1")        # walker2dvel-diverse-expert-v0, OpenAI gym environment name kitchen-complete-v0,antmaze-large-play-v0, hammer-human-v0,hopper-medium-v0 walker2d-medium-expert-v0
	parser.add_argument("--seed", default=10, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=2e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)  # Discount factor
	parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
	parser.add_argument("--normalize", default=True)
	parser.add_argument("--action_normalize", default=False)

	parser.add_argument("--latent_cont_dim", default=2, type=int)
	parser.add_argument("--latent_disc_dim", default=0, type=int)

	parser.add_argument("--scale", default=3.0, type=float)
	parser.add_argument("--score_type", default='adv')
	parser.add_argument("--weight_type", default='clamp')

	parser.add_argument("--schedule", default='cosine')  # schedule None or 'cosine'
	parser.add_argument("--info_weight", default=False)
	parser.add_argument("--scale_schedule", default=True)
	parser.add_argument("--weighted_q", default=True)
	parser.add_argument("--v_scale", default=1.0, type=float)
	parser.add_argument("--vae_steps", default=1e4, type=int)
	parser.add_argument("--info_lr_rate", default=0.3, type=int)
	parser.add_argument("--z_width", default=2.0, type=float)
	parser.add_argument("--info_alpha", default=2.0, type=float)

	parser.add_argument("--hidden", default=(256, 256))
	parser.add_argument("--device_id", default=-1, type=int)
	args = parser.parse_args()

	schedule = ''
	if args.schedule is not None:
		schedule = '_' + args.schedule
	info_weight = '_info_weight' if args.info_weight else ''
	scale_schedule = '_scale_schedule' if args.scale_schedule else ''
	weighted_q = '_weighted_q' if args.weighted_q else ''
	v_scale = '_vscale' + str(args.v_scale) if args.v_scale != 3.0 else ''

	if args.policy == 'DiveOff':
		file_name = args.policy + '_info_lr_rate' + str(args.info_lr_rate) + '_inf_alp' + str(args.info_alpha) + '_ldim' + str(args.latent_cont_dim)\
					+ '_scale' + str(args.scale) + '_' + args.weight_type + schedule + weighted_q + v_scale + info_weight + scale_schedule + '_' + args.env + '_' + str(args.seed)

	print("---------------------------------------")
	print("Policy:", args.policy, "Env: ",args.env," Seed: ", args.seed)
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if not os.path.exists("./results/"+args.policy):
		os.makedirs("./results/"+args.policy)

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy
	if args.policy == 'DiveOff':
		policy = DiveOff.DiveOff(state_dim=state_dim, action_dim=action_dim,
							 max_action=max_action, latent_cont_dim=args.latent_cont_dim, discount=args.discount,
							 policy_freq=args.policy_freq, weight_type=args.weight_type,
							 scale=args.scale, hidden=args.hidden, info_lr_rate=args.info_lr_rate,
							 vae_steps=args.vae_steps, info_alpha=args.info_alpha, 
							 info_weight=args.info_weight, scale_schedule=args.scale_schedule, schedule=args.schedule,
							 Tmax=args.max_timesteps, T_scale_max=args.max_timesteps * 0.5, weighted_q=args.weighted_q,
							 v_scale=args.v_scale)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load('./models/'+ policy_file)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)


	if 'diverse' in args.env:
		replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env, dataset=env.get_dataset(h5path='./dataset/'+ args.env +'.hdf5')))
	else:
		replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	if args.action_normalize:
		a_mean, a_std = replay_buffer.normalize_action()
	else:
		a_mean, a_std = 0, 1


	print('data size', replay_buffer.size)
	
	evaluations_d4rl= []
	evaluations_return = []
	critic_log = []
	critic_loss_log_i = 0
	model_loss = []
	model_loss_log_i = 0

	if 'VAE' in args.policy or 'DiveOff' in args.policy:
		for t in range(int(args.vae_steps)):
			policy.train_vae(replay_buffer, args.batch_size)

	for t in range(int(args.max_timesteps)):
		critic_loss_t = policy.train(replay_buffer, args.batch_size)

		critic_loss_log_i = critic_loss_log_i + critic_loss_t

		# Evaluate episode
		if (t+1) % args.eval_freq == 0:
			print('Time steps: ', t+1)
			if 'antmaze' in args.env and t+1 == args.max_timesteps:
				d4rl_score, ave_return, d4l_score_epi, return_epi = eval_policy(policy, args.env, args.seed, mean, std, a_mean, a_std, args, eval_episodes=100)
			else:
				d4rl_score, ave_return, d4l_score_epi, return_epi = eval_policy(policy, args.env, args.seed, mean, std, a_mean, a_std, args)
			evaluations_d4rl.append(d4rl_score)
			evaluations_return.append(ave_return)
			critic_log.append(critic_loss_log_i/args.eval_freq)
			critic_loss_log_i = 0

			np.savetxt('./results/'+ args.policy + '/'+ file_name + '_d4rl.txt', evaluations_d4rl)
			np.savetxt('./results/'+ args.policy + '/'+ file_name + '_return.txt', evaluations_return)
			np.savetxt('./results/'+ args.policy + '/'+ file_name + '_critic_log.txt', critic_log)

			if args.save_model:
				try:
					import pathlib
					pathlib.Path("./models/" + args.policy).mkdir(parents=True, exist_ok=True)

				except:
					print("A result directory does not exist and cannot be created. The trial results are not saved")

				policy.save("./models/" + args.policy + "/" + file_name)
