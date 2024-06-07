import numpy as np
import torch

import gym
import argparse
import os
import d4rl
import d4rl.gym_mujoco

from d4rl.gym_mujoco.gym_envs import OfflineWalker2dEnv

import h5py
from tqdm import tqdm
from PIL import Image

import utils
import DiveOff

import velEnv


def evaluate_greedy(env_test, policy, mean, std, args, latent_cont_dim, latent_disc_dim, video_folder, test_num = 5, zmax=0.9 ):

    latent_dim = latent_cont_dim + latent_disc_dim
    cont_traversal = np.linspace(-zmax, zmax, test_num)

    st_frame = 400
    end_frame = 800
    frame_freq = 20

    return_set = []
    for i in range(test_num):
        z = np.zeros((1, latent_dim))

        z[0, 0] = cont_traversal[i]

        for j in range(test_num):
            state, done = env_test.reset(), False
            return_epi_test = 0
            z[0, 1] = cont_traversal[j]

            if args.save_screenshot:
                im_folder = video_folder + '/z_' + str(np.around(z, decimals=2))
                try:
                    import pathlib
                    pathlib.Path(im_folder).mkdir(parents=True, exist_ok=True)

                except:
                    print("A result directory does not exist and cannot be created. The trial results are not saved")

            for t_test in range(int(args.max_episode_len)):

                state = (np.array(state).reshape(1, -1) - mean) / std

                if args.save_screenshot:
                    if t_test % frame_freq == 0 and t_test >= st_frame and t_test <= end_frame:
                        screen = env_test.render(mode='rgb_array')
                        im = Image.fromarray(screen.astype(np.uint8))
                        fname = video_folder + '/z_' + str(np.around(z, decimals=2)) + '/z_' + str(np.around(z, decimals=2)) + 't_' + str(t_test) + '.jpg'
                        im.save(fname)

                action = policy.select_latent_action(state, np.reshape(z, (1, latent_dim)))
                state2, reward_test, terminal_test, info_test = env_test.step(action)
                terminal_bool = float(terminal_test) if t_test < int(args.max_episode_len) else 0

                state = state2
                return_epi_test = return_epi_test + reward_test
                if terminal_bool:
                    break

            print('epi_len', t_test)
            print('z', np.around(z, decimals=2), end=' ')
            print('test {:d}, return: {:d}'.format(int(i), int(return_epi_test)))
            return_set.append(return_epi_test)

    env_test.close()
    fname = video_folder + '/test_return.txt'
    np.savetxt(fname, np.asarray(return_set))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="DiveOff")  # Policy name,walker2d-expert-v2
    parser.add_argument("--env",
                        default="walker2dvel-diverse-expert-medium-v1")
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", default=True, type=bool)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--normalize", default=True)

    parser.add_argument("--latent_cont_dim", default=2, type=int)
    parser.add_argument("--latent_disc_dim", default=0, type=int)

    parser.add_argument("--scale", default=3.0, type=float)
    parser.add_argument("--score_type", default='adv')
    parser.add_argument("--weight_type", default='clamp')

    parser.add_argument("--schedule", default='cosine')  # schedule None or 'cosine'
    parser.add_argument("--info_weight", default=False)
    parser.add_argument("--scale_schedule", default=True)  # action='store_true'
    parser.add_argument("--weighted_q", default=True)
    parser.add_argument("--v_scale", default=1.0, type=float)
    parser.add_argument("--vae_steps", default=1e5, type=int)
    parser.add_argument("--info_lr_rate", default=0.3, type=int)
    parser.add_argument("--z_width", default=2.0, type=float)
    parser.add_argument("--info_alpha", default=2.0, type=float)

    parser.add_argument("--hidden", default=(256, 256))
    parser.add_argument("--device_id", default=-1, type=int)

    parser.add_argument("--video_folder", default='./video/offline/')
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--save-screenshot", default=True, type=bool)
    parser.add_argument("--zmax", default=1.8, type=float)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)

    args = parser.parse_args()

    schedule = ''
    if args.schedule is not None:
        schedule = '_' + args.schedule

    info_weight = '_info_weight' if args.info_weight else ''
    scale_schedule = '_scale_schedule' if args.scale_schedule else ''
    weighted_q = '_weighted_q' if args.weighted_q else ''
    v_scale = '_vscale' + str(args.v_scale) if args.v_scale != 3.0 else ''

    if args.policy == 'DiveOff':
        file_name = args.policy + '_info_lr_rate' + str(args.info_lr_rate) + '_inf_alp' + str(
            args.info_alpha) + '_ldim' + str(args.latent_cont_dim) \
                    + '_scale' + str(
            args.scale) + '_' + args.weight_type + schedule + weighted_q + v_scale + info_weight + scale_schedule + '_' + args.env + '_' + str(
            args.seed)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/" + args.policy):
        os.makedirs("./results/" + args.policy)

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if 'walker2dvel' in args.env:
        env = gym.make('Walker2dVel-v0')
    elif 'hoppervel' in args.env:
        env = gym.make('HopperVel-v0')
    elif 'halfcheetahvel' in args.env:
        env = gym.make('HalfCheetahVel-v0')
    elif 'antvel' in args.env:
        env = gym.make('AntVel-v0')
    else:
        env = None
        print('env not implemented.')

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    env_off = gym.make(args.env)

    # Initialize policy
    if args.policy == 'DiveOff':
        policy = DiveOff.DiveOff(state_dim=state_dim, action_dim=action_dim,
                                 max_action=max_action, latent_cont_dim=args.latent_cont_dim, discount=args.discount,
                                 policy_freq=args.policy_freq, weight_type=args.weight_type,
                                 scale=args.scale, hidden=args.hidden, info_lr_rate=args.info_lr_rate,
                                 vae_steps=args.vae_steps, info_alpha=args.info_alpha,
                                 info_weight=args.info_weight, scale_schedule=args.scale_schedule,
                                 schedule=args.schedule,
                                 Tmax=args.max_timesteps, T_scale_max=args.max_timesteps * 0.5,
                                 weighted_q=args.weighted_q,
                                 v_scale=args.v_scale  )

    # if args.load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    video_folder = args.video_folder + '/' + args.policy + '/' + file_name + '_' + str(args.zmax)
    latent_cont_dim = args.latent_cont_dim
    latent_disc_dim = args.latent_disc_dim

    try:
        import pathlib
        pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)

    except:
        print("A result directory does not exist and cannot be created. The trial results are not saved")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    if 'diverse' in args.env:
        replay_buffer.convert_D4RL(
            d4rl.qlearning_dataset(env, dataset=env_off.get_dataset(h5path='./dataset/' + args.env + '.hdf5')))
    else:
        replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env_off))

    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    print('data size', replay_buffer.size)
    print('state_dim', state_dim)

    if args.save_video:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger= lambda episode_id: episode_id%1==0)

    policy.load(f"./models/{args.policy}/{file_name}")

    evaluate_greedy(env, policy, mean, std, args, latent_cont_dim, latent_disc_dim, video_folder=video_folder, zmax=args.zmax)
