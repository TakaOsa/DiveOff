from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.registration import register
from d4rl.gym_mujoco import gym_envs
from d4rl import infos


register(
    id='HopperLongHead-v0',
    max_episode_steps=1000,
    entry_point='adaptEnv.hopper_longhead:HopperLongHeadEnv'
)

register(
    id='HopperLowKnee-v0',
    max_episode_steps=1000,
    entry_point='adaptEnv.hopper_lowknee:HopperLowKneeEnv'
)

