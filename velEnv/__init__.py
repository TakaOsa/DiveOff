from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.registration import register
from d4rl.gym_mujoco import gym_envs
from d4rl import infos

register(
    id='Walker2dVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_vel:Walker2dVelEnv'
)

register(
    id='HopperVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_vel:HopperVelEnv'
)


register(
    id='HalfCheetahVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.half_cheetah_vel:HalfCheetahVelEnv'
)

register(
    id='AntVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.ant_vel:AntVelEnv'
)


WALKERVEL_RANDOM_SCORE = -4.02
WALKERVEL_EXPERT_SCORE = 2860

HOPPERVEL_RANDOM_SCORE = 6.84
HOPPERVEL_EXPERT_SCORE = 1962

HUMANOIDVEL_RANDOM_SCORE = 83.67
HUMANOIDVEL_EXPERT_SCORE = 5304

HALFCHEETAHVEL_RANDOM_SCORE = -324.79
HALFCHEETAHVEL_EXPERT_SCORE = 1869

ANTVEL_RANDOM_SCORE = -379.33
ANTVEL_EXPERT_SCORE = 2265

datatype = ['expert', 'medium', 'medium-replay', 'expert-medium']

for data_i in datatype:
    walker_idname = 'walker2dvel-diverse-' + data_i + '-v1'
    register(
        id=walker_idname,
        entry_point='velEnv.gym_vel_envs:get_walkervel_env',
        max_episode_steps=1000,
        kwargs={
            'ref_min_score': WALKERVEL_RANDOM_SCORE,
            'ref_max_score': WALKERVEL_EXPERT_SCORE
        }
    )

    hopper_idname = 'hoppervel-diverse-' + data_i + '-v1'
    register(
        id=hopper_idname,
        entry_point='velEnv.gym_vel_envs:get_hoppervel_env',
        max_episode_steps=1000,
        kwargs={
            'ref_min_score': HOPPERVEL_RANDOM_SCORE,
            'ref_max_score': HOPPERVEL_EXPERT_SCORE
        }
    )

    halfcheetah_idname = 'halfcheetahvel-diverse-' + data_i + '-v1'
    register(
        id=halfcheetah_idname,
        entry_point='velEnv.gym_vel_envs:get_halfcheetahvel_env',
        max_episode_steps=1000,
        kwargs={
            'ref_min_score': HALFCHEETAHVEL_RANDOM_SCORE,
            'ref_max_score': HALFCHEETAHVEL_EXPERT_SCORE
        }
    )

    ant_idname = 'antvel-diverse-' + data_i + '-v1'
    register(
        id=ant_idname,
        entry_point='velEnv.gym_vel_envs:get_antvel_env',
        max_episode_steps=1000,
        kwargs={
            'ref_min_score': ANTVEL_RANDOM_SCORE,
            'ref_max_score': ANTVEL_EXPERT_SCORE
        }
    )
