from d4rl import offline_env
# from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
from d4rl.utils.wrappers import NormalizedBoxEnv
from gym.envs.mujoco.humanoid import HumanoidEnv

from .walker2d_vel import Walker2dVelEnv
from .hopper_vel import HopperVelEnv
from .half_cheetah_vel import HalfCheetahVelEnv
from .ant_vel import AntVelEnv


class OfflineHopperVelEnv(HopperVelEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperVelEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineWalker2dVelEnv(Walker2dVelEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dVelEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineHalfCheetahVelEnv(HalfCheetahVelEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahVelEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineAntVelEnv(AntVelEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntVelEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_hoppervel_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperVelEnv(**kwargs))


def get_walkervel_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dVelEnv(**kwargs))

def get_halfcheetahvel_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahVelEnv(**kwargs))

def get_antvel_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntVelEnv(**kwargs))


if __name__ == '__main__':
    """Example usage of these envs"""
    pass
