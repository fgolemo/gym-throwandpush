import gym
import numpy as np
import time
from gym import utils, error, spaces

from gym_throwandpush.envs import MujocoEnvPusher2
from gym_throwandpush.envs.mujoco_env_cheetah import MujocoEnvCheetah2

NOT_INITIALIZED_ERR = "Before doing a reset or your first " \
                      "step in the environment " \
                      "please call env._init()."

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class Cheetah2Env(MujocoEnvCheetah2, utils.EzPickle):
    isInitialized = False

    def _init(self, torques={}, colored=True):
        params = {
            "torques": torques,
        }
        self.isInitialized = True
        utils.EzPickle.__init__(self)
        xml = 'half_cheetah'
        if colored:
            xml += "-colored"

        MujocoEnvCheetah2.__init__(self, xml + '.xml', 5, params)

    def __init__(self):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }
        self.obs_dim = 18
        self.act_dim = 6

        self.action_space = spaces.Box(
            -np.ones(self.act_dim),
            np.ones(self.act_dim)
        )

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()


    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == '__main__':
    import gym_throwandpush

    env = gym.make("HalfCheetah-v1")

    print (env.action_space.high, env.action_space.low)
    print (env.observation_space)

    env = gym.make("HalfCheetah2-v0")

    print (env.action_space.high, env.action_space.low)
    env.env._init(
        torques={ # defaults
            "bthigh": 120,
            "bshin": 90,
            "bfoot": 60,
            "fthigh": 120,
            "fshin": 60,
            "ffoot": 30

            # "bthigh": 240,
            # "bshin": 45,
            # "bfoot": 120,
            # "fthigh": 60,
            # "fshin": 120,
            # "ffoot": 30

            # "bthigh": 1200,
            # "bshin": 9,
            # "bfoot": 600,
            # "fthigh": 12,
            # "fshin": 600,
            # "ffoot": 3
        },
        colored=True,
    )
    env.reset()


    def split_obs(obs):
        qpos = obs[3:9]  # robot has 6 DOF, so 6 angular positions
        qvel = obs[11:18]  # 7 angular velocities
        # these two above vectors are what's interesting for sim+

        return (qpos, qvel)


    for i in range(100):
        env.render()
        action = env.action_space.sample()
        print (action)
        # print(action.shape)
        obs, reward, done, misc = env.step(action)
        # print (obs.shape)
        # quit()

        obs_tup = split_obs(np.around(obs, 3))
        print(obs_tup)
        time.sleep(.03)
