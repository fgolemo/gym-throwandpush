#import torch
import gym
import numpy as np
import time

import scipy
from gym import utils, error, spaces

from gym_throwandpush.envs.mujoco_env_pusher3dof import MujocoEnvPusher3Dof2

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


class Pusher3Dof2Env(MujocoEnvPusher3Dof2, utils.EzPickle):
    isInitialized = False

    def _init(self, torques=[], distal_1=0.5, distal_2=0.4, topDown=False, colored=True):
        params = {
            "torques": torques,
            "distal_1": distal_1,
            "distal_2": distal_2
        }
        if topDown:
            self.viewer_setup = self.top_down_cam

        self.isInitialized = True
        utils.EzPickle.__init__(self)
        xml = '3link_gripper_push_2d'
        if colored:
            xml += "-colored"

        MujocoEnvPusher3Dof2.__init__(self, xml + '.xml', 5, params)

    def __init__(self):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }
        self.obs_dim = 12

        self.action_space = spaces.Box(
            -np.ones(3),
            np.ones(3)
        )

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _step(self, a):
        if not self.isInitialized:
            raise Exception(NOT_INITIALIZED_ERR)

        vec_1 = self.get_body_com("object") - self.get_body_com("distal_4")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        a *= 3  # important - in the original env the range of actions is tripled

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        # reward = reward_dist + 0.1 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        img = None
        # img = self.render('rgb_array')
        # img = scipy.misc.imresize(img, (128, 128, 3))

        return ob, reward, done, dict(img=img)

    def viewer_setup(self):
        coords = [.7, -.5, 0]
        for i in range(3):
            self.viewer.cam.lookat[i] = coords[i]
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 2

    def reset_model(self):

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            object_ = [np.random.uniform(low=-1.0, high=-0.4),
                       np.random.uniform(low=0.3, high=1.2)]
            goal = [np.random.uniform(low=-1.2, high=-0.8),
                    np.random.uniform(low=0.8, high=1.2)]
            if np.linalg.norm(np.array(object_) - np.array(goal)) > 0.45: break
        self.object = np.array(object_)
        self.goal = np.array(goal)
        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.object = np.array(self._kwargs['object'])
            self.goal = np.array(self._kwargs['goal'])

        geompostemp = np.copy(self.model.geom_pos)
        for body in range(len(geompostemp)):
            if 'object' in str(self.model.geom_names[body]):
                pos_x = np.random.uniform(low=-0.9, high=0.9)
                pos_y = np.random.uniform(low=0, high=1.0)
                geompostemp[body, 0] = pos_x
                geompostemp[body, 1] = pos_y

        self.model.geom_pos = geompostemp

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            self.get_body_com("distal_4")[:2],
            self.get_body_com("object")[:2],
            self.get_body_com("goal")[:2],
        ])


if __name__ == '__main__':
    import gym_throwandpush

    env = gym.make("Pusher3Dof2-v0")
    env._init(
        torques=[1, 10, .001],
        colored=True
    )
    env.reset()


    # def split_obs(obs):
    #     qpos = obs[:7]  # robot has 7 DOF, so 7 angular positions
    #     qvel = obs[7:14]  # 7 angular velocities
    #     # these two above vectors are what's interesting for sim+
    #
    #     tip_pos = obs[14:17]  # 1 tip position in 3D space
    #     obj_pos = obs[17:20]  # 1 object position in 3D space
    #     gol_pos = obs[20:23]  # 1 goal position in 3D space
    #     return (qpos, qvel, tip_pos, obj_pos, gol_pos)


    for i in range(100):
        env.render()
        action = env.action_space.sample()
        print(action)
        obs, reward, done, misc = env.step(action)
        # obs_tup = split_obs(np.around(obs, 3))
        print (np.around(obs, 2))
        # print(obs_tup)
        # time.sleep(1)
