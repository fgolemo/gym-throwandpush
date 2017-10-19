import gym
import numpy as np
import time
from gym import utils, error, spaces

from gym_throwandpush.envs import MujocoEnvPusher2

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


class Pusher2Env(MujocoEnvPusher2, utils.EzPickle):
    isInitialized = False

    def _init(self, torques={}, topDown=False):
        # if colors is None:
        #     # color values are "R G B", for red, green, and blue respectively
        #     # in the range from 0 to 1. For example white it "1 1 1". Red is
        #     # "1 0 0".
        #
        #     colors = {
        #         "arenaBackground": ".9 .9 .9",
        #         "arenaBorders": "0.9 0.4 0.6",
        #         "arm0": "0.0 0.4 0.6",
        #         "arm1": "0.0 0.4 0.6"
        #     }
        params = {
            "torques": torques,
            # "fov": fov,
            # "colors": colors
        }
        if topDown:
            self.viewer_setup = self.top_down_cam

        self.isInitialized = True
        utils.EzPickle.__init__(self)
        MujocoEnvPusher2.__init__(self, 'pusher.xml', 5, params)

    def __init__(self):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }
        self.obs_dim = 23

        self.action_space = spaces.Box(
            np.array([-1., -1.]),
            np.array([1., 1.])
        )

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _step(self, a):
        if not self.isInitialized:
            raise Exception(NOT_INITIALIZED_ERR)

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def top_down_cam(self):
        self.viewer.cam.trackbodyid = -1  # id of the body to track
        self.viewer.cam.distance = self.model.stat.extent * 1.2  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # qpos (shape [4,1]) holds the angles of the 2 joints
        # it has the form: [
        # x angle joint 1 in rad,
        # x angle joint 2 in rad,
        # y angle joint 1 (constant)
        # y angle joint 2 (constant)
        # ]
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            # np.cos(theta),
            # np.sin(theta), # this is the most meaningful data here,
            # # containing the sine of the two joint angles
            # self.model.data.qpos.flat[2:],
            # self.model.data.qvel.flat[:2], # angular momentum of the two joints
            # self.get_body_com("fingertip") - self.get_body_com("target")
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])


if __name__ == '__main__':
    import gym_throwandpush
    env = gym.make("Pusher2-v0")
    env.env._init(
        torques={
            "r_shoulder_pan_joint": 0.001,
            "r_shoulder_lift_joint": 1000,
            "r_upper_arm_roll_joint": 0.001,
            "r_elbow_flex_joint": 1000,
            "r_forearm_roll_joint": 0.001,
            "r_wrist_flex_joint": 1000,
            "r_wrist_roll_joint": 0.001
        },
        topDown=True
    )
    env.reset()

    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
        # time.sleep(.5)
