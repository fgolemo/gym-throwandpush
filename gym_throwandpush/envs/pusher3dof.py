import numpy as np
import time
from gym import utils
from gym.envs.mujoco import mujoco_env
import scipy.misc
import os


class PusherEnv3Dof(mujoco_env.MujocoEnv, utils.EzPickle):
    itr = 0

    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = '3link_gripper_push_2d.xml'
        full_model_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
        mujoco_env.MujocoEnv.__init__(self, full_model_path, 5)
        self.itr = 0

    def _step(self, a):
        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        a *= 3  # important - in the original env the range of actions is doubled

        # reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = reward_dist + 0.1 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        img = self.render('rgb_array')
        # idims = self._kwargs['imsize']
        img = scipy.misc.imresize(img, (128,128,3))

        self.itr += 1
        return ob, reward, done, dict(img=img)

    def viewer_setup(self):
        coords = [.7,-.5,0]
        for i in range(3):
            self.viewer.cam.lookat[i] = coords[i]
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 2

    def reset_model(self):

        self.itr = 0
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

        # if hasattr(self, "_kwargs") and 'geoms' in self._kwargs:
        #     geoms = self._kwargs['geoms']
        #     ct = 0
        #     for body in range(len(geompostemp)):
        #         if 'object' in str(self.model.geom_names[body]):
        #             geompostemp[body, 0] = geoms[ct][1]
        #             geompostemp[body, 1] = geoms[ct][2]
        #             ct += 1

        self.model.geom_pos = geompostemp

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        print (qpos, qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

from tkinter import *

class PusherCtrlGui:
    def __init__(self, env):
        self.env = env
        self.root = Tk()
        self.slider1 = Scale(self.root, variable=0.1, from_=-3, to=3, resolution=.1,
                               orient="horizontal")
        self.slider2 = Scale(self.root, variable=0, from_=-3, to=3, resolution=.1,
                               orient="horizontal")
        self.slider3 = Scale(self.root, variable=-0.1, from_=-3, to=3, resolution=.1,
                               orient="horizontal")



        b = Button(self.root, text="STEP", command=self.button_step)
        b.pack()

        b = Button(self.root, text="RESET", command=self.button_reset)
        b.pack()

        self.slider1.pack()
        self.slider2.pack()
        self.slider3.pack()

        self.root.bind("<space>", self.key_step)

        self.root.mainloop()

    def key_step(self, event):
        self.button_step()

    def button_step(self):
        action = [self.slider1.get(), self.slider2.get(), self.slider3.get()]
        # print (action)
        obs, rew, done, misc = self.env.step(action)
        print (np.around(obs,2), rew, misc["img"].shape)
        self.env.render()

    def button_reset(self):
        self.slider1.set(0.0)
        self.slider2.set(0.0)
        self.slider3.set(0.0)
        self.env.reset()
        self.env.render()

    # def updateValue(self, event):
    #     print("new value:", self.slider.get())





if __name__ == '__main__':


    import gym
    import gym_throwandpush

    env = gym.make("Pusher3Dof-v0")
    env.reset()
    env.render()

    app = PusherCtrlGui(env)


    #
    # for i in range(100):
    #     env.render()
    #     action = env.action_space.sample()
    #     print(action)
    #     obs, reward, done, misc = env.step(action)
    #     print(obs, reward, done, misc)
    #     # time.sleep(1)
