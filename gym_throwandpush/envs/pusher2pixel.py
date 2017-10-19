import gym
import numpy as np
import time
from gym import spaces


class MujocoPusherPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MujocoPusherPixelWrapper, self).__init__(env)
        self.observation_space = [self.observation_space, spaces.Box(0, 255, [500, 500, 3])]

    def get_viewer(self):
        return self.env.unwrapped._get_viewer()

    def _observation(self, observation):
        super_obs = self.env.env._get_obs()
        self.get_viewer().render()
        data, width, height = self.get_viewer().get_image()
        return [super_obs, np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]]


def Pusher2PixelEnv(base_env_id):
    return MujocoPusherPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_throwandpush
    import cv2
    env = gym.make("Pusher2Pixel-v0")
    env.env.env._init(
        torques={
            "r_shoulder_pan_joint": 0.001,
            "r_shoulder_lift_joint": 1000,
            "r_upper_arm_roll_joint": 0.001,
            "r_elbow_flex_joint": 1000,
            "r_forearm_roll_joint": 0.001,
            "r_wrist_flex_joint": 1000,
            "r_wrist_roll_joint": 0.001
        },
        topDown=True,
        colored=True
    )
    env.reset()

    for i in range(100):
        (obs, img), reward, finished, misc = env.step(env.action_space.sample())
        print (obs.shape)
        print (img.shape)

        cv2.imshow('image', img[:,:,::-1])
        cv2.waitKey(1)
        env.render()
        time.sleep(.5)
