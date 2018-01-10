import gym
import numpy as np
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
