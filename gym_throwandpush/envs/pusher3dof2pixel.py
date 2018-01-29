import gym
import numpy as np
import time

from gym_throwandpush.envs.mujoco_pixel_wrapper import MujocoPusherPixelWrapper


def Pusher3Dof2PixelEnv(base_env_id):
    return MujocoPusherPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_throwandpush
    env = gym.make("Pusher3Dof2Pixel-v0")
    env.env.env._init(
        torques=[1, 10, .001],
        colored=True
    )
    env.reset()

    for i in range(100):
        (obs, img), reward, finished, misc = env.step(env.action_space.sample())
        print (obs.shape)
        print (img.shape)

        env.render()
        time.sleep(.1)
