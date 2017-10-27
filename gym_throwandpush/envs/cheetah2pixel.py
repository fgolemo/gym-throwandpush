import gym
import numpy as np
import time
from gym import spaces
from gym_reacher2.envs.reacher2pixel import MujocoPixelWrapper


def Cheetah2PixelEnv(base_env_id):
    return MujocoPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_throwandpush
    env = gym.make("HalfCheetah2Pixel-v0")
    env.env.env._init(
        torques={
            "bthigh": 120,
            "bshin": 90,
            "bfoot": 60,
            "fthigh": 120,
            "fshin": 60,
            "ffoot": 30
        },
        colored=False,
    )
    env.reset()

    for i in range(100):
        (obs, img), reward, finished, misc = env.step(env.action_space.sample())
        print (obs.shape)
        print (img.shape)

        #cv2.imshow('image', img[:,:,::-1])
        #cv2.waitKey(1)
        env.render()
        time.sleep(.03)
