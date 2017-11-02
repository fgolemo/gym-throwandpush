import gym
import numpy as np
import time

from gym_throwandpush.envs.mujoco_pixel_wrapper import MujocoPusherPixelWrapper


def Pusher2PixelEnv(base_env_id):
    return MujocoPusherPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_throwandpush
    env = gym.make("Pusher2Pixel-v0")
    env.env.env._init(
        torques={
            "r_shoulder_pan_joint": 0.05,
            "r_shoulder_lift_joint": 500,
            "r_upper_arm_roll_joint": 0.05,
            "r_elbow_flex_joint": 500,
            "r_forearm_roll_joint": 0.05,
            "r_wrist_flex_joint": 500,
            "r_wrist_roll_joint": 0.05
        },
        #topDown=False,
        #colored=False
        topDown=True,
        colored=True
    )
    env.reset()

    for i in range(100):
        (obs, img), reward, finished, misc = env.step(env.action_space.sample())
        print (obs.shape)
        print (img.shape)

        #cv2.imshow('image', img[:,:,::-1])
        #cv2.waitKey(1)
        env.render()
        time.sleep(.1)
