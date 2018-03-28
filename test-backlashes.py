import time
import gym
import gym_throwandpush


def run_env(dmax=200, width=20):
    if dmax == 0 or width == 0:
        width = 0
        dmax = 0

    env = gym.make("Pusher3Dof2Pixel-v0")
    env.env.env._init(
        xml='3link_gripper_push_2d_backlash_{}_{}'.format(dmax, width),
        colored=False
    )
    env.reset()

    for torque in [-0.5, -1]:
        print("Torque:",torque)
        for i in range(20):
            (obs, img), reward, finished, misc = env.step([0, torque, 0])
            # print(obs.shape)
            # print(img.shape)
            #
            env.render()
            time.sleep(.1)
            # print(env.action_space.sample())

    print("========")


for dmax in [0, 50, 100, 200, 400]:
    run_env(dmax=dmax)

for width in [5, 10, 20, 40]:
    run_env(width=width)
