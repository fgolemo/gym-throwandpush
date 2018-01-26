import gym
import gym_throwandpush

env = gym.make("Pusher3Dof2-v0")
env.env._init(
    torques=[1.,1.,1.]
)

env.reset()
env.render()

for i in range(1000):
    action = env.action_space.sample()
    a,b,c,d = env.step(action)
    env.render()
    # quit()
