import gym
import gym_throwandpush

env = gym.make("Striker3Dof-v0")
# env.env._init(
#     torques=[1.,1.,1.],
#     proximal_1=0.41,
#     distal_1=0.38,
#     distal_2=0.41,
# )

env.reset()
env.render()

for i in range(1000):
    action = env.action_space.sample()
    a,b,c,d = env.step(action)
    env.render()
    # quit()
