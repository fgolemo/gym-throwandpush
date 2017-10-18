import gym

env = gym.make("Pusher-v0")

env.reset()
env.render()

for i in range(100):
    action = env.action_space.sample()
    env.step(action)
    env.render()
