import gym

env = gym.make("Pusher-v0")

env.reset()
env.render()

for i in range(100):
    action = env.action_space.sample()
    a,b,c,d = env.step(action)
    print (a) # obs
    print (len(a)) # 23
    print (b) # reward, scalar
    print (c) # is_done, bool
    print (d) # misc
    env.render()
    quit()
