import torch
import gym
import numpy as np
from gym import spaces


relevant_items = range(14)

def obs_to_net(obs):
    obs_v = torch.autograd.Variable(torch.from_numpy(obs[relevant_items].astype(np.float32), ), requires_grad=False)
    return obs_v

def net_to_obs(net, obs):
    for i, idx in enumerate(relevant_items):
        obs[idx] = net[0,i].data.numpy()[0]
    return obs

class Pusher2InferenceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Pusher2InferenceWrapper, self).__init__(env)
        env.load_model = self.load_model
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        print("DBG: MODEL LOADED:",modelPath)

    def _observation(self, observation):
        super_obs = self.env.env._get_obs()
        # print(super_obs)
        obs_v = obs_to_net(super_obs)
        # print(obs_v)
        obs_plus = self.net.forward(obs_v)
        # print(obs_plus)

        self._set_to_simplus(obs_plus.cpu().data.numpy()[0])

        obs_plus_full = net_to_obs(obs_plus, super_obs)
        # print(obs_plus_full)

        return obs_plus_full

    def _set_to_simplus(self, obs_plus):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()
        # print (qpos[:2], np.sin(qpos[:2]), obs_plus[:2], np.arcsin(obs_plus[:2]))
        qpos[:7] = obs_plus[:7]
        qvel[:7] = obs_plus[7:]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        # print("DBG: HIDDEN STATE RESET AND DETACHED")
        return super(Pusher2InferenceWrapper, self)._reset()

def Pusher2PlusEnv(base_env_id):
    return Pusher2InferenceWrapper(gym.make(base_env_id))

