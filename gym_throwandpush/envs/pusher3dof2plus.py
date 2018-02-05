import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable


class Pusher3DofInferenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(Pusher3DofInferenceWrapper, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        print("DBG: MODEL LOADED:",modelPath)

    def _step(self, action):
        obs_pre_action = self.env.env._get_obs()
        obs_sim, rew, done, info = self.env.step(action)
        input_ = np.concatenate([obs_pre_action[None,None,:6], action[None,None,:], obs_sim[None,None,:6]], axis=2)
        variable = Variable(from_numpy(input_).float(), volatile=True)
        obs_real = self.net.forward(variable)

        # Udpate the environnement with the new state
        self._set_to_simplus(obs_real.data.cpu().numpy()[0,0,:])
        return self.env.env._get_obs(), rew, done, info

    def _set_to_simplus(self, correction):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        qpos[:3] = qpos[:3] + correction[:3]
        qvel[:3] = qvel[:3] + correction[3:6]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()

def Pusher3Dof2Plus(base_env_id):
    return Pusher3DofInferenceWrapper(gym.make(base_env_id))