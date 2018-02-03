import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable


class Pusher3DofInferenceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Pusher3DofInferenceWrapper, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.cuda()
        print("DBG: MODEL LOADED:",modelPath)

    def _observation(self, observation):
        obs = self.env.env._get_obs()
        variable = Variable(from_numpy(obs[None,None,:-4]).cuda().float(), volatile=True, requires_grad=False)
        obs_plus = self.net.forward(variable)

        self._set_to_simplus(obs, obs_plus.data.cpu().numpy()[0,0,:])
        return self.env.env._get_obs()

    def _set_to_simplus(self, origin, correction):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        qpos[:3] = qpos[:3] + correction[:3]
        qvel[:3] = qvel[:3] + correction[3:6]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        # print("DBG: HIDDEN STATE RESET AND DETACHED")
        return super(Pusher3DofInferenceWrapper, self)._reset()

def Pusher3Dof2Plus(base_env_id):
    return Pusher3DofInferenceWrapper(gym.make(base_env_id))