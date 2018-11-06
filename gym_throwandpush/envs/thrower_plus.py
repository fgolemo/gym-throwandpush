import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

means = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/mean_obs.npy'),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/mean_s_transition_obs.npy'),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/mean_correction.npy'),
}

std = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/std_obs.npy'),
    'a': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/std_actions.npy'),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/std_s_transition_obs.npy'),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/thrower_15000/std_correction.npy'),
}


class ThrowerInferenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ThrowerInferenceWrapper, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        self.net.eval()
        print("DBG: MODEL LOADED:",modelPath)

    def _create_input(self, obs_pre_action, action, obs_sim):
        input_ = np.concatenate([
            (obs_pre_action[None,None,:] - means['o']) / std['o'],
            action[None,None,:] / std['a'],
            (obs_sim[None,None,:] - means['s']) / std['s']
        ], axis=2)
        return Variable(from_numpy(input_).float(), volatile=True)

    def _step(self, action):
        obs_pre_action = self.env.env._get_obs()
        obs_sim, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_pre_action, action, obs_sim)
        obs_real = self.net.forward(variable).data.cpu().numpy()[0,0,:]

        # Udpate the environnement with the new state
        self._set_to_simplus((obs_real * std['c']) + means['c'])
        return self.env.env._get_obs(), rew, done, info

    def _set_to_simplus(self, correction):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        qpos[:7] = qpos[:7] + correction[:7]
        qvel[:7] = qvel[7:14] + correction[7:14]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()

def ThrowerPlus(base_env_id):
    return ThrowerInferenceWrapper(gym.make(base_env_id))
