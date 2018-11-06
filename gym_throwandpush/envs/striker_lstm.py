import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

max_steps = 25000
means = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_obs.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_s_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_correction_obs.npy'.format(max_steps)),
}

std = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_obs.npy'.format(max_steps)),
    'a': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_actions.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_s_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_correction_obs.npy'.format(max_steps)),
}
std['c'][19] = 1.0


class StrikerInferenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StrikerInferenceWrapper, self).__init__(env)
        self.env = env
        self.current_obs = None

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        print("DBG: MODEL LOADED:",modelPath)

    def _create_input(self, obs_pre_action, action):
        input_ = np.concatenate([
            (obs_pre_action[None,None,:] - means['o']) / std['o'],
            action[None,None,:] / std['a'],
        ], axis=2)
        return Variable(from_numpy(input_).float(), volatile=True)

    def _step(self, action):
        obs_pre_action = self.current_obs
        _, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_pre_action, action)
        correction_scaled = self.net.forward(variable).data.cpu().numpy()[0,0,:]

        # Udpate the environnement with the new state
        correction = np.append((correction_scaled * std['c']) + means['c'], [0])
        self.current_obs = self.current_obs + correction
        self._set_to_simplus(self.current_obs)
        return correction, rew, done, info

    def _set_to_simplus(self, obs):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        qpos[:7] = obs[:7]
        qvel[:7] = obs[7:14]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        self.current_obs = self.env.reset()
        return self.current_obs

def StrikerLstm(base_env_id):
    return StrikerInferenceWrapper(gym.make(base_env_id))
