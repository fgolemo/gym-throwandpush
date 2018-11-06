import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

max_steps = 2000
means = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/mean_obs.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/mean_s_transition_obs.npy'.format(max_steps)),
    'r': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/mean_r_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/mean_correction.npy'.format(max_steps)),
}

std = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/std_obs.npy'.format(max_steps)),
    'a': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/std_actions.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/std_s_transition_obs.npy'.format(max_steps)),
    'r': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/std_r_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_{}/std_correction.npy'.format(max_steps)),
}

class Pusher3DofInferenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(Pusher3DofInferenceWrapper, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        self.net.eval()
        print("DBG: MODEL LOADED:",modelPath)

    def _create_input(self, obs, transition):
        input_ = np.concatenate([
            (obs[None,None,:] - means['o']) / std['o'],
            (transition[None,None,:] - means['r']) / std['r'],
        ], axis=2)
        return Variable(from_numpy(input_).float(), volatile=True)

    def _step(self, action):
        obs_pre_action = self.env.env._get_obs()
        obs_sim, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_pre_action, obs_sim)
        action_real = self.net.forward(variable).data.cpu().numpy()[0,0,:]

        self._set_to_simplus(obs_pre_action)
        return self.env.step((action + action_real * std['a']).clip(min=-1., max=1.))

    def _set_to_simplus(self, obs):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        qpos[:3] = obs[:3]
        qvel[:3] = obs[3:6]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()

def Pusher3Dof2Inverse(base_env_id):
    return Pusher3DofInferenceWrapper(gym.make(base_env_id))
