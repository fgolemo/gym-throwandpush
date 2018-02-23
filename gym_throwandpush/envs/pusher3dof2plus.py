import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

means = {
    'o': np.array([2.2281456, 1.93128324, 1.63007331, 0.48472479, 0.4500702, 0.30325469], dtype='float32'),
    's': np.array([2.25090551, 1.94997263, 1.6495719, 0.43379614, 0.3314755, 0.43763939], dtype='float32'),
    'c': np.array([0.00173789, 0.00352129, -0.00427585, 0.05105286, 0.11881274, -0.13443381], dtype='float32')
    # 'r': np.array([2.25277853,  1.95338345, 1.64534044, 0.48487723, 0.45031613, 0.30320421], dtype='float32')
}

std = {
    'o': np.array([0.56555426, 0.5502255 , 0.59792095, 1.30218685, 1.36075258, 2.37941241], dtype='float32'),
    's': np.array([0.5295766, 0.51998389, 0.57609886, 1.35480666, 1.40806067, 2.43865967], dtype='float32'),
    'c': np. array([0.01608515, 0.0170644, 0.01075647, 0.46635619, 0.53578401, 0.32062387], dtype='float32')
    # 'r':  np.array([0.52004296, 0.51547343, 0.57784373, 1.30222356, 1.36113203, 2.38046765], dtype='float32')
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
        print("DBG: MODEL LOADED:",modelPath)

    def _create_input(self, obs_pre_action, action, obs_sim):
        input_ = np.concatenate([
            (obs_pre_action[None,None,:6] - means['o']) / std['o'],
            action[None,None,:],
            (obs_sim[None,None,:6] - means['s']) / std['s']
        ], axis=2)
        return Variable(from_numpy(input_).float(), volatile=True)

    def _step(self, action):
        obs_pre_action = self.env.env._get_obs()
        obs_sim, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_pre_action, action, obs_sim)
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
