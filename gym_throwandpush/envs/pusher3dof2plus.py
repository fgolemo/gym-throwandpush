import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

# means = {
#     'o': np.array([-0.4417094, 1.50765455, -0.02639891, -0.05560728, 0.39159551, 0.03819341, 0.76052153, 0.23057458, 0.63315856, -0.6400153, 1.01691067, -1.02684915], dtype='float32'),
#     's': np.array([-0.44221497, 1.52240622, -0.02244471, 0.01573334, 0.23615479, 0.10089023, 0.7594685, 0.23817146, 0.63317519, -0.64011943, 1.01691067, -1.02684915], dtype='float32'),
#     'c': np.array([-2.23746197e-03, 4.93022148e-03, -2.03814497e-03, -6.97841570e-02, 1.53955221e-01, -6.21460043e-02], dtype='float32')
#     # 'r': np.array([2.25277853,  1.95338345, 1.64534044, 0.48487723, 0.45031613, 0.30320421], dtype='float32')
# }
#
# std = {
#     'o': np.array([0.38327965, 0.78956741, 0.48310387, 0.33454728, 0.53120506, 0.51319438, 0.20692779, 0.36664706, 0.25205335, 0.15865214, 0.11554158, 0.1132608], dtype='float32'),
#     's': np.array([0.38500383, 0.78036022, 0.48781601, 0.35502997, 0.60374367, 0.56180185, 0.21046612, 0.36828887, 0.25209084, 0.15857539, 0.11554158, 0.1132608], dtype='float32'),
#     'c': np.array([7.19802594e-03, 1.59114692e-02, 7.24539673e-03, 2.23035514e-01, 4.93483037e-01, 2.18238667e-01,], dtype='float32'),
#     'a': np.array([0.57690412, 0.57732242, 0.57705152], dtype='float32')
#     # 'r':  np.array([0.52004296, 0.51547343, 0.57784373, 1.30222356, 1.36113203, 2.38046765], dtype='float32')
# }
max_steps = 2000
means = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/mean_obs.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/mean_s_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/mean_correction_obs.npy'.format(max_steps))[:6],
}

std = {
    'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/std_obs.npy'.format(max_steps)),
    'a': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/std_actions.npy'.format(max_steps)),
    's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/std_s_transition_obs.npy'.format(max_steps)),
    'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/pusher_reb_{}/std_correction_obs.npy'.format(max_steps))[:6],
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
