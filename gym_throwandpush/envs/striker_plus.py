import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

means = {
    'o': np.array([1.12845600e+00, 2.85855811e-02, 6.05528504e-02, -9.97073829e-01, 6.56818366e-03, 1.40352142e+00,3.84055846e-03, 1.02492295e-01,
        7.11518712e-03, 4.65221144e-02, -2.46240869e-01, 1.76974421e-03, 3.31832051e-01, -6.37228310e-04, 2.20609486e-01, -2.38019347e-01,
        8.07869807e-02, 4.95883346e-01, -1.74267560e-01, -2.71298617e-01, 4.24245656e-01, 5.50004303e-01, -3.22469383e-01], dtype='float32'),
    's': np.array([9.59522069e-01, 2.64616702e-02, -1.47129979e-03, -1.00690019e+00, 8.95156711e-03, -2.26142392e-01, 4.37198719e-03, -3.18137455e+00,
        -8.15565884e-02, -1.13330793e+00, -1.15818921e-02, 3.88979465e-02,-3.10147667e+01, 1.01360520e-02, 2.72106588e-01, -1.99255228e-01, 8.47409219e-02, 4.95847106e-01,
        -1.74359173e-01, -2.71298617e-01, 4.24228728e-01, 5.50020635e-01, -3.22452545e-01], dtype='float32'),
    'c': np.array([1.74270540e-01, 2.45837355e-03, 6.43057898e-02, -2.44984333e-03, -2.27138959e-03, 1.64594924e+00, -5.81401051e-04, 3.28941011e+00, 8.86482969e-02, 1.17964220e+00, -2.34902933e-01, -3.72229367e-02, 3.13308201e+01, -1.16425110e-02], dtype='float32'),
}

std = {
    'o': np.array([0.81256843, 0.25599328, 1.04780889, 0.78443158, 0.98964226, 0.66285115, 0.99393243, 1.09946322, 0.49455127, 2.67891765,
         2.24620128, 3.26377749, 2.13994312, 3.29273677, 4.68308300e-01, 2.50030875e-01, 1.48312315e-01, 5.18861376e-02, 5.50917946e-02,
         1.29705272e-03, 1.62378117e-01, 2.58738667e-01, 2.16630325e-02], dtype='float32'),
    's': np.array([0.57472217, 0.24750113, 1.07578814, 0.78113234, 0.99487597, 0.11505181, 1.00299263, 6.00020313,
        1.16489089, 4.44655132, 4.39647961, 3.27095556, 14.71065998, 3.32328629, 3.99973363e-01, 2.70193309e-01, 1.47384644e-01,
        5.26564866e-02, 5.60375415e-02, 1.29705272e-03,1.62536576e-01, 2.58769482e-01, 2.21352559e-02], dtype='float32'),
    'a': np.array([1.73093116, 1.73142111, 1.73274243, 1.73260796, 1.73203647, 1.73140657, 1.7309823], dtype='float32'),
    'c': np.array([3.13339353e-01, 3.22756357e-02, 1.76987484e-01, 2.07477063e-01, 1.05615752e-02, 7.54027247e-01, 1.01595912e-02, 5.85010672e+00, 1.06995499e+00, 3.47300959e+00, 3.72587252e+00, 3.02333444e-01, 1.42488728e+01, 3.63102883e-01], dtype='float32'),
}



class StrikerInferenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StrikerInferenceWrapper, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath, max_steps=25000):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        self.net.eval()
        # self.means = {
        #     'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_obs.npy'.format(max_steps)),
        #     's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_s_transition_obs.npy'.format(max_steps)),
        #     'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/mean_correction.npy'.format(max_steps)),
        # }
        #
        # self.std = {
        #     'o': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_obs.npy'.format(max_steps)),
        #     'a': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_actions.npy'.format(max_steps)),
        #     's': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_s_transition_obs.npy'.format(max_steps)),
        #     'c': np.load('/u/alitaiga/repositories/sim-to-real/normalization/striker_{}/std_correction.npy'.format(max_steps)),
        # }
        # self.std['c'][19] = 1.0
        # self.std['c'][22] = 1.0
        self.means = {
            'o': np.array([1.12845600e+00, 2.85855811e-02, 6.05528504e-02, -9.97073829e-01, 6.56818366e-03, 1.40352142e+00,3.84055846e-03, 1.02492295e-01,
                7.11518712e-03, 4.65221144e-02, -2.46240869e-01, 1.76974421e-03, 3.31832051e-01, -6.37228310e-04, 2.20609486e-01, -2.38019347e-01,
                8.07869807e-02, 4.95883346e-01, -1.74267560e-01, -2.71298617e-01, 4.24245656e-01, 5.50004303e-01, -3.22469383e-01], dtype='float32'),
            's': np.array([9.59522069e-01, 2.64616702e-02, -1.47129979e-03, -1.00690019e+00, 8.95156711e-03, -2.26142392e-01, 4.37198719e-03, -3.18137455e+00,
                -8.15565884e-02, -1.13330793e+00, -1.15818921e-02, 3.88979465e-02,-3.10147667e+01, 1.01360520e-02, 2.72106588e-01, -1.99255228e-01, 8.47409219e-02, 4.95847106e-01,
                -1.74359173e-01, -2.71298617e-01, 4.24228728e-01, 5.50020635e-01, -3.22452545e-01], dtype='float32'),
            'c': np.array([1.74270540e-01, 2.45837355e-03, 6.43057898e-02, -2.44984333e-03, -2.27138959e-03, 1.64594924e+00, -5.81401051e-04, 3.28941011e+00, 8.86482969e-02, 1.17964220e+00, -2.34902933e-01, -3.72229367e-02, 3.13308201e+01, -1.16425110e-02], dtype='float32'),
        }

        self.std = {
            'o': np.array([0.81256843, 0.25599328, 1.04780889, 0.78443158, 0.98964226, 0.66285115, 0.99393243, 1.09946322, 0.49455127, 2.67891765,
                 2.24620128, 3.26377749, 2.13994312, 3.29273677, 4.68308300e-01, 2.50030875e-01, 1.48312315e-01, 5.18861376e-02, 5.50917946e-02,
                 1.29705272e-03, 1.62378117e-01, 2.58738667e-01, 2.16630325e-02], dtype='float32'),
            's': np.array([0.57472217, 0.24750113, 1.07578814, 0.78113234, 0.99487597, 0.11505181, 1.00299263, 6.00020313,
                1.16489089, 4.44655132, 4.39647961, 3.27095556, 14.71065998, 3.32328629, 3.99973363e-01, 2.70193309e-01, 1.47384644e-01,
                5.26564866e-02, 5.60375415e-02, 1.29705272e-03,1.62536576e-01, 2.58769482e-01, 2.21352559e-02], dtype='float32'),
            'a': np.array([1.73093116, 1.73142111, 1.73274243, 1.73260796, 1.73203647, 1.73140657, 1.7309823], dtype='float32'),
            'c': np.array([3.13339353e-01, 3.22756357e-02, 1.76987484e-01, 2.07477063e-01, 1.05615752e-02, 7.54027247e-01, 1.01595912e-02, 5.85010672e+00, 1.06995499e+00, 3.47300959e+00, 3.72587252e+00, 3.02333444e-01, 1.42488728e+01, 3.63102883e-01], dtype='float32'),
        }
        print("DBG: MODEL LOADED:",modelPath)

    def _create_input(self, obs_pre_action, action, obs_sim):
        input_ = np.concatenate([
            (obs_pre_action[None,None,:] - self.means['o']) / self.std['o'],
            action[None,None,:] / self.std['a'],
            (obs_sim[None,None,:] - self.means['s']) / self.std['s']
        ], axis=2)
        return Variable(from_numpy(input_).float(), volatile=True)

    def _step(self, action):
        obs_pre_action = self.env.env._get_obs()
        obs_sim, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_pre_action, action, obs_sim)
        obs_real = self.net.forward(variable).data.cpu().numpy()[0,0,:]

        # Udpate the environnement with the new state
        self._set_to_simplus((obs_real * self.std['c']) + self.means['c'])
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

def StrikerPlus(base_env_id):
    return StrikerInferenceWrapper(gym.make(base_env_id))
