import gym
import numpy as np
from gym import spaces
from torch import from_numpy, load
from torch.autograd import Variable

PUSHER3DOF_POS_MAX = 2.6
PUSHER3DOF_POS_MIN = -0.2
PUSHER3DOF_VEL_MAX = 12.5
PUSHER3DOF_VEL_MIN = -2.5
PUSHER3DOF_POS_DIFF = PUSHER3DOF_POS_MAX - PUSHER3DOF_POS_MIN
PUSHER3DOF_VEL_DIFF = PUSHER3DOF_VEL_MAX - PUSHER3DOF_VEL_MIN


class Pusher3DofInferenceWrapper2(gym.Wrapper):
    def __init__(self, env):
        super(Pusher3DofInferenceWrapper2, self).__init__(env)
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net = self.net.cpu()
        print("DBG: MODEL LOADED:", modelPath)

    def _normalize(self, state):
        ## measured:
        ## max_pos: 2.5162039
        ## min_pos: -0.1608184
        ## max_vel: 12.24464
        ## min_vel: -2.2767675
        state[:3] -= PUSHER3DOF_POS_MIN  # add the minimum
        state[:3] /= PUSHER3DOF_POS_DIFF  # divide by range to bring into [0,1]

        state[3:] -= PUSHER3DOF_VEL_MIN
        state[3:] /= PUSHER3DOF_VEL_DIFF

        state *= 2  # double and
        state -= 1  # shift left by one to bring into range [-1,1]

        return state

    def _denormalize(self, state):
        state += 1
        state /= 2 # now it's back in range [0,1]

        state[:3] *= PUSHER3DOF_POS_DIFF
        state[:3] += PUSHER3DOF_POS_MIN # now it's uncentered and shifted

        state[3:] *= PUSHER3DOF_VEL_DIFF
        state[3:] += PUSHER3DOF_VEL_MIN

        return state


    def _create_input(self, obs_next, action, obs_current):
        _input = np.hstack([
            self._normalize(obs_next[:6]),
            self._normalize(obs_current[:6]),
            action
        ])
        return Variable(from_numpy(_input).float().unsqueeze(0).unsqueeze(0), volatile=True)

    def _step(self, action):
        obs_current = self.env.env._get_obs()
        obs_next, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_next.copy(), action.copy(), obs_current.copy())
        obs_correction = self.net.forward(variable).data.cpu().squeeze(0).squeeze(0).numpy()
        # print(np.around(obs_next, 2), np.around(obs_correction.data.cpu().numpy(), 2))

        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        # FIRST NORMALIZE SIMULATED RESULTS,
        normalized_obs = self._normalize(obs_next[:6].copy())
        # THEN APPLY CORRECTION,
        corrected_obs = normalized_obs + obs_correction
        # DENORMALIZE, AND APPLY INTERNALLY
        denormalized_obs = self._denormalize(corrected_obs)

        qpos = np.hstack((denormalized_obs[:3], qpos[3:]))
        qvel = np.hstack((denormalized_obs[3:], qvel[3:]))

        self.env.env.set_state(qpos, qvel)

        # Update the environnement with the new state
        return self.env.env._get_obs(), rew, done, info

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()


def Pusher3Dof2Plus2(base_env_id):
    return Pusher3DofInferenceWrapper2(gym.make(base_env_id))
