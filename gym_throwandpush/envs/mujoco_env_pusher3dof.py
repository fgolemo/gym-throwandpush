import datetime
import os
import tempfile
import xml.etree.cElementTree as ET

import numpy as np
from gym import error
from gym.envs.mujoco import MujocoEnv

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class MujocoEnvPusher3Dof2(MujocoEnv):
    def __init__(self, model_path, frame_skip, model_parameters):
        assert "torques" in model_parameters
        assert len(model_parameters["torques"]) == 3

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        modified_xml_path = self._modifyXml(fullpath, model_parameters)
        self.model = mujoco_py.MjModel(modified_xml_path)

        self.frame_skip = frame_skip
        self.data = self.model.data
        self.viewer = None

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()

    def _modifyXml(self, xml_file, model_parameters):

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()

        for joint_idx,joint_name in enumerate(["proximal_j_1","distal_j_1", "distal_j_2"]):
            joint = root.find('actuator/motor[@joint="{}"]'.format(joint_name))
            joint.set('gear', str(float(model_parameters["torques"][joint_idx])))

        file_name = os.path.basename(xml_file)
        tmp_dir = tempfile.gettempdir()
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_name_with_date = "{}-{}".format(now, file_name)

        new_file_path = os.path.join(tmp_dir, file_name_with_date)

        tree.write(new_file_path, "UTF-8")

        return new_file_path

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()
