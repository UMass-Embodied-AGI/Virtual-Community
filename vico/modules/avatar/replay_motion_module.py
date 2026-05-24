import os
import pickle
import numpy as np
from .base_motion_module import BaseMotionModule
from .avatar_robot import AvatarRobot
from .utils import Mixamo_data_to_controller_pose, Mixamo_node_processing
from genesis.utils.misc import get_cvx_cache_dir
import genesis as gs

class ReplayMotionModule(BaseMotionModule):
    def __init__(
        self,
        motion_name: str,
        motion_data,
        robot: AvatarRobot,
        name=None
    ):
        super().__init__(motion_name, robot)
        self.data = []
        self.raw_data = []
        self.node_data = []
        self.global_mat = motion_data["mat"][0]
        self.global_mat_inv = np.array([np.linalg.inv(m) for m in self.global_mat])


        cached = False
        if name is not None:
            cache_file = os.path.join(get_cvx_cache_dir(), f"{motion_name}_{name}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    self.node_data = pickle.load(f)
                # print(f"Loaded cached node data from {cache_file}")
                cached = True

        for i in range(motion_data["trans"].shape[0]):
            self.data.append(Mixamo_data_to_controller_pose(
                motion_data["trans"][i], motion_data["rot"][i], motion_data["joint"][i]
            ))
            self.raw_data.append(motion_data["joint"][i])
            vgeom = robot.skin.links[0]._vgeoms[0]
            if not cached:
                self.node_data.append(Mixamo_node_processing(vgeom, self.data[-1], self.global_mat, self.global_mat_inv))
        
        if name is not None and not cached:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.node_data, f)
            # print(f"Cached node data saved to {cache_file}")
        self.at_frame = 0