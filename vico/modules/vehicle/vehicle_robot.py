import os
import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
from ...tools.utils import get_assets_dir

class VehicleRobot():
    def __init__(self, env, name, vehicle_asset_path, position=np.zeros(3, dtype=np.float64), rotation=np.zeros(3, dtype=np.float64), dt=0.01):
        self.dt = dt
        if not os.path.isabs(vehicle_asset_path):
            vehicle_asset_path = os.path.join(get_assets_dir(), vehicle_asset_path)

        frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
        if ".glb" in vehicle_asset_path:
            self.body = env.add_entity(
                type="vehicle",
                name=name,
                material=frictionless_rigid,
                morph=gs.morphs.Mesh(
                    file=vehicle_asset_path,
                    collision=False,
                ),
            )
        elif ".urdf" in vehicle_asset_path:
            self.body = env.add_entity(
                type="vehicle",
                name=name,
                material=frictionless_rigid,
                morph=gs.morphs.URDF(
                    scale=1.7,
                    file=vehicle_asset_path,
                    pos=position,
                    euler=rotation,
                    collision=False,
                ),
            )
        else:
            self.body = env.add_entity(
                type="vehicle",
                name=name,
                material=frictionless_rigid,
                morph=gs.morphs.Box(
                    lower=(-1.0, -1.0, -0.1),
                    upper=(1.0,  1.0,  0.1),
                    fixed=False,
                    collision=False,
                ),
            )

        self.velocity = np.zeros(3)
        self.angluer_vel = np.zeros(3)

        self.global_trans = np.zeros(3)
        self.global_rot = np.eye(3)
    
    def rotate_yaw(self, angle):
        self.global_rot = geom_utils.euler_to_R((0,0,angle)) @ self.global_rot

    def reset(
        self,
        global_trans: np.ndarray = np.zeros(3),
        global_rot: np.ndarray = np.eye(3),
    ):
        self.global_trans = global_trans
        self.global_rot = global_rot

        self.update()
    
    def transform_q(self, base_trans, base_rot):
        q = np.zeros(1 * 7)
        q[:3] = base_trans
        q[3:7] = geom_utils.R_to_quat(base_rot)
        return q

    def update(self): # update only does pose update, does not do steps!
        _step_q = self.transform_q(self.global_trans, self.global_rot)
        self.body.set_qpos(_step_q, np.array([x for x in range(0, 7)]))
