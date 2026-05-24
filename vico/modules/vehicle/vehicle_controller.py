import os
import numpy as np
import pickle as pkl
from enum import Enum
import time

import genesis as gs
import genesis.utils.geom as geom_utils

from ...tools.utils import *

from .vehicle_robot import VehicleRobot

class VehicleState(Enum):
    IDLE    = "BASE_IDLE"
    MOVING  = "BASE_MOVING"
    TURNING = "BASE_TURNING"
    NAVIGATING = "BASE_NAVIGATING"

class VehicleController():
    '''
    Vehicle Controllers, convert highlevel commands into vehicle actions
    '''
    def __init__(self,
        env,
        name,
        vehicle_asset_path,
        ego_view_options = None,
        position = np.zeros(3, dtype=np.float64),
        rotation = np.zeros(3, dtype=np.float64),
        dt = 1e-2,
        forward_speed_m_per_s = 5,
        angular_speed_deg_per_s = 360,
        terrain_height_path = None,
    ):
        self.scene = env.scene
        self.name = name
        self.dt = dt
        self.robot = VehicleRobot(env, name, vehicle_asset_path, position, rotation, self.dt)
        self.target_angle = 0
        self.cur_angle = 0
        self.target_pos = None
        self.state = VehicleState.IDLE
        self.occupied = False
        self.forward_speed = forward_speed_m_per_s
        self.angular_speed = angular_speed_deg_per_s

        if ego_view_options is not None:
            self.ego_view = self.scene.add_camera(
                res=ego_view_options["res"],
                pos=(0.0, 0.0, 0.0),
                lookat=(1.0, 0.0, 0.0),
                fov=ego_view_options["fov"],
                GUI=ego_view_options["GUI"],
                far=16000.0,
            )
        else:
            self.ego_view = None

        self.terrain_height_field = None
        if terrain_height_path:
            terrain_height_path = os.path.join(get_assets_dir(), terrain_height_path)
            self.terrain_height_field = load_height_field(terrain_height_path)
    
    def reset(
        self,
        global_trans: np.ndarray = np.zeros(3, dtype=np.float64),
        global_rot: np.ndarray = np.eye(3, dtype=np.float64),
    ):
        '''
        global_trans and global_rot only contain a rough transformation with no transformations from motion.
        '''
        new_global_trans = global_trans.copy()
        if self.terrain_height_field is not None: # Auto Height
            hx, hy = new_global_trans[:2]
            new_height = get_height_at(self.terrain_height_field, hx, hy)
            new_global_trans[2] = new_height
        self.robot.reset(new_global_trans, global_rot)
        self.target_angle = 0
        self.cur_angle = 0
        self.state = VehicleState.IDLE
        self.occupied = False

    def set_target_pos(
        self, target_pos
    ):
        self.target_pos = target_pos
        self.distance = -1.

    def get_global_xy(
        self,
    ):
        return self.robot.global_trans[0], self.robot.global_trans[1]
    
    def get_global_height(
        self,
    ):
        return self.robot.global_trans[2]
    
    def get_global_pose(
        self,
    ):
        x, y = self.get_global_xy()
        z = self.get_global_height()
        ypr = geom_utils.R_to_ypr(self.robot.global_rot)
        return np.array([x, y, z, ypr[2], ypr[1], ypr[0]], dtype=self.robot.global_trans.dtype)

    def render_ego_view(
        self,
        rotation_offset = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        depth = False,
        segmentation = False,
    ):
        camera_pos = self.robot.global_trans
        camera_rot = self.robot.global_rot
        camera_pos = camera_rot @ np.array([6.0, 0.0, 4.0]) + camera_pos
        self.ego_view.set_pose(pos=camera_pos, lookat=camera_rot@np.array([1,0,0])+camera_pos)
        rgb, depth, seg, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg, self.ego_view.transform
    
    def stop(self):
        self.state = VehicleState.IDLE
        self.robot.velocity = np.zeros(3)
        self.robot.angluer_vel = np.zeros(3)
        self.target_pos = None
        self.target_angle = 0
        self.cur_angle = 0

    def move_forward(self, target_pos, speed=None):
        if speed == None:
            speed = self.forward_speed
        self.speed = speed
        self.state = VehicleState.MOVING
        self.set_target_pos(target_pos)
        self.robot.velocity = np.array((speed,0,0))
    
    def turn_left(self, angle=30):
        self.robot.rotate_yaw(angle=angle)
    
    def turn_right(self, angle=30):
        self.robot.rotate_yaw(angle=-angle)

    def turn_left_schedule(self, angle=30, speed=None):
        if speed == None:
            speed = self.angular_speed
        self.state = VehicleState.TURNING
        self.target_angle = angle
        self.cur_angle = 0
        self.robot.angluer_vel = np.array((0,0,speed))
    
    def turn_right_schedule(self, angle=30, speed=None):
        if speed == None:
            speed = self.angular_speed
        self.state = VehicleState.TURNING
        self.target_angle = -angle
        self.cur_angle = 0
        self.robot.angluer_vel = np.array((0,0,-speed))
    
    def spare(self):
        return self.state == VehicleState.IDLE
    
    def step(self):
        if self.spare(): return 0 # a great optimization - can save time by 33%
        delta_angle = 0 # the angle the vehicle actually turns in this step
        if self.cur_angle is not None and self.target_angle is not None:
            # if abs(self.target_angle)  > 1e-10:
            #     print("vehicle step cur_angle:", self.cur_angle, "target_angle:", self.target_angle)
            if abs(self.cur_angle - self.target_angle) >= abs(self.robot.angluer_vel[-1] * self.dt): # 0.5 is safe because dt is 1e-2 and angular vel is 30
                delta_angle = self.robot.angluer_vel[-1] * self.dt
                self.cur_angle += delta_angle
            else:
                delta_angle = self.target_angle - self.cur_angle
                self.state = VehicleState.MOVING if np.any(self.robot.velocity) else VehicleState.IDLE
                self.robot.angluer_vel = np.zeros(3)
                self.target_angle = 0
                self.cur_angle = 0
        if self.target_pos is not None:
            if self.near(pos=self.target_pos):
                self.robot.global_trans = np.array([self.target_pos[0], self.target_pos[1], 3.0])
                self.state = VehicleState.TURNING if np.any(self.robot.angluer_vel) else VehicleState.IDLE
                self.robot.velocity = np.zeros(3)
                self.target_pos = None

        # Calculate restricted velocity (velocity should be low when turning sharply - this is also what we do in the actual situation)
        restricted_speed = self.robot.velocity[0]
        if delta_angle > 0:
            restricted_speed = min(restricted_speed, 300.0/(delta_angle/self.dt))
        restricted_velocity = np.array((restricted_speed,0.,0.))
        # Pose stepping according to velocity
        self.robot.global_rot = self.robot.global_rot @ geom_utils.euler_to_R(np.array((0,0,delta_angle)))
        self.robot.global_trans = self.robot.global_rot @ (restricted_velocity * self.dt) + self.robot.global_trans

        if self.terrain_height_field is not None: # Auto Height
            hx, hy = self.get_global_xy()
            new_height = get_height_at(self.terrain_height_field, hx, hy)
            self.robot.global_trans[2] = new_height
        
        self.robot.update()
        return
        # gs.logger.info(f"velocity: {self.robot.velocity}, global trans: {self.robot.total_trans}")
        # gs.logger.info(f"target angle: {self.target_angle}, current angle: {self.cur_angle}, state: {self.state}")
    
    def near(self, pos, threshold=1):
        threshold = max(threshold, self.dt * self.speed + 0.1)
        if self.distance > 0 and self.distance < np.sqrt(sum(np.power((pos[:2] - self.robot.global_trans[:2]), 2))):
            return True # Vehicles shall stop when the distance is becoming longer (This is important since in current vehicle controller version, vehicles won't perform turning strictly before moving, while the traffic manager assumes so.)
        self.distance = np.sqrt(sum(np.power((pos[:2] - self.robot.global_trans[:2]), 2)))
        if self.distance < threshold:
            return True
        else:
            return False
        