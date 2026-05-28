import os
import pickle

import time
import tqdm
import numpy as np
import pickle as pkl
import genesis as gs
import genesis.utils.geom as geom_utils
from genesis.utils.misc import get_cvx_cache_dir
from ...tools.asset_utils import ensure_asset

from .utils import *
from ...tools.utils import *
from .avatar_robot import AvatarRobot

from .walk_motion import WalkMotion
from .run_motion import RunMotion
from .turn_motion import TurnMotion
from .sleep_motion import SleepMotion
from .wake_motion import WakeMotion
from .pick_motion import PickMotion
from .put_motion import PutMotion
from .reach_motion import ReachMotion
from .throw_motion import ThrowMotion
from .sit_motion import SitMotion
from .stand_motion import StandMotion
from .open_motion import OpenMotion
from .close_motion import CloseMotion
from .drink_motion import DrinkMotion
from .eat_motion import EatMotion
from .play_motion import PlayMotion
from .enter_bus_motion import EnterBusMotion
from .exit_bus_motion import ExitBusMotion
from .enter_bike_motion import EnterBikeMotion
from .exit_bike_motion import ExitBikeMotion
from .enter_car_motion import EnterCarMotion
from .exit_car_motion import ExitCarMotion
from .play_animation_motion import PlayAnimationMotion
from .reach_obj_motion import ReachObjMotion

Supported_Motions = {
    "walk": (WalkMotion, "walk"),
    "run": (RunMotion, "Running"),
    "turn": (TurnMotion, None),
    "sleep": (SleepMotion, None),
    "wake": (WakeMotion, None),
    "pick": (PickMotion, "pick"),
    "put": (PutMotion, "pick"),
    "reach": (ReachMotion, "reach"),
    "reachobj": (ReachObjMotion, None),
    "throw": (ThrowMotion, "throw"),
    "sit": (SitMotion, "sit"),
    "stand": (StandMotion, "stand"),
    "open": (OpenMotion, "reach"),
    "close": (CloseMotion, "reach"),
    "drink": (DrinkMotion, "drink"),
    "eat": (EatMotion, "drink"),
    "play": (PlayMotion, "play"),
    "enter_bus": (EnterBusMotion, None),
    "exit_bus": (ExitBusMotion, None),  # No motion used for bus
    "enter_bike": (EnterBikeMotion, "entering_car"),
    "exit_bike": (ExitBikeMotion, "exiting_car"),  # The motion used for bike now is the car one
    "enter_car": (EnterCarMotion, "entering_car"),
    "exit_car": (ExitCarMotion, "exiting_car"),
    "play_animation": (PlayAnimationMotion, None),
}


class AvatarController():
    '''
    Avatar Controller, converts high level actions into joints positions.
    '''
    full_motion_data = None
    full_motion_data_path = None

    def __init__(
            self,
            env,
            motion_data_path: str,
            skin_options=None,
            ego_view_options=None,
            frame_ratio=1.0,
            terrain_height_path=None,
            third_person_camera_resolution=None,
            enable_collision=True,
            name=None
    ):
        self.scene = env.scene
        self.robot = AvatarRobot(env, skin_options, name)
        self.box = self.robot.box

        motion_data_path = ensure_asset(motion_data_path)
        if AvatarController.full_motion_data_path is None:
            AvatarController.full_motion_data_path = f"{motion_data_path}.full"

        # Load general stop pose for avatar
        with open(motion_data_path, "rb") as f:
            motion_data = pkl.load(f)
        self.motion_data = motion_data
        self.frame_ratio = frame_ratio
        
        start_time = time.perf_counter()
        self.robot.stop_pose = Mixamo_data_to_controller_pose(
            motion_data["idle"]["trans"][0], motion_data["idle"]["rot"][0], motion_data["idle"]["joint"][0])
        self.robot.stop_mat = motion_data["idle"]["mat"][0]
        self.robot.stop_mat_inv = np.array([np.linalg.inv(m) for m in self.robot.stop_mat])
        vgeom=self.robot.skin.links[0]._vgeoms[0]
        self.robot.stop_node=Mixamo_node_processing(vgeom, self.robot.stop_pose, self.robot.stop_mat, self.robot.stop_mat_inv)

        self.robot.sit_pose = Mixamo_data_to_controller_pose(
            motion_data["stand"]["trans"][0], motion_data["stand"]["rot"][0], motion_data["stand"]["joint"][0])
        self.robot.sit_mat = motion_data["stand"]["mat"][0]
        self.robot.sit_mat_inv = np.array([np.linalg.inv(m) for m in self.robot.sit_mat])
        self.robot.sit_node=Mixamo_node_processing(vgeom, self.robot.sit_pose, self.robot.sit_mat, self.robot.sit_mat_inv)

        self.motion_modules = {}
        
        cache_file = os.path.join(get_cvx_cache_dir(), f"walk_{name + str(frame_ratio)}.pkl")
        if os.path.exists(cache_file):
            motion_iter = Supported_Motions.items()
        else:
            motion_iter = tqdm.tqdm(Supported_Motions.items(), desc="Preprocessing motions", unit="motion")

        for motion_name, (motion_class, motion_data_name) in motion_iter:
            if motion_name == "play_animation":
                for motion_key in motion_data:
                    if motion_key not in Supported_Motions:
                        md = motion_data[motion_key].copy()
                        for k, v in md.items():
                            v_len = max(2, int(v.shape[0] * frame_ratio))
                            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
                        self.motion_modules[motion_key] = motion_class(
                            motion_name=motion_key,
                            motion_data=md,
                            robot=self.robot,
                            name=name + str(frame_ratio)
                        )
            elif motion_data_name is not None:
                if motion_data_name in motion_data:
                    md = motion_data[motion_data_name].copy()
                    for k, v in md.items():
                        v_len = max(2, int(v.shape[0] * frame_ratio))
                        md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
                    self.motion_modules[motion_name] = motion_class(
                        motion_name=motion_name,
                        motion_data=md,
                        robot=self.robot,
                        name=name + str(frame_ratio)
                    )
                else:
                    gs.logger.warning(f"The motion {motion_data_name} is not in motion_data!")
            else:
                self.motion_modules[motion_name] = motion_class(
                    motion_name=motion_name,
                    robot=self.robot,
                )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if elapsed_time >= 1:
            gs.logger.info(f"Avatar processing done, time cost: {elapsed_time:.6f}s")
        else:
            gs.logger.debug(f"Avatar processing done, time cost: {elapsed_time:.6f}s")
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

        self._keep_motion_name = None

        self.terrain_height_field = None
        if terrain_height_path:
            self.terrain_height_field = load_height_field(os.path.join(get_assets_dir(), terrain_height_path))

        self.last_step_trans = None
        self.last_step_rot = None

        if third_person_camera_resolution:
            self.third_person_camera = self.scene.add_camera(
                res=(third_person_camera_resolution, third_person_camera_resolution),
                pos=(0.0, 0.0, 0.0),
                lookat=(0.0, 0.0, 0.0),
                fov=90,
                GUI=False,
                far=16000.0,
            )

        self.enable_collision = enable_collision

    def reset(
            self,
            global_trans: np.ndarray = np.zeros(3, dtype=np.float64),
            global_rot: np.ndarray = np.eye(3, dtype=np.float64),
    ):
        '''
        global_trans and global_rot only contain a rough transformation with no transformations from motion.
        '''
        if not isinstance(global_trans, np.ndarray) or not isinstance(global_rot, np.ndarray):
            raise TypeError("Avatar global trans and global rot should be np.ndarray")
        self.robot.reset(global_trans, global_rot)
        self._keep_motion_name = None

    def reset_with_global_xy(
            self,
            global_xy: np.ndarray = np.zeros(2, dtype=np.float64),
    ):
        if not isinstance(global_xy, np.ndarray):
            raise TypeError("Avatar global xy should be np.ndarray")
        new_height = get_height_at(self.terrain_height_field, global_xy[0], global_xy[1])
        global_trans = np.array([global_xy[0], global_xy[1], new_height], dtype=np.float64)
        global_rot = np.eye(3, dtype=np.float64)
        self.robot.reset(global_trans, global_rot)
        self._keep_motion_name = None

    def get_global_xy(
            self,
    ):
        return self.robot.get_global_xy()

    def get_global_height(
            self,
    ):
        return self.robot.get_global_height()

    def get_global_pose(
            self,
    ):
        return self.robot.get_global_pose()

    def set_global_height(
            self,
            height,
    ):
        self.robot.global_trans[2] = height

        self.robot.update()

    def do(
            self,
            action_name: str,
            **kwargs,
    ):
        if action_name not in self.motion_modules.keys():
            raise NotImplementedError()
        self.motion_modules[action_name].start(**kwargs)

    def step(
            self, skip_avatar_animation=False
    ):
        finished = False
        if self.robot.action_state == AvatarState.NO_ACTION:
            # pass
            if self.robot.base_state != AvatarState.IN_VEHICLE:
                return
        elif self.robot.action_state in self.motion_modules.keys():
            finished = self.motion_modules[self.robot.action_state].step(skip_avatar_animation)

        if self.robot.base_state == AvatarState.IN_VEHICLE and self.robot.action_state == AvatarState.NO_ACTION:
            self.robot.global_trans = self.robot._h_attach_to.robot.global_trans
            self.robot.global_rot = self.robot._h_attach_to.robot.global_rot
        elif self.terrain_height_field is not None and self.robot.global_trans[2] >= 0:  # Auto Height
            # It is still correct to set this height for non-standing case, since it's aligned (the height difference is in the motion pose)
            hx, hy = self.get_global_xy()
            new_height = get_height_at(self.terrain_height_field, hx, hy)
            self.robot.global_trans[2] = new_height

        self.robot.update()

        gs.logger.debug(
            f"action state: {self.robot.action_state}, global rotation: {geom_utils.R_to_quat(self.robot.global_rot)}, global trans: {self.robot.global_trans}"
        )

    def initialize_no_collision(
            self,
            entities,
    ):
        self.robot.initialize_no_collision(entities)

    def post_step(
            self,
            collision_pairs,
    ):
        if not self.enable_collision:
            return
        collide = False
        for i in range(collision_pairs.shape[0]):
            if collision_pairs[i, 0] == self.robot.box.geoms[0].idx and collision_pairs[
                i, 1] not in self.robot.no_collision_idx:
                collide = True
            if collision_pairs[i, 1] == self.robot.box.geoms[0].idx and collision_pairs[
                i, 0] not in self.robot.no_collision_idx:
                collide = True
        if self.robot.base_state == AvatarState.IN_VEHICLE:  # In vehicle no action, or doing vehicle enter/exit action
            collide = False
        if collide:
            if self.last_step_trans is None:
                raise RuntimeError("Collide in the beginning")
            self.reset(self.last_step_trans, self.last_step_rot)
            self.robot.action_status = ActionStatus.COLLIDE
        else:
            self.last_step_trans = self.get_global_pose()[:3]
            self.last_step_rot = self.robot.global_rot

    def spare(
            self,
    ):
        if self.robot.base_state == AvatarState.IN_VEHICLE and self.robot.action_state == AvatarState.NO_ACTION:
            return self.robot._h_attach_to.spare()
        else:
            return self.robot.action_state is AvatarState.NO_ACTION

    def action_status(
            self,
    ):
        return self.robot.action_status

    def attach_to(
            self,
            obj_handle,
    ):
        gs.logger.info(f"attach to {obj_handle}")
        self.robot._h_attach_to = obj_handle

    def get_third_person_camera_rgb(
        self,
        indoor=False
    ):
        if self.robot.base_state == AvatarState.SLEEPING:
            return None
        global_xy = np.array(self.get_global_xy())
        if self.robot.base_state == AvatarState.IN_VEHICLE and self.robot._h_attach_to.name != "bicycle":
            self.third_person_camera.set_pose(pos=np.array([global_xy[0]-self.robot.global_rot[0, 0]*7.0, global_xy[1]-self.robot.global_rot[1, 0]*7.0, self.get_global_height()+5.5]),
                                                                lookat=np.array([global_xy[0], global_xy[1], self.get_global_height()+3.0]))
        elif indoor:
            pose_x=global_xy[0]-self.robot.global_rot[0, 0]*1.5
            pose_y=global_xy[1]-self.robot.global_rot[1, 0]*1.5
            self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height()+2]),
                                                                lookat=np.array([global_xy[0], global_xy[1], self.get_global_height()+1.3]))
        
        else:
            self.third_person_camera.set_pose(pos=np.array([global_xy[0]-self.robot.global_rot[0, 0]*3.0, global_xy[1]-self.robot.global_rot[1, 0]*3.0, self.get_global_height()+3.0]),
                                                                lookat=np.array([global_xy[0], global_xy[1], self.get_global_height()+3.0]))
        rgb, _, _, _ = self.third_person_camera.render(depth=False)
        return rgb

    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        if self.robot.base_state == AvatarState.SLEEPING:
            return None, None, None, self.ego_view.transform
        if self.robot.base_state == AvatarState.IN_VEHICLE:
            if self.robot._h_attach_to.name != "bicycle" and self.robot._h_attach_to.ego_view is not None:
                return self.robot._h_attach_to.render_ego_view(rotation_offset, depth, segmentation)
        head_pos, head_rot = self.robot.skin.get_global_translation("HeadTop_End")
        head_pos[2] -= 0.05 # Lower it to eye level.
        head_pos += head_rot @ np.array([0.2, 0.0, 0.0]) # Move the camera forward by a certain distance to prevent it from seeing the avatar.
        
        self.ego_view.set_pose(pos=head_pos, lookat=head_rot@np.array([1,0,0])+head_pos)
        rgb, depth, seg_idxc_arr, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg_idxc_arr, self.ego_view.transform

    #################### Motions ####################

    def walk(self, distance, speed = 1.0):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["walk"].start(distance, speed)

    def run(self, distance, speed = 1.0):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["run"].start(distance, speed)

    def reachobj(self, hand_id, obj, obj_pos=None):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["reachobj"].start(hand_id, obj, obj_pos)

    def move_forward(self, distance, speed=1):
        if self.robot.base_state == AvatarState.IN_VEHICLE:
            self.robot._h_attach_to.move_forward(
                target_pos=self.robot._h_attach_to.robot.global_trans + distance * self.robot._h_attach_to.robot.global_rot @ np.array(
                    [1.0, 0.0, 0.0])
            )
        else:
            self.walk(distance, speed)

    def run_forward(self, distance, speed=1.0):
        if self.robot.base_state == AvatarState.IN_VEHICLE:
            self.robot._h_attach_to.move_forward(
                target_pos=self.robot._h_attach_to.robot.global_trans + distance * self.robot._h_attach_to.robot.global_rot @ np.array(
                    [speed, 0.0, 0.0])
            )
        else:
            self.walk(distance, speed)

    def sleep(self):
        self.motion_modules["sleep"].start()

    def wake(self):
        self.motion_modules["wake"].start()

    def turn_left(self, angle, turn_frame_limit=15, turn_sec_limit=1500):
        if self.robot.base_state == AvatarState.IN_VEHICLE:
            self.robot._h_attach_to.turn_left_schedule(angle)
        else:
            if self.robot.base_state == AvatarState.SLEEPING:
                self.robot.base_state = AvatarState.STANDING
            self.motion_modules["turn"].start(angle, turn_frame_limit, turn_sec_limit)

    def turn_right(self, angle, turn_frame_limit=15, turn_sec_limit=1500):
        if self.robot.base_state == AvatarState.IN_VEHICLE and self.robot.action_state == AvatarState.NO_ACTION:
            self.robot._h_attach_to.turn_right_schedule(angle)
        else:
            if self.robot.base_state == AvatarState.SLEEPING:
                self.robot.base_state = AvatarState.STANDING
            self.motion_modules["turn"].start(-angle, turn_frame_limit, turn_sec_limit)

    def pick(self, hand_id, obj):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["pick"].start(hand_id, obj)

    def put(self, hand_id, location=None, rotation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["put"].start(hand_id, location, rotation)

    def reach(self, hand_id, obj):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["reach"].start(hand_id, obj)

    def touch(self, hand_id, obj):
        self.motion_modules["reach"].start(hand_id, obj)

    def press(self, hand_id, obj):
        self.motion_modules["reach"].start(hand_id, obj)

    def throw(self, hand_id, orientation=None):
        self.motion_modules["throw"].start(hand_id, orientation)

    def sit(self, obj=None, position=None):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["sit"].start(obj, position)

    def stand_up(self):
        self.motion_modules["stand"].start()

    def open(self, hand_id, obj):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["open"].start(hand_id, obj)

    def close(self, hand_id, obj):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["close"].start(hand_id, obj)

    def drink(self, hand_id):
        self.motion_modules["drink"].start(hand_id)

    def eat(self, hand_id):
        self.motion_modules["eat"].start(hand_id)

    def play(self, hand_id, duration):
        self.motion_modules["play"].start(hand_id, duration)

    def enter_bus(self, bus):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["enter_bus"].start(bus)

    def exit_bus(self):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["exit_bus"].start()

    def enter_bike(self, hand_id, car):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["enter_bike"].start(hand_id, car)

    def exit_bike(self, hand_id):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["exit_bike"].start(hand_id)

    def enter_car(self, hand_id, car):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["enter_car"].start(hand_id, car)

    def exit_car(self, hand_id):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["exit_car"].start(hand_id)

    def play_animation(self, name):
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pickle.load(open(AvatarController.full_motion_data_path, "rb"))
            if name in AvatarController.full_motion_data:
                md = AvatarController.full_motion_data[name].copy()
            else:
                print(f"Motion {name} not found!")
                return
        else:
            md = self.motion_data[name].copy()
        for k, v in md.items():
            v_len = max(1, int(v.shape[0] * self.frame_ratio))
            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
        self.motion_modules[name] = PlayAnimationMotion(
            motion_name=name,
            motion_data=md,
            robot=self.robot,
        )
        self.motion_modules[name].start()

    def keep(self, motion_name):
        assert motion_name in self.motion_modules
        self.robot.action_state = AvatarState.KEEPING
        self._keep_motion_name = motion_name

    def release(self):
        self.robot.action_state = AvatarState.NO_ACTION
