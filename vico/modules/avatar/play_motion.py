from .replay_motion_module import ReplayMotionModule
from .utils import AvatarState, ActionStatus, Mirrored_Mixamo_data, Mixamo_data_to_controller_pose
import genesis as gs

class PlayMotion(ReplayMotionModule):
    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name)
        self.mirrored_data = []
        mirrored_motion_data = Mirrored_Mixamo_data(motion_data)
        for i in range(mirrored_motion_data["trans"].shape[0]):
            self.mirrored_data.append(Mixamo_data_to_controller_pose(
                mirrored_motion_data["trans"][i], mirrored_motion_data["rot"][i], mirrored_motion_data["joint"][i]
            ))

    def start(self, hand_id, duration):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        if self.robot.attached_object[hand_id] is None:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: hand {hand_id} is not holding an object.")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.hand_id = hand_id
        self.duration = duration
        self.at_stage = 0
        self.at_frame = 0
    
    def step(self, skip_avatar_animation=False):
        data = self.data if self.hand_id == 1 else self.mirrored_data
        if skip_avatar_animation:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.pose = self.robot.stop_pose
            self.robot.node_trans = self.robot.stop_node
            self.robot.global_mat = self.robot.stop_mat
            self.robot.global_mat_inv = self.robot.stop_mat_inv
            return
        if self.at_stage == 0:
            self.at_frame += 1
            if self.at_frame == len(data) - 1:
                self.at_stage = 1
            self.robot.pose = data[self.at_frame]
            self.robot.node_trans = self.node_data[self.at_frame]
            self.robot.global_mat = self.global_mat
            self.robot.global_mat_inv = self.global_mat_inv
        elif self.at_stage == 1:
            self.duration -= 1
            if self.duration == 0:
                self.at_stage = 2
            self.robot.pose = data[-1]
            self.robot.node_trans = self.node_data[-1]
            self.robot.global_mat = self.global_mat
            self.robot.global_mat_inv = self.global_mat_inv
        elif self.at_stage == 2:
            self.at_frame -= 1
            if self.at_frame == 0:
                self.robot.action_state = AvatarState.NO_ACTION
                self.robot.action_status = ActionStatus.SUCCEED
                self.robot.pose = self.robot.stop_pose
                self.robot.node_trans = self.robot.stop_node
                self.robot.global_mat = self.robot.stop_mat
                self.robot.global_mat_inv = self.robot.stop_mat_inv
            else:
                self.robot.pose = data[self.at_frame]
                self.robot.node_trans = self.node_data[self.at_frame]
                self.robot.global_mat = self.global_mat
                self.robot.global_mat_inv = self.global_mat_inv