import numpy as np
from .. import *

class AutoAvatar:
    def __init__(self, avatar, simple_avatar, debug=False, logger=None, name="default_name"):
        self.name = name
        self.avatar = avatar # should be an AvatarController
        self.simple_avatar = simple_avatar
        self.forward_speed = None
        self.rotation_speed = None
        self._action_queue = []
        self.debug = debug
        self.logger = logger

    def reset(self, global_trans, global_rot):
        self._action_queue = []
        if global_trans is not None and global_rot is not None:
            self.avatar.reset(global_trans, global_rot)
        self.pose = self.avatar.get_global_pose()
        self.logger.debug(f"Avatar {self.name} reset to {self.pose}.")

    def spare(self):
        return self.avatar.spare()
    
    def step(self):
        if self.avatar.action_status()==ActionStatus.ONGOING:
            return None
        # detect whether hit the current target
        # navigate to the target
        # move
        self.pose = self.avatar.get_global_pose()
        nav_result = self.navigate(self.pose[:2], self.avatar.robot.global_rot, self.simple_avatar.get_pos())
        # print(f"{self.avatar.name} is going from {self.avatar.get_global_pose()[:2]} to {self.simple_avatar.get_pos()}. It's current action queue is {self._action_queue}. It's current nav_result is {nav_result}.")
        if len(self._action_queue) == 0:
            if nav_result is None:
                return None
            elif isinstance(nav_result, list):
                assert len(nav_result) > 0
                self._action_queue.extend(nav_result)
            else:
                self._action_queue.append(nav_result)
        avatar_action = self._action_queue.pop(0)
        if avatar_action is not None:
            self.logger.debug(f"avatar action: {avatar_action}")
            if avatar_action['type'] == 'turn_right_schedule':
                self.avatar.turn_right(angle=avatar_action['arg1'])
            elif avatar_action['type'] == 'turn_left_schedule':
                self.avatar.turn_left(angle=avatar_action['arg1'])
            elif avatar_action['type'] == 'move_forward':
                self.avatar.move_forward(avatar_action['arg2'])
            else:
                raise NotImplementedError(f"avatar action type {avatar_action['type']} is not supported")

    def navigate(self, cur_trans, cur_rot, cur_goal):
        if cur_goal is None:
            return None
		# move
        target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
        R = cur_rot
        cur_rad = np.arctan2(R[1, 0], R[0, 0])
        delta_rad = target_rad - cur_rad
        # print("delta angle:", np.rad2deg(delta_rad))
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi

		# print("delta angle:", np.rad2deg(delta_rad))

        if delta_rad > 0:
            action = [{
                'type': 'turn_left_schedule',
                'arg1': np.rad2deg(delta_rad),
                'arg2': self.rotation_speed,
                'arg3': cur_goal
            }]
        else:
            action = [{
                'type': 'turn_right_schedule',
                'arg1': np.rad2deg(-delta_rad),
                'arg2': self.rotation_speed,
                'arg3': cur_goal
            }]
        action.append({
            'type': 'move_forward',
            'arg1': cur_goal,
            'arg2': np.linalg.norm(cur_goal - cur_trans),
            'arg3': cur_goal
        })
        return action