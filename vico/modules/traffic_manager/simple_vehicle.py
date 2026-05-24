import numpy as np

class SimpleVehicle:
    '''
    used in traffic manager
    '''
    def __init__(self, id, init_pos, init_rot, init_road, init_s, std_speed=5.):
        self.id = id
        self.pos = init_pos
        self.rot = init_rot
        self.road = init_road
        self.s = init_s
        self.action = {"type": "stop"}
        self.std_speed = std_speed

    def set_pos(self, pos):
        self.pos=pos

    def get_pos(self):
        return self.pos

    def set_loc(self, road, s):
        self.road=road
        self.s=s

    def get_loc(self):
        return self.road, self.s
    
    def set_rot(self, rot):
        self.rot=rot

    def get_rot(self):
        return self.rot
    
class AutoVehicle:
    def __init__(self, vehicle, simple_vehicle, debug=False, logger=None):
        self.vehicle = vehicle
        self.simple_vehicle = simple_vehicle
        self.forward_speed = None
        self.rotation_speed = None
        self._action_queue = []
        self.debug = debug
        self.logger = logger

    def reset(self, global_trans, global_rot):
        self._action_queue = []
        if global_trans is not None and global_rot is not None:
            self.vehicle.reset(global_trans, global_rot)
        self.pose = self.vehicle.get_global_pose()
        self.logger.debug(f"Vehicle {self.vehicle.name} reset to {self.pose}.")

    def spare(self):
        return self.vehicle.spare()
    
    def status(self):
        self.pose = self.vehicle.get_global_pose()
        return f"{self.vehicle.name} is going from {self.pose[:2]} to {self.simple_vehicle.get_pos()}. It's current action queue is {self._action_queue}. Current state is {self.vehicle.state}"
    
    def step(self):
        # detect whether hit the current target
        # navigate to the target
        # move
        self.pose = self.vehicle.get_global_pose()
        nav_result = self.navigate(self.pose[:2], self.vehicle.robot.global_rot, self.simple_vehicle.get_pos())
        # print(f"{self.vehicle.name} is going from {self.pose[:2]} to {self.simple_vehicle.get_pos()}. It's current action queue is {self._action_queue}. It's current nav_result is {nav_result}.")
        if len(self._action_queue) == 0:
            if nav_result is None:
                return None
            elif isinstance(nav_result, list):
                assert len(nav_result) > 0
                self._action_queue.extend(nav_result)
            else:
                self._action_queue.append(nav_result)
        vehicle_action = self._action_queue.pop(0)
        if vehicle_action is not None:
            self.logger.debug(f"vehicle action: {vehicle_action}")
            if vehicle_action['type'] == 'move_forward':
                self.vehicle.move_forward(target_pos=vehicle_action['arg1'], speed=vehicle_action['arg2'])
            elif vehicle_action['type'] == 'turn_left_schedule':
                self.vehicle.turn_left_schedule(angle=vehicle_action['arg1'], speed=vehicle_action['arg2'])
            elif vehicle_action['type'] == 'turn_right_schedule':
                self.vehicle.turn_right_schedule(angle=vehicle_action['arg1'], speed=vehicle_action['arg2'])
            else:
                raise NotImplementedError(f"vehicle action type {vehicle_action['type']} is not supported")

    def navigate(self, cur_trans, cur_rot, cur_goal):
        if cur_goal is None:
            return None
		# move
        target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
        R = cur_rot
        cur_rad = np.arctan2(R[1, 0], R[0, 0])
        delta_rad = target_rad - cur_rad
        # print("delta angle:", np.rad2deg(target_rad), np.rad2deg(cur_rad))
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi

		# print("delta angle:", np.rad2deg(delta_rad))

        if np.linalg.norm(cur_trans-cur_goal) >=1:
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
        else:
            action = []
        action.append({
            'type': 'move_forward',
            'arg1': cur_goal,
            'arg2': self.forward_speed, # np.linalg.norm(cur_goal - cur_trans),
            'arg3': cur_goal
        })
        return action