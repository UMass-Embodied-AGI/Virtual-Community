import numpy as np
import time
import random
import os
from .agent import Agent
from .scene_graph import SceneGraph
from ..tools.utils import is_near_goal

class TourAgent(Agent):
	def __init__(self, name, pose, info, sim_path, tour_spatial_memory, no_react=False, debug=False, logger=None):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		self.detect_interval = 1
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, fov=self.fov, debug=self.debug, logger=self.logger)
		self.tour_spatial_memory = tour_spatial_memory
		self.tour_building_names = [building for building in tour_spatial_memory.keys() if tour_spatial_memory[building]["bounding_box"] is not None]
		random.shuffle(self.tour_building_names)
		self.curr_idx= 0
		self.curr_goal_name = self.tour_building_names[self.curr_idx]
		curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
		self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
		self.curr_goal_bbox = curr_goal_dict["bounding_box"]

	def reset(self, name, pose):
		super().reset(name, pose)
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, fov=self.fov, debug=self.debug, logger=self.logger) #todo: log the num_frames
		self.curr_idx= 0
		self.curr_goal_name = self.tour_building_names[self.curr_idx]
		curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
		self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
		self.curr_goal_bbox = curr_goal_dict["bounding_box"]

	def _process_obs(self, obs):
		self.s_mem.update(obs)
		self.obs = obs

	def _act(self, obs):
		if obs['current_place'] is not None:
			return {
				'type': 'enter',
				'arg1': 'open space'
			}
		if is_near_goal(self.pose[0], self.pose[1], self.curr_goal_bbox, self.curr_goal_pos):
			self.curr_idx += 1
			if self.curr_idx == len(self.tour_building_names):
				exit()
			self.curr_goal_name = self.tour_building_names[self.curr_idx]
			curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
			self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
			self.curr_goal_bbox = curr_goal_dict["bounding_box"]
		self.logger.info(f"{self.name} is navigating to {self.curr_goal_name} at {self.curr_goal_pos}.")
		start = time.time()
		obs["goal_building"] = self.tour_building_names[self.curr_idx]
		obs["goal_pos_xy"] = [self.tour_spatial_memory[self.curr_goal_name]['places'][0]['location'][0], self.tour_spatial_memory[self.curr_goal_name]['places'][0]['location'][1]]
		nav_action = self.navigate(self.s_mem.get_sg(), self.curr_goal_pos, goal_bbox=self.curr_goal_bbox)
		self.logger.info(f"Navigation used {time.time() - start}s.")
		return nav_action
	
	def navigate(self, sg, goal_pos, goal_bbox=None):
		from ..tools.utils import get_bbox, get_axis_aligned_bbox
		if goal_pos is None:
			return None
		cur_trans = np.array(self.pose[:2])
		goal_bbox = get_bbox(goal_bbox, goal_pos)
		path = sg.volume_grid_builder.navigate(cur_trans, goal_bbox, self.last_path)
		self.last_path = None
		if path is None:
			self.logger.error(f"No path found when navigate agent {self.name} to {goal_pos}.")
			return None
		else:
			if self.current_vehicle == "bicycle":
				nav_grid_num = int(self.BIKE_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			elif self.current_vehicle is None:
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			else:
				self.logger.error(f"Unsupported vehicle type {self.current_vehicle} for navigation.")
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			cur_goal = path[min(nav_grid_num, len(path) - 1)]
			if sg.volume_grid_builder.has_obstacle(get_axis_aligned_bbox(np.array([cur_goal, cur_trans]), None)):
				cur_goal = path[min(2, len(path) - 1)]
		
		self.logger.debug(f"Path {path[:3]}\n...\n{path[-3:]}")
		from .sg.builder.volume_grid import convex_hull, dist_to_hull
		dist = dist_to_hull(path[-1], convex_hull(goal_bbox))
		if dist > 2:
			self.logger.warning(f"Unable to find a path to the target bounding box. The optimal available path is still a distance of {dist} away from the target bounding box. The optimal path has been automatically adopted.")
		if self.action_status == "COLLIDE":
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is colliding with obstacles, path found was {path}.")
		# move
		target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
		delta_rad = target_rad - self.pose[-1]
		if delta_rad > np.pi:
			delta_rad -= 2 * np.pi
		elif delta_rad < -np.pi:
			delta_rad += 2 * np.pi

		if delta_rad > np.deg2rad(15):
			action = {
				'type': 'turn_left',
				'arg1': np.rad2deg(delta_rad),
			}
			self.last_path = path
		elif delta_rad < -np.deg2rad(15):
			action = {
				'type': 'turn_right',
				'arg1': np.rad2deg(-delta_rad),
			}
			self.last_path = path
		else: action = {
			'type': 'move_forward',
			'arg1': np.linalg.norm(cur_goal - cur_trans),
		}
		if action['arg1'] < 0.1:
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is too close, path found was {path}.")
		return action
