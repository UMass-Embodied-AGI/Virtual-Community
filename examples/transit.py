import argparse
import json
import os
import random
import numpy as np
import shutil, errno
from datetime import datetime

import genesis as gs

from vico.agents.agent import Agent
from vico.agents.scene_graph import SceneGraph
from vico.tools.utils import *
from vico.env import VicoEnv
from vico.agents import AgentProcess
from vico.modules import *

class TransitAgent(Agent):
	def __init__(self, name, pose, info, sim_path, spatial_memory, transit_info, no_react=False, debug=False, logger=None):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		self.detect_interval = 1
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, debug=self.debug, logger=self.logger)
		self.spatial_memory = spatial_memory
		self.tour_building_names = [building for building in spatial_memory.keys() if spatial_memory[building]["bounding_box"] is not None]
		self.goal_place = self.tour_building_names[random.randint(0, len(self.tour_building_names) - 1)]
		self.logger.info(f"{self.name}'s goal place is {self.goal_place}.")
		goal_place_dict = self.spatial_memory[self.goal_place]
		self.goal_place_pos = np.array([goal_place_dict['places'][0]['location'][0], goal_place_dict['places'][0]['location'][1]])
		self.goal_place_bbox = goal_place_dict["bounding_box"]
		self.curr_idx = 0
		self.transit_info = transit_info
		all_transit = {
			"walk": ["nav_to_goal", "idle"], 
			"bicycle": ["nav_to_bicycle", "enter_bike", "nav_to_goal", "exit_bike", "idle"], 
			"bus": ["nav_to_bus", "enter_bus", "drop_off", "nav_to_goal", "exit_bike", "idle"]
			}
		self.transit_type = random.choice(list(all_transit.keys()))
		if self.transit_type == "walk":
			self.sequential_goals = all_transit["walk"]
		elif self.transit_type == "bicycle":
			self.sequential_goals = all_transit["bicycle"]
			bicycle_stations = self.transit_info["bicycle"]["stations"]
			self.bicycle_station_pos = bicycle_stations[random.choice(list(bicycle_stations.keys()))]
		elif self.transit_type == "bus":
			self.sequential_goals = all_transit["bus"]
			bus_schedule = self.transit_info["bus"]["schedule"][random.choice(list(self.transit_info["bus"]["schedule"].keys()))]
			starting_bus_stop_name = bus_schedule[0]["stop_name"]
			ending_bus_stop_name = bus_schedule[-1]["stop_name"]
			bus_stops = self.transit_info["bus"]["stops"]
			self.starting_bus_stop_pos, self.starting_bus_stop_rot = bus_stops[starting_bus_stop_name]["position"], bus_stops[starting_bus_stop_name]["target_rad"]
			self.ending_bus_stop_pos, self.ending_bus_stop_rot = bus_stops[ending_bus_stop_name]["position"], bus_stops[ending_bus_stop_name]["target_rad"]

	def reset(self, name, pose):
		super().reset(name, pose)
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, debug=self.debug, logger=self.logger) #todo: log the num_frames
		self.curr_idx = 0
		

	def _process_obs(self, obs):
		self.s_mem.update(obs)
		if self.debug:
			self.s_mem.get_sg().volume_grid_builder.get_occ_map(self.pose[:3], os.path.join(self.storage_path, "semantic_memory", f"occ_map_{self.s_mem.num_frames}.png"))
		self.obs = obs
	
	def _act(self, obs):
		if obs['current_place'] is not None:
			return {
				'type': 'enter',
				'arg1': 'open space'
			}
		if self.sequential_goals[self.curr_idx] == "nav_to_bus":
			cur_trans = np.array(self.pose[:2])
			self.curr_goal_rot = self.starting_bus_stop_rot
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}. {np.linalg.norm(np.array([cur_trans[0], cur_trans[1]]) - self.starting_bus_stop_pos)} meters left to the bus.")
			if is_near_goal(cur_trans[0], cur_trans[1], None, np.array(self.starting_bus_stop_pos), threshold=1):
				rot_action = self.rotate()
				self.curr_idx += 1
				self.curr_goal_rot = None
				return rot_action
			else:
				nav_action = self.navigate(self.s_mem.get_sg(), np.array(self.starting_bus_stop_pos), goal_bbox=None)
				return nav_action
		elif self.sequential_goals[self.curr_idx] == "enter_bus":
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}.")
			cur_trans = np.array(self.pose[:2])
			if "bus" in obs['accessible_places']:
				self.curr_idx += 1
				return {
					'type': 'enter_bus',
					'arg1': None
				}
			else:
				return None
		elif self.sequential_goals[self.curr_idx] == "drop_off":
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}.")
			cur_trans = np.array(self.pose[:2])
			if obs["curr_time"].time() > datetime.strptime("09:03:00", "%H:%M:%S").time() and is_near_goal(cur_trans[0], cur_trans[1], None, np.array(self.ending_bus_stop_pos), threshold=8):
				self.curr_idx += 1
				return {
					'type': 'exit_bus'
				}
			else:
				return None
		elif self.sequential_goals[self.curr_idx] == "nav_to_bicycle":
			cur_trans = np.array(self.pose[:2])
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}. {np.linalg.norm(np.array([cur_trans[0], cur_trans[1]]) - self.bicycle_station_pos)} meters left to the bicycle.")
			if is_near_goal(cur_trans[0], cur_trans[1], None, np.array(self.bicycle_station_pos)):
				self.curr_idx += 1
			else:
				nav_action = self.navigate(self.s_mem.get_sg(), np.array(self.bicycle_station_pos), goal_bbox=None)
				return nav_action
		elif self.sequential_goals[self.curr_idx] == "enter_bike":
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}.")
			self.curr_idx += 1
			return {
				'type': 'enter_bike'
			}
		elif self.sequential_goals[self.curr_idx] == "nav_to_goal":
			cur_trans = np.array(self.pose[:2])
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}. {np.linalg.norm(np.array([cur_trans[0], cur_trans[1]]) - np.array([self.goal_place_pos[0] - 1000, self.goal_place_pos[1] - 1000]))} meters left to the goal place.")
			if is_near_goal(cur_trans[0], cur_trans[1], self.goal_place_bbox, np.array(self.goal_place_pos), threshold=3):
				self.curr_idx += 1
			else:
				nav_action = self.navigate(self.s_mem.get_sg(), np.array(self.goal_place_pos), goal_bbox=self.goal_place_bbox)
				return nav_action
		elif self.sequential_goals[self.curr_idx] == "exit_bike":
			self.logger.info(f"{self.name}'s current subgoal is {self.sequential_goals[self.curr_idx]}.")
			self.goal_place_pos = None
			self.curr_idx += 1
			return {
				'type': 'exit_bike'
			}
		elif self.sequential_goals[self.curr_idx] == "idle":
			self.logger.info(f"{self.name} has reached the goal place {self.goal_place}.")
			return None

	def rotate(self):
		if self.curr_goal_rot is None:
			return None
		delta_rad = self.curr_goal_rot - self.pose[-1]
		if delta_rad > 0:
			action = {
				'type': 'turn_left',
				'arg1': np.rad2deg(delta_rad),
			}
		else:
			action = {
				'type': 'turn_right',
				'arg1': np.rad2deg(-delta_rad),
			}
		return action

	def navigate(self, sg, goal_pos, goal_bbox=None):
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



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--precision", type=str, default='32')
	parser.add_argument("--logging_level", type=str, default='info')
	parser.add_argument("--backend", type=str, default='gpu')
	parser.add_argument("--head_less", '-l', action='store_true')
	parser.add_argument("--multi_process", '-m', action='store_true')
	parser.add_argument("--model_server_port", type=int, default=0)
	parser.add_argument("--output_dir", "-o", type=str, default='output')
	parser.add_argument("--debug", action='store_true')
	parser.add_argument("--overwrite", action='store_true')
	parser.add_argument("--challenge", type=str, default='full')

	### Simulation configurations
	parser.add_argument("--resolution", type=int, default=512)
	parser.add_argument("--enable_collision", action='store_true')
	parser.add_argument("--skip_avatar_animation", action='store_true')
	parser.add_argument("--enable_gt_segmentation", action='store_true')
	parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
	parser.add_argument("--save_per_seconds", type=int, default=10)
	parser.add_argument("--enable_third_person_cameras", action='store_true')
	parser.add_argument("--curr_time", type=str)

	### Scene configurations
	parser.add_argument("--scene", type=str, default='NY')
	parser.add_argument("--no_load_indoor_scene", action='store_true')
	parser.add_argument("--no_load_indoor_objects", action='store_true')
	parser.add_argument("--no_load_outdoor_objects", action='store_true')
	parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
	parser.add_argument("--no_load_scene", action='store_true')

	# Traffic configurations
	parser.add_argument("--no_traffic_manager", action='store_true')
	parser.add_argument("--tm_vehicle_num", type=int, default=0)
	parser.add_argument("--tm_avatar_num", type=int, default=0)
	parser.add_argument("--enable_tm_debug", action='store_true')

	### Agent configurations
	parser.add_argument("--num_agents", type=int, default=15)
	parser.add_argument("--config", type=str, default='agents_num_15')
	parser.add_argument("--agent_type", type=str, choices=['transit_agent'], default='transit_agent')
	parser.add_argument("--detect_interval", type=int, default=1)

	args = parser.parse_args()

	args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

	if args.overwrite and os.path.exists(args.output_dir):
		print(f"Overwrite the output directory: {args.output_dir}")
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)
	config_path = os.path.join(args.output_dir, 'curr_sim')
	if not os.path.exists(config_path):
		seed_config_path = os.path.join('assets/scenes', args.scene, args.config)
		print(f"Initiate new simulation from config: {seed_config_path}")
		try:
			shutil.copytree(seed_config_path, config_path)
		except OSError as exc:
			if exc.errno in (errno.ENOTDIR, errno.EINVAL):
				shutil.copy(seed_config_path, config_path)
			else:
				raise
	else:
		print(f"Continue simulation from config: {config_path}")

	if args.debug:
		args.enable_third_person_cameras = True
		if args.curr_time is not None:
			config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
			config['curr_time'] = args.curr_time
			atomic_save(os.path.join(config_path, 'config.json'), json.dumps(config, indent=4, default=json_converter))

	os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

	env = VicoEnv(
		seed=args.seed,
		precision=args.precision,
		logging_level=args.logging_level,
		backend= gs.cpu if args.backend == 'cpu' else gs.gpu,
		head_less=args.head_less,
		resolution=args.resolution,
		challenge=args.challenge,
		num_agents=args.num_agents,
		config_path=config_path,
		scene=args.scene,
		enable_indoor_scene=not args.no_load_indoor_scene,
		enable_outdoor_objects=not args.no_load_outdoor_objects,
		outdoor_objects_max_num=args.outdoor_objects_max_num,
		enable_collision=args.enable_collision,
		skip_avatar_animation=args.skip_avatar_animation,
		enable_gt_segmentation=args.enable_gt_segmentation,
		no_load_scene=args.no_load_scene,
		output_dir=args.output_dir,
		enable_third_person_cameras=args.enable_third_person_cameras,
		no_traffic_manager=args.no_traffic_manager,
		enable_tm_debug=args.enable_tm_debug,
		tm_vehicle_num=args.tm_vehicle_num,
		tm_avatar_num=args.tm_avatar_num,
		save_per_seconds=args.save_per_seconds,
		defer_chat=True,
		debug=args.debug,
		enable_indoor_objects=not args.no_load_indoor_objects,
	)

	agents = []
	for i in range(args.num_agents):
		basic_kwargs = dict(
			name = env.agent_names[i],
			pose = env.config["agent_poses"][i],
			info = env.agent_infos[i],
			sim_path = config_path,
			debug = args.debug,
			logging_level = args.logging_level,
			multi_process = args.multi_process
		)
		if args.agent_type == 'transit_agent':
			agents.append(AgentProcess(TransitAgent, **basic_kwargs, spatial_memory=env.building_metadata, transit_info=env.transit_info))
		else:
			raise NotImplementedError(f"agent type {args.agent_type} is not supported in the transit example.")

	if args.multi_process:
		gs.logger.info("Start agent processes")
		for agent in agents:
			agent.start()
		gs.logger.info("Agent processes started")

	# Simulation loop
	obs = env.reset()
	agent_actions = {}
	agent_actions_to_print = {}
	args.max_steps = args.max_seconds // env.sec_per_step
	while True:
		lst_time = time.perf_counter()
		for i, agent in enumerate(agents):
			agent.update(obs[i])
		for i, agent in enumerate(agents):
			agent_actions[i] = agent.act()
			agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None
			if agent_actions[i] is not None and agent_actions[i]['type'] == 'converse':
				agent_actions[i]['request_chat_func'] = agent.request_chat
				agent_actions[i]['get_utterance_func'] = agent.get_utterance

		gs.logger.info(f"current time: {env.curr_time}, ViCo steps: {env.steps}, agents actions: {agent_actions_to_print}")
		dt_agent = time.perf_counter() - lst_time
		env.config["dt_agent"] = (env.config["dt_agent"] * env.steps + dt_agent) / (env.steps + 1)
		lst_time = time.perf_counter()
		obs, _, done, info = env.step(agent_actions)

		dt_sim = time.perf_counter() - lst_time
		env.config["dt_sim"] = (env.config["dt_sim"] * (env.steps - 1) + dt_sim) / env.steps
		gs.logger.info(f"Time used: {dt_agent:.2f}s for agents, {dt_sim:.2f}s for simulation, "
					f"average {env.config['dt_agent']:.2f}s for agents, "
					f"{env.config['dt_sim']:.2f}s for simulation, "
					f"{env.config['dt_chat']:.2f}s for post-chatting over {env.steps} steps.")
		if env.steps > args.max_steps:
			break

	for agent in agents:
		agent.terminate()
	env.close()