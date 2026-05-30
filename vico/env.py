import logging
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygltflib")
import sys
import time
import random
from datetime import datetime, timedelta
from collections import defaultdict
import tqdm

from gymnasium import Env, spaces
import genesis as gs
from genesis.utils.tools import FPSTracker
from genesis.utils.misc import tensor_to_array
from .tools.utils import get_assets_dir
from .tools.asset_utils import get_asset_path
import genesis.utils.geom as geom_utils
from genesis.options import CoacdOptions
from genesis.engine.entities.rigid_entity import RigidEntity
import numpy as np
import json
import string
from PIL import Image

from .tools.constants import ASSETS_PATH, LIGHTS, ENV_OTHER_METADATA
from .tools.utils import *
from .modules import *

from .agents import get_agent_cls as _default_get_agent_cls

class VicoEnv:
	def __init__(self,
				 seed,
				 precision,
				 logging_level,
				 backend=gs.cpu,
				 head_less=False,
				 resolution=512,
				 challenge='full',
				 num_agents=5,
				 config_path='',
				 scene='NY',
				 enable_indoor_scene=False,
				 enable_indoor_objects=False,
				 enable_outdoor_objects=False,
				 outdoor_objects_max_num=10,
				 enable_collision=False,
				 enable_decompose=False,
				 skip_avatar_animation=False,
				 enable_gt_segmentation=False,
				 no_load_scene=False,
				 output_dir='output',
				 enable_third_person_cameras=True,
				 enable_demo_camera=False,
				 no_traffic_manager=False,
				 tm_vehicle_num=0,
				 tm_avatar_num=0,
				 enable_tm_debug=False,
				 save_per_seconds=10,
				 defer_chat=False,
				 debug=False,
				 user_controlled_index=None,
				 dt_sim=0.01,
				 batch_renderer=False,
				 get_agent_cls=None):
		if get_agent_cls is None:
			get_agent_cls = _default_get_agent_cls
		self._get_agent_cls = get_agent_cls
		if not gs._initialized:
			gs.init(seed=seed, precision=precision, logging_level=logging_level, backend=backend)
		fh = logging.FileHandler(os.path.join(output_dir, 'raw.log'))
		fh.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		gs.logger._logger.addHandler(fh)
		self.resolution = resolution
		self.challenge = challenge
		self.num_agents = num_agents
		self.output_dir = output_dir
		self.enable_third_person_cameras = enable_third_person_cameras
		self.enable_demo_camera = enable_demo_camera
		self.save_per_seconds = save_per_seconds
		self.seed = seed
		self.skip_avatar_animation = skip_avatar_animation
		self.defer_chat = defer_chat
		self.debug = debug
		self.enable_indoor_scene = enable_indoor_scene
		if self.enable_indoor_scene:
			self.coarse_indoor_scene = json.load(open(os.path.join(ASSETS_PATH, "coarse_type_to_indoor_scene.json"), 'r'))
		self.active_places_info = {}
		self.active_places_agents = defaultdict(list)
		self.enable_outdoor_objects = enable_outdoor_objects
		self.enable_indoor_objects = enable_indoor_objects
		self.outdoor_objects_max_num = outdoor_objects_max_num
		self.enable_collision = enable_collision
		self.enable_decompose = enable_decompose
		self.enable_gt_segmentation = enable_gt_segmentation
		self.batch_renderer = batch_renderer
		self.scene_name = scene
		self.entity_idx_to_info = defaultdict(dict)
		self.entity_idx_to_color = []
		self.place_cameras = {}
		self.user_controlled_index = int(user_controlled_index) if user_controlled_index is not None else None

		self.config_path = config_path
		self.config = json.load(open(os.path.join(self.config_path, 'config.json'), 'r'))
		if "sps_agent" not in self.config:
			self.config["sps_agent"] = 0
		if "sps_sim" not in self.config:
			self.config["sps_sim"] = 0
		if "sps_chat" not in self.config:
			self.config["sps_chat"] = 0
		if "dt_control" not in self.config:
			self.config["dt_control"] = [1.0] * self.num_agents
		if "dt_rgb_obs" not in self.config:
			self.config["dt_rgb_obs"] = [1.0] * self.num_agents
		robot_agent_id_list = self.config.get("robot_agent_id_list")
		self.enable_robots = bool(robot_agent_id_list)
		self.robot_agent_id_list = robot_agent_id_list or []
		self.agent_control_freq = [int(dt_control / dt_sim) for dt_control in self.config["dt_control"]]
		self.sec_per_step = min(self.agent_control_freq) / 100.
		self.traffic_control_freq = int(1.0 / dt_sim)
		self.agent_visual_obs_freq = [int(dt_visual_obs / dt_sim) for dt_visual_obs in self.config["dt_visual_obs"]]
		self.dt_sim = dt_sim
		self.config["dt_sim"] = dt_sim

		if self.challenge == 'full':
			assert self.num_agents == self.config[
				'num_agents'], f"num_agents in config file is {self.config['num_agents']}, but got {self.num_agents}"

		self.curr_time: datetime = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
		self.steps = self.config['step']
		self.seconds = self.steps * self.sec_per_step
		self.genesis_steps = int(self.config['step'] * (1 / dt_sim))
		self.building_metadata = json.load(open(os.path.join(config_path, "building_metadata.json"), 'r'))
		self.place_metadata = json.load(open(os.path.join(config_path, "place_metadata.json"), 'r'))
		self.transit_info = json.load(open(os.path.join(ASSETS_PATH, "scenes", self.scene_name, "transit.json"), 'r'))
		self.env_other_meta = ENV_OTHER_METADATA
		self.agent_names_to_group_name = {agent_name: group_name for group_name, group in self.config['groups'].items() for agent_name in group['members']} if 'groups' in self.config else {}
		self.events = EventSystem()
		self.obs = {i: {} for i in range(self.num_agents)}
		self.obs['agent_list_to_update'] = [i for i in range(self.num_agents)]

		self.scene_assets_dir = get_asset_path(f"scene/v1/{scene}")
		self.vehicles = []
		self.enable_tm_debug = enable_tm_debug

		initial_lights = self.config.get('initial_lights', LIGHTS)

		self.scene = gs.Scene(
			# viewer_options=None,
			viewer_options=gs.options.ViewerOptions(
				res=(1024, 1024),
				camera_pos=np.array([0.0, 0.0, 1000]),
				camera_lookat=np.array([0, 0.0, 0.0]),
				camera_fov=60,
			),
			rigid_options=gs.options.RigidOptions(
				gravity=(0.0, 0.0, -9.8),
				enable_collision=self.enable_collision,
				max_collision_pairs=400,
				dt=dt_sim
			),
			avatar_options=gs.options.AvatarOptions(
				enable_collision=self.enable_collision,
			),
			renderer=gs.renderers.Rasterizer() if not self.batch_renderer else gs.renderers.BatchRenderer(use_rasterizer=True),
			vis_options=gs.options.VisOptions(
				show_world_frame=False,
				segmentation_level="entity",
				lights=initial_lights
			),
			profiling_options=gs.options.ProfilingOptions(show_FPS=False),
			show_viewer=not head_less,
		)
		if self.batch_renderer:
			for light in initial_lights:
				self.scene.add_light(
					pos=(1.0, 1.0, 1.0),
					dir=light['dir'],
					color=light['color'],
					intensity=light['intensity'],
					directional=True,
				)
		### Load city scene
		start_time = time.time()
		if self.enable_demo_camera:
			self.demo_camera = self.scene.add_camera(
				res=(1024, 1024),
				pos=self.config['demo_camera_pos'],
				lookat=self.config['demo_camera_lookat'],
				fov=90,
				far=16000.0,
				GUI=False,
				debug=True,
			)
			os.makedirs(os.path.join(self.output_dir, 'demo'), exist_ok=True)
		self.terrain = self.load_city_scene(self.scene_assets_dir, no_load_scene)
		self.height_field = load_height_field(
			os.path.join(self.scene_assets_dir, "height_field.npz"))
		gs.logger.info(f"loading city scene took {time.time() - start_time:.2f}s")

		if not no_load_scene and self.terrain is not None:
			no_collision_entities = [self.terrain]
		else:
			no_collision_entities = []

		### Load avatars
		self.agents = []
		self.agent_names: list[str] = self.config['agent_names']
		self.agent_infos: list[dict] = self.config['agent_infos']
		start_time = time.time()
		frame_ratio = 0.0 if skip_avatar_animation else 5.0

		# robot parameters
		self.robots = []
		self.robot_names = []
		self.robot_obs = {}
		self.robots_types = []
		self.robot_actions = []
		self.robot_control_freq = []
		self.robot_step_id = 0
		for i in range(self.num_agents):
			if i in self.robot_agent_id_list:
				# initialize agent as robot
				robot_idx = len(self.robots)
				robot_type = self.config['robot_types'][robot_idx]
				self.config['agent_poses'][i][2] += ROBOT_POSITION_OFFSETS[robot_type][2] # only count z-axis offset
				self.agents.append(self.add_robot(name=self.agent_names[i],
												  robot_type=robot_type,
												  position=np.array(self.config['agent_poses'][i][:3],
																	dtype=np.float64),
												  config_path=os.path.join(ASSETS_PATH, "robot_cfgs", ROBOT_CONFIGS[robot_type]),
												  terrain_height_path=f"{self.scene_assets_dir}/height_field.npz",
												  third_person_camera_resolution=self.resolution if self.enable_third_person_cameras else None))
				self.robot_obs[robot_idx] = self.robots[robot_idx].get_observations()
				self.robot_control_freq.append(self.robots[robot_idx].env_cfg['control_freq'])
			else:
				# initialize agent as avatar
				self.agents.append(self.add_avatar(name=self.agent_names[i],
												   motion_data_path='avatars/motions/motion.pkl',
												   skin_options={
													   'glb_path': self.config['agent_skins'][i],
													   'euler': (-90, 0, 90),
													   'pos': (0.0, 0.0, -0.959008030)
												   },
												   ego_view_options={
													   "res": (self.resolution, self.resolution),
													   "fov": 90,
													   "GUI": False,
												   },
												   frame_ratio=frame_ratio,
												   terrain_height_path=os.path.join(self.scene_assets_dir, "height_field.npz"),
												   third_person_camera_resolution=self.resolution if self.enable_third_person_cameras else None,
												   enable_collision=enable_collision))
				no_collision_entities += [self.agents[i].robot.box]

			if self.enable_third_person_cameras:
				os.makedirs(os.path.join(self.output_dir, 'tp', self.agent_names[i]), exist_ok=True)
			os.makedirs(os.path.join(self.output_dir, 'ego', self.agent_names[i]), exist_ok=True)
			os.makedirs(os.path.join(self.output_dir, 'steps', self.agent_names[i]), exist_ok=True)

		for i, agent in enumerate(self.agents):
			if i not in self.robot_agent_id_list:
				agent.initialize_no_collision(no_collision_entities)
		gs.logger.info(f"loading {self.num_agents} avatars took {time.time() - start_time:.2f}s")

		### Load indoor scenes
		if self.enable_indoor_scene:
			start_time = time.time()
			for place in self.place_metadata: # loading living places
				if self.place_metadata[place]["coarse_type"] == "accommodation" and "'s room" in place:
					self.load_indoor_scene(place)
			for group in self.config['groups'].values():
				self.load_indoor_scene(group['place'])
			if 'stores' in self.config:
				for store in self.config['stores']:
					self.load_indoor_scene(store)
			gs.logger.info(f"loading {len(self.active_places_info)} indoor scenes took {time.time() - start_time:.2f}s")

			# load a default room
			self.active_places_info['default_room'], self.place_cameras['default_room'] = \
				load_default_room(self, "default_room")
		
		for vgeom in self.scene.rigid_solver.vgeoms:
			vgeom.surface.double_sided = True

		if no_traffic_manager:
			tm_vehicle_num = 0
			tm_avatar_num = 0
		self.traffic_manager = TrafficManager(self, scene, vehicle_number=tm_vehicle_num, pedestrian_number=tm_avatar_num, enable_tm_debug=self.enable_tm_debug, logger=gs.logger, debug=debug)

		self.scene.build()
		self.scene.reset()

		self.traffic_manager.init_post_scene_build()

		# forward pass
		if self.enable_gt_segmentation:
			# to make sure color is consistent across runs
			prev_seed = np.random.get_state()
			np.random.seed(42)
			self.entity_idx_to_color = np.random.randint(0, 255, (len(self.entity_idx_to_info) + 1, 3), dtype=np.uint8)
			# background uses black
			self.entity_idx_to_color[0, :] = 0
			np.random.set_state(prev_seed)

		self.sim_frames_per_step = int(self.sec_per_step / self.scene.dt)
		gs.logger.info(f"running {self.sim_frames_per_step} scene steps for one ViCo step of {self.sec_per_step}s")

		self.traffic_manager.reset()

		for i, agent in enumerate(self.agents):
			agent.reset(np.array(self.config['agent_poses'][i][:3], dtype=np.float64),
						geom_utils.euler_to_R(np.degrees(np.array(self.config['agent_poses'][i][3:], dtype=np.float64))))
			if self.config['agent_infos'][i]['current_vehicle'] == 'bus':
				agent.enter_bus(self.traffic_manager.bus.bus)
				gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()}")
			elif self.config['agent_infos'][i]['current_vehicle'] == 'bicycle':
				bike_idx = self.enter_bike(i)
				if bike_idx is None:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter bike because no bike is available. This is not normal!")
				else:
					gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bicycle {bike_idx} at {self.traffic_manager.shared_bicycles.bicycles[bike_idx].get_global_pose().tolist()}")

		frame = 0
		while not all([agent.spare() for agent in self.agents]):
			frame += 1
			self.scene_step()

		gs.logger.info(f"In initialization, prepare all agents with current vehicle for {frame} frames")

		rgb_space = spaces.Box(0, 256,(3, self.resolution, self.resolution), dtype=np.int32)
		depth_space = spaces.Box(0, 256, (self.resolution, self.resolution), dtype=np.float32)
		self.observation_space_single = spaces.Dict({
			"rgb": rgb_space,
			"depth": depth_space,
			"segmentation": spaces.Box(0, 10000, (self.resolution, self.resolution), dtype=np.int32),
			'extrinsics': spaces.Box(-30, 30, (4, 4), dtype=np.float32),
			'pose': spaces.Box(-1400, 1400, (6,), dtype=np.float32),
			'accessible_places': spaces.Sequence(spaces.Text(max_length=1000, charset=string.printable)),
			'action_status': spaces.Text(max_length=1000, charset=string.printable),
			'current_building': spaces.Text(max_length=1000, charset=string.printable),
			'current_place': spaces.Text(max_length=1000, charset=string.printable),
			"cash": spaces.Box(0, 1000, (1,), dtype=np.int32)
		})
		self.observation_space = spaces.Dict({
			i: self.observation_space_single for i in range(self.num_agents)
		})

		self.action_space_single = spaces.Dict({
			'type': spaces.Discrete(7),
			'arg1': spaces.Text(10000),
		})
		self.action_space = spaces.Dict({
			i: self.action_space_single for i in range(self.num_agents)
		})

		self.fps_tracker = FPSTracker(0)
		# compute bounding boxes for each entity
		if len(self.entity_idx_to_info) != len(self.scene.entities):
			gs.logger.error(f"Number of entities in scene {len(self.scene.entities)} does not match number of entities in entity_idx_to_info {len(self.entity_idx_to_info)}")
		for i, e in self.entity_idx_to_info.items():
			rigid = self.scene.entities[i]
			if isinstance(rigid, RigidEntity) and e["type"] == "object":
				mx, mn = np.zeros(3), np.zeros(3)
				for geom in rigid.vgeoms:
					verts = geom._vmesh.verts
					mx = np.maximum(mx, verts.max(axis=0))
					mn = np.minimum(mn, verts.min(axis=0))
				e["bbox"] = np.stack([mn, mx])
		self.idxc_to_info = {idxc: self.entity_idx_to_info[entity_idx] for idxc, entity_idx in self.scene.visualizer.segmentation_idx_dict.items() if entity_idx != -1}

		os.makedirs(os.path.join(self.output_dir, 'steps', 'env'), exist_ok=True)

	def add_entity(self, type, name, morph, material=None,
				   surface=None, visualize_contact=False, vis_mode=None,):
		"""
		:param type: One of "structure", "building", "object", "avatar", "avatar_box", "vehicle"
		:param name:
		:param morph:
		:param material:
		:param surface:
		:param visualize_contact:
		:param vis_mode:
		:return:
		"""
		entity = self.scene.add_entity(morph=morph, material=material, surface=surface, visualize_contact=visualize_contact, vis_mode=vis_mode)
		self.entity_idx_to_info[entity.idx] = {"type": type, "name": name}
		return entity

	def log_step_vehicle_info(self):
		step_info = {
			"curr_bus_pose": self.traffic_manager.bus.bus.get_global_pose().tolist(),
			"curr_bus_stop": self.traffic_manager.bus.current_stop_name,
			"curr_bicycle_poses": [bicycle.get_global_pose().tolist() for bicycle in self.traffic_manager.shared_bicycles.bicycles]
		}
		step_info_path = os.path.join(self.config_path.replace('curr_sim', 'steps'), 'env', f"{self.steps:06d}.json")
		atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))


	def setup_user_controlled_camera(self, agent_id):
		"""Setup initial camera position for user-controlled agent"""
		if not hasattr(self.scene, 'viewer') or self.scene.viewer is None:
			self.logger.warning("No viewer available for camera setup")
			return

		agent = self.agents[agent_id]
		agent_pos = agent.get_global_pose()[:3]

		# Set third-person camera position (behind and above the agent)
		camera_offset = np.array([-5.0, 0.0, 3.0])  # Behind, same Y, above
		camera_pos = agent_pos + camera_offset

		# Set camera to look at the agent
		self.scene.viewer.set_camera_pose(
			pos=camera_pos,
			lookat=agent_pos
		)

	def scene_step(self, avatar_sim_early_end=False):
		self.scene.step()

		if self.enable_demo_camera:
			rgb, _, _, _ = self.demo_camera.render(depth=False)
			Image.fromarray(rgb).save(os.path.join(self.output_dir, 'demo', f"{self.genesis_steps:06d}.png"))

		for robot in self.robots:
			robot.step(actions=None, perform_physics_step=True)

		if avatar_sim_early_end:
			return

		if self.vehicles:
			for vehicle in self.vehicles:
				vehicle.step()
			self.log_step_vehicle_info()

		if self.agents:
			for agent_id, avatar in enumerate(self.agents):
				if agent_id in self.robot_agent_id_list:
					continue
				avatar.step(self.skip_avatar_animation)

		if self.traffic_manager is not None:
			for avatar in self.traffic_manager.avatars:
				avatar.avatar.step()

		if self.agents and self.enable_collision:
			collision_pairs = self.scene.rigid_solver.detect_collision()
			for i, avatar in enumerate(self.agents):
				if i in self.robot_agent_id_list:
					continue
				avatar.post_step(collision_pairs)

	def add_avatar(
			self,
			name: str,
			motion_data_path: str,
			skin_options = None,
			ego_view_options = None,
			frame_ratio = 5.0,
			terrain_height_path = None,
			third_person_camera_resolution = None,
			enable_collision = True,
	):
		avatar = AvatarController(
			env = self,
			motion_data_path = motion_data_path,
			skin_options = skin_options,
			ego_view_options = ego_view_options,
			frame_ratio = frame_ratio,
			terrain_height_path = terrain_height_path,
			third_person_camera_resolution = third_person_camera_resolution,
			enable_collision = enable_collision,
			name=name
		)
		return avatar

	def add_vehicle(
			self,
			name,
			vehicle_asset_path,
			ego_view_options,
			position = np.zeros(3, dtype=np.float64),
			rotation = np.zeros(3, dtype=np.float64),
			dt=1e-2,
			forward_speed_m_per_s=5,
			angular_speed_deg_per_s=360,
			terrain_height_path = None,
	):
		if self.skip_avatar_animation:
			dt = self.sec_per_step
		vehicle = VehicleController(
			env = self,
			name = name,
			vehicle_asset_path = vehicle_asset_path,
			ego_view_options = ego_view_options,
			position = position,
			rotation = rotation,
			dt = dt,
			forward_speed_m_per_s=forward_speed_m_per_s,
			angular_speed_deg_per_s=angular_speed_deg_per_s,
			terrain_height_path=terrain_height_path,
		)
		self.vehicles.append(vehicle)
		return vehicle

	def add_robot(
			self,
			name: str,
			robot_type: str,
			config_path: str,
			position=np.zeros(3, dtype=np.float64),
			terrain_height_path="height_field.npz",
			third_person_camera_resolution=None,
	):
		self.robots_types.append(robot_type)
		controller = ROBOT_CONTROLLERS[robot_type]
		robot = controller(
			env=self,
			scene=self.scene,
			name=name,
			terrain_height_path=terrain_height_path,
			ego_view_options={
				"res": (self.resolution, self.resolution),
				"fov": 90,
				"GUI": False,
			},
			device=gs.backend.name,
			dt=self.config["dt_sim"],
			config_path=config_path,
			position=position,
			third_person_camera_resolution=third_person_camera_resolution,
			debug=False
		)
		robot.robot.action_status = ActionStatus.INIT
		self.robots.append(robot)
		return robot

	def load_city_scene(self, scene_assets_dir, no_load_scene):
		height_field = load_height_field(
			os.path.join(scene_assets_dir, "height_field.npz"))
		ratio = 5
		X = np.arange(-500, 501, ratio)
		Y = np.arange(-500, 501, ratio)
		X, Y = np.meshgrid(X, Y)
		terrain = None
		if self.enable_collision:
			terrain = self.add_entity(
				type="structure",
				name="terrain",
				morph=gs.morphs.Terrain(
					pos=(-500, -500, 0),
					vertical_scale=1.0,
					horizontal_scale=ratio,
					height_field=height_field(Y, X) + 0.2,
					visualization=False
				),
			)
		if no_load_scene:
			return terrain
		self.add_entity(
			type = "structure",
			name = "terrain",
			material=gs.materials.Rigid(
				sdf_min_res=4,
				sdf_max_res=4,
			),
			morph=gs.morphs.Mesh(
				file=os.path.join(scene_assets_dir, 'terrain.glb'),
				euler=(90.0, 0, 0),
				fixed=True,
				collision=False,
				merge_submeshes_for_collision=False,
				group_by_material=True,
			),
		)
		buildings_dir = str(os.path.join(scene_assets_dir, 'buildings'))
		building_glb2name = {}
		for building_name in self.building_metadata:
			if building_name != 'open space':
				building_glb2name[self.building_metadata[building_name]['building_glb']] = building_name
		if os.path.exists(buildings_dir):
			for building in os.listdir(buildings_dir):
				if building.endswith('.glb'):
					if building in building_glb2name:
						self.add_entity(
							type="building",
							name=building_glb2name[building],
							material=gs.materials.Rigid(
								sdf_min_res=4,
								sdf_max_res=4,
							),
							morph=gs.morphs.Mesh(
								file=os.path.join(scene_assets_dir, 'buildings', building),
								euler=(90.0, 0, 0),
								fixed=True,
								collision=self.enable_collision,
								merge_submeshes_for_collision=False,
								group_by_material=True,
								decompose_object_error_threshold=float("inf") if not self.enable_decompose else 0.15,
								convexify=self.enable_collision,
								coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
							),
						)
					else:
						building_name = building.strip('.glb')
						self.add_entity(
							type="structure",
							name=building_name,
							material=gs.materials.Rigid(
								sdf_min_res=4,
								sdf_max_res=4,
							),
							morph=gs.morphs.Mesh(
								file=os.path.join(scene_assets_dir, 'buildings', building),
								euler=(90.0, 0, 0),
								fixed=True,
								collision=self.enable_collision,
								merge_submeshes_for_collision=False,
								group_by_material=True,
								decompose_object_error_threshold=float("inf") if not self.enable_decompose else 0.15,
								convexify=self.enable_collision,
								coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
							),
						)
		else:
			self.add_entity(
				type= "structure",
				name= "buildings",
				material=gs.materials.Rigid(
					sdf_min_res=4,
					sdf_max_res=4,
				),
				morph=gs.morphs.Mesh(
					file=os.path.join(scene_assets_dir, 'buildings.glb'),
					euler=(90.0, 0, 0),
					fixed=True,
					collision=self.enable_collision,
					merge_submeshes_for_collision=False,  # Buildings are constructed separately
					group_by_material=True,
					convexify=self.enable_collision,
					decompose_object_error_threshold=float("inf") if not self.enable_decompose else 0.15,
					coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
				),
			)

		# self.add_entity(
		# 	type="structure",
		# 	name="roof",
		# 	material=gs.materials.Rigid(
		# 		sdf_min_res=4,
		# 		sdf_max_res=4,
		# 	),
		# 	morph=gs.morphs.Mesh(
		# 		file=os.path.join(scene_assets_dir, 'roof.glb'),
		# 		euler=(90.0, 0, 0),
		# 		fixed=True,
		# 		collision=False,  # No collision needed for roof
		# 		group_by_material=True,
		# 	),
		# )

		if self.enable_outdoor_objects:
			outdoor_object_context = OutdoorObjectContext(
				scene_name=self.scene_name,
				objects_cfg_dir=os.path.join(scene_assets_dir, 'objects'),
				assets_dir='objects/outdoor_objects',
				max_objects=self.outdoor_objects_max_num,
				seed=self.seed,
				terrain_height_field_path=f"{scene_assets_dir}/height_field.npz",
				road_info_path=os.path.join(ASSETS_PATH, "scenes", self.scene_name, "road_data", "roads.pkl"),
			)
			load_outdoor_objects(self, outdoor_object_context, self.transit_info)

		return terrain

	def enter_bike(self, agent_idx):
		nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_nearest_bicycle(self.agents[agent_idx].get_global_xy())
		if nearest_bicycle is not None:
			self.agents[agent_idx].enter_bike(0, nearest_bicycle)
			self.traffic_manager.shared_bicycles.start_timer(nearest_bicycle_idx, self.curr_time)
			self.agent_infos[agent_idx]["current_vehicle"] = "bicycle"
		return nearest_bicycle_idx

	def exit_bike(self, agent_idx):
		if self.agent_infos[agent_idx]["current_vehicle"] != "bicycle":
			gs.logger.warning(f"Agent {self.agent_names[agent_idx]} cannot exit bike because current vehicle is not bike.")
			return False
		nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_riding_bicycle(self.agents[agent_idx].get_global_xy())
		cost = self.traffic_manager.shared_bicycles.end_timer(nearest_bicycle_idx, self.curr_time)
		self.agent_infos[agent_idx]["cash"] -= cost
		self.agent_infos[agent_idx]["current_vehicle"] = None
		self.agents[agent_idx].exit_bike(0)
		return True

	def perform_action(self, agent_id, action, is_robot=False):
		if action is None:
			return
		agent = self.agents[agent_id]
		agent.robot.action_status = ActionStatus.SUCCEED

		# converse and enter actions work for both robot and avatars
		if action['type'] == 'converse':
			if agent.robot.base_state == AvatarState.SLEEPING:
				agent.robot.base_state = AvatarState.STANDING
			agent_pos = self.config['agent_poses'][agent_id][:3]
			converse_range = action['arg2'] if 'arg2' in action else 10
			priority = random.randint(0, 100)
			if converse_range > 10:
				gs.logger.warning(
					f"Agent {self.agent_names[agent_id]} attempted to converse with range {converse_range} which is larger than 10. Ignored.")
				agent.robot.action_status = ActionStatus.FAIL
				return
			deleted_subjects = self.events.add(type="speech", pos=agent_pos, r=converse_range, content=action['arg1'],
											   priority=priority, subject=self.agent_names[agent_id], predicate="is",
											   object="talk")
			# if interleaved with other speech events, keep only the highest priority one, drop others and give it fail
			for deleted_subject in deleted_subjects:
				self.agents[self.agent_names.index(deleted_subject)].robot.action_status = ActionStatus.FAIL
			return
		elif action['type'] == 'enter' or action['type'] == 'force_enter':
			if action['type'] == 'force_enter' or action['arg1'] in self.obs[agent_id]['accessible_places']:
				if action['arg1'] == 'open space':
					if self.agent_infos[agent_id]["current_place"] in self.active_places_agents:
						try:
							self.active_places_agents[self.agent_infos[agent_id]["current_place"]].remove(self.agent_names[agent_id])
						except ValueError:
							gs.logger.warning(
								f"Agent {self.agent_names[agent_id]} is not among the names that are in place {self.agent_infos[agent_id]['current_place']}.")
					if is_robot:
						pos = np.array(self.agent_infos[agent_id]["outdoor_pose"][:3])
						robot_type = self.robots_types[self.robot_agent_id_list.index(agent_id)]
						pos[2] += ROBOT_POSITION_OFFSETS[robot_type][2]
						agent.reset(pos, geom_utils.euler_to_R(np.degrees(np.array(self.agent_infos[agent_id]["outdoor_pose"][3:], dtype=np.float64))))
					else:
						agent.reset(np.array(self.agent_infos[agent_id]["outdoor_pose"][:3]), geom_utils.euler_to_R(
							np.degrees(np.array(self.agent_infos[agent_id]["outdoor_pose"][3:], dtype=np.float64))))
					self.agent_infos[agent_id]["current_building"] = 'open space'
					self.agent_infos[agent_id]["current_place"] = None
				else:
					if self.place_metadata[action['arg1']]['building'] == 'open space':
						return
					if self.agent_infos[agent_id]["current_building"] == 'open space':
						self.agent_infos[agent_id]["outdoor_pose"] = self.config['agent_poses'][agent_id]
					self.load_indoor_scene(action['arg1'])  # load new scenes should be wrong now
					place_info = self.active_places_info.get(action['arg1'])
					if place_info is not None and "init_avatar_poses" in place_info:
						pos = place_info["init_avatar_poses"][0]["pos"]
						euler = place_info["init_avatar_poses"][0]["euler"]
						x, y, z = self.place_metadata[action['arg1']]['location']
						pos = np.array([pos[0] + x, pos[1] + y, z])
						agent.reset(pos, geom_utils.euler_to_R(np.degrees(np.array(euler, dtype=np.float64))))
					else:
						offset = len(self.active_places_agents[action['arg1']]) * 0.2
						self.active_places_agents[action['arg1']].append(self.agent_names[agent_id])
						x, y, z = self.place_metadata[action['arg1']]['location']
						agent.reset(np.array([x + offset, y + offset, z]),
									geom_utils.euler_to_R(np.degrees(np.array([0, 0, 0]))))  # elevator entrance
					self.agent_infos[agent_id]["current_building"] = self.place_metadata[action['arg1']]['building']
					self.agent_infos[agent_id]["current_place"] = action['arg1']
			else:
				gs.logger.warning(
					f"Agent {self.agent_names[agent_id]} cannot enter {action['arg1']} because it is not in accessible places.")
				agent.robot.action_status = ActionStatus.FAIL
			return

		# robot actions
		if is_robot:
			assert action['type'] == 'control', f"Robot action {action['type']} is not supported."
			agent.perform_control(action)
			return

		# avatar actions
		if action['type'] == 'move_forward':
			agent.move_forward(action['arg1'], self.sec_per_step * 1.0)
		elif action['type'] == 'teleport':
			agent.reset_with_global_xy(np.array(action['arg1']))
		elif action['type'] == 'turn_left':
			agent.turn_left(action['arg1'], turn_sec_limit=self.sec_per_step * 1500)
		elif action['type'] == 'turn_right':
			agent.turn_right(action['arg1'], turn_sec_limit=self.sec_per_step * 1500)
		elif action['type'] == 'look_at':
			target_pos = action['arg1']
			## make avatar look at target_pos by turn_left or turn_right
			agent_pos = agent.robot.global_trans
			agent_rot = agent.robot.global_rot
			agent_dir = agent_rot[:, 0]
			target_dir = target_pos - agent_pos
			agent_dir[2] = 0
			target_dir[2] = 0
			agent_dir = agent_dir / np.linalg.norm(agent_dir)
			target_dir = target_dir / np.linalg.norm(target_dir)
			cross = np.cross(agent_dir, target_dir)
			dot = np.dot(agent_dir, target_dir)
			angle = np.arccos(dot)
			if cross[2] > 0:
				agent.turn_left(angle, turn_sec_limit=self.sec_per_step * 1500)
			else:
				agent.turn_right(angle, turn_sec_limit=self.sec_per_step * 1500)
		elif action['type'] == 'sleep':
			agent.sleep()
		elif action['type'] == 'wake':
			agent.wake()
		elif action['type'] == 'enter' or action['type'] == 'force_enter':
			if action['type'] == 'force_enter' or action['arg1'] in self.obs[agent_id]['accessible_places']:
				if action['arg1'] == 'open space':
					if self.agent_infos[agent_id]["current_place"] in self.active_places_agents:
						try:
							self.active_places_agents[self.agent_infos[agent_id]["current_place"]].remove(self.agent_names[agent_id])
						except ValueError:
							gs.logger.warning(
								f"Agent {self.agent_names[agent_id]} is not among the names that are in place {self.agent_infos[agent_id]['current_place']}.")
					agent.reset(np.array(self.agent_infos[agent_id]["outdoor_pose"][:3]), geom_utils.euler_to_R(
						np.degrees(np.array(self.agent_infos[agent_id]["outdoor_pose"][3:], dtype=np.float64))))
					self.agent_infos[agent_id]["current_building"] = 'open space'
					self.agent_infos[agent_id]["current_place"] = None
				else:
					if self.place_metadata[action['arg1']]['building'] == 'open space':
						return
					if self.agent_infos[agent_id]["current_building"] == 'open space':
						self.agent_infos[agent_id]["outdoor_pose"] = self.config['agent_poses'][agent_id]
					self.load_indoor_scene(action['arg1'])  # load new scenes should be wrong now
					place_info = self.active_places_info.get(action['arg1'])
					if place_info is not None and "init_avatar_poses" in place_info:
						pos = place_info["init_avatar_poses"][0]["pos"]
						euler = place_info["init_avatar_poses"][0]["euler"]
						x, y, z = self.place_metadata[action['arg1']]['location']
						pos = np.array([pos[0] + x, pos[1] + y, z])
						agent.reset(pos, geom_utils.euler_to_R(np.degrees(np.array(euler, dtype=np.float64))))
					else:
						offset = len(self.active_places_agents[action['arg1']]) * 0.2
						self.active_places_agents[action['arg1']].append(self.agent_names[agent_id])
						x, y, z = self.place_metadata[action['arg1']]['location']
						agent.reset(np.array([x + offset, y + offset, z]),
									geom_utils.euler_to_R(np.degrees(np.array([0, 0, 0]))))  # elevator entrance
					self.agent_infos[agent_id]["current_building"] = self.place_metadata[action['arg1']]['building']
					self.agent_infos[agent_id]["current_place"] = action['arg1']

			else:
				gs.logger.warning(
					f"Agent {self.agent_names[agent_id]} cannot enter {action['arg1']} because it is not in accessible places.")
				agent.robot.action_status = ActionStatus.FAIL
		elif action['type'] == 'enter_bus':
			if "bus" not in self.obs[agent_id]['accessible_places']:
				gs.logger.warning(
					f"Agent {self.agent_names[agent_id]} cannot enter bus because bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()} is not in accessible places.")
				agent.robot.action_status = ActionStatus.FAIL
			agent.enter_bus(self.traffic_manager.bus.bus)
			self.agent_infos[agent_id]["current_vehicle"] = "bus"
		# self.agent_infos[agent_id]["cash"] -= self.traffic_manager.transit_system_info["bus"][0]["price"]
		elif action['type'] == 'exit_bus':
			if self.agent_infos[agent_id]["current_vehicle"] != "bus":
				gs.logger.warning(f"Agent {self.agent_names[agent_id]} cannot exit bus because current vehicle is not bus.")
				agent.robot.action_status = ActionStatus.FAIL
			agent.exit_bus()
			self.agent_infos[agent_id]["current_vehicle"] = None
		elif action['type'] == 'enter_bike':
			if not self.enter_bike(agent_id):
				agent.robot.action_status = ActionStatus.FAIL
		elif action['type'] == 'exit_bike':
			if not self.exit_bike(agent_id):
				agent.robot.action_status = ActionStatus.FAIL
		elif action['type'] == 'pick':  # arg1: hand id [0,1], arg2: position
			if type(action['arg2']) == list:
				if len(action['arg2']) == 3:
					pos = np.array(action['arg2'])
				else:
					gs.logger.warning(f"Action {action['type']} arg2 must be a list of 3 elements, but got {len(action['arg2'])}. Fall back to no react.")
					agent.robot.action_status = ActionStatus.FAIL
					return
				pos = np.array(action['arg2'])
			elif type(action['arg2']) == np.ndarray:
				pos = action['arg2']
			else:
				gs.logger.warning(f"Action {action['type']} arg2 must be a list of 3 elements or a numpy array, but got {type(action['arg2'])}. Fall back to no react.")
				agent.robot.action_status = ActionStatus.FAIL
				return
			min_volume, entity_idx = 1e10, None
			for j, e in self.entity_idx_to_info.items():
				if "bbox" in e:
					bbox = e["bbox"]
					rigid: RigidEntity = self.entities[j]
					rel_pos = pos - rigid.get_pos().cpu().numpy()
					if np.all(rel_pos > bbox[0] - 0.02) and np.all(rel_pos < bbox[1] + 0.02):
						volume = np.prod(bbox[1] - bbox[0])
						if volume < min_volume:
							min_volume, entity_idx = volume, j

			if entity_idx is None:
				gs.logger.warning(f"Agent {self.agent_names[agent_id]} cannot pick at {pos} because no entity is found.")
				agent.robot.action_status = ActionStatus.FAIL
				return
			self.agent_infos[agent_id]["held_objects"][action['arg1']] = self.entity_idx_to_info[entity_idx]["name"]
			agent.pick(action['arg1'], self.entities[entity_idx])
		elif action['type'] == 'put':  # arg1: hand id [0,1]
			agent.put(action['arg1'], action.get('arg2', None))
		elif action['type'] == 'stand':
			agent.stand_up()
		elif action['type'] == 'sit':
			agent.sit(position=np.array(action['arg1'][0]))
		elif action['type'] == 'drink':
			agent.drink(action['arg1'])
		elif action['type'] == 'eat':
			agent.eat(action['arg1'])
		elif action['type'] == 'exchange':  # arg1: target agent name, arg2: amount
			if agent.robot.base_state == AvatarState.SLEEPING:
				agent.robot.base_state = AvatarState.STANDING
			if action['arg2'] > self.agent_infos[agent_id]["cash"]:
				gs.logger.warning(
					f"Agent {self.agent_names[agent_id]} cannot exchange {action['arg2']} cash with {action['arg1']} because it does not have enough cash.")
				agent.robot.action_status = ActionStatus.FAIL
				return
			target_agent_idx = self.agent_names.index(action['arg1'])
			self.agent_infos[agent_id]["cash"] -= action['arg2']
			self.agent_infos[target_agent_idx]["cash"] += action['arg2']
		elif action['type'] == 'play_animation':
			agent.play_animation(name=action['arg1'])
		elif action['type'] == 'wait':
			return
		else:
			raise NotImplementedError(f"agent action type {action['type']} is not supported")

	def post_generate_chat(self, agent_actions):
		# post-generate utterances for remained speech events
		start_time = time.perf_counter()
		for idx, event in self.events.events.items():
			if event["type"] == "speech":
				agent_id = self.agent_names.index(event["subject"])
				agent_actions[agent_id]['request_chat_func'](event["content"])
		to_delete_id = []
		for idx, event in self.events.events.items():
			if event["type"] == "speech":
				agent_id = self.agent_names.index(event["subject"])
				event["content"] = agent_actions[agent_id]['get_utterance_func'](self.steps)
				if event["content"] is None:
					to_delete_id.append(idx)
		self.events.delete(to_delete_id)

		sps_chat = time.perf_counter() - start_time
		self.config["sps_chat"] = (self.config["sps_chat"] * self.steps + sps_chat) / (self.steps + 1)

	def check_avatar_sim_early_end(self):
		if self.skip_avatar_animation:
			return True
		if all([agent.spare() for agent in self.agents]) and \
				(self.traffic_manager is None or self.traffic_manager.spare()):
			return True
		return False

	def update_config(self):
		self.config['step'] = self.steps
		self.config['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		self.config['agent_poses'] = []
		for i, agent in enumerate(self.agents):
			self.config['agent_poses'].append(agent.get_global_pose().tolist())
		self.config['agent_infos'] = self.agent_infos
		for i in range(0, len(self.config["bicycle_poses"])):
			self.config["bicycle_poses"][i] = self.traffic_manager.shared_bicycles.bicycles[
				i].get_global_pose().tolist()
		atomic_save(os.path.join(self.config_path, 'config.json'),
					json.dumps(self.config, indent=4, default=json_converter))

	def calculate_simulation_step(self):
		simulate_to_genesis_step = self.genesis_steps
		return_obs_with_agent_idx = []
		while len(return_obs_with_agent_idx) == 0:
			simulate_to_genesis_step += 1
			for idx, freq in enumerate(self.agent_control_freq):
				if simulate_to_genesis_step % freq == 0:
					return_obs_with_agent_idx.append(idx)
		return simulate_to_genesis_step, return_obs_with_agent_idx

	def step(self, agent_actions):
		simulate_to_genesis_step, agent_idx_to_return = self.calculate_simulation_step()
		agent_list_to_update = agent_actions.pop('agent_list_to_update')
		for i, agent in enumerate(self.agents):
			if i not in agent_list_to_update:
				continue
			action = agent_actions[i]
			self.perform_action(i, action, is_robot=i in self.robot_agent_id_list)
		if self.defer_chat:
			self.post_generate_chat(agent_actions)
		avatar_sim_early_end = False
		for _ in tqdm.tqdm(range(simulate_to_genesis_step - self.genesis_steps), desc="simulating", ):
			self.scene_step(avatar_sim_early_end=avatar_sim_early_end)
			self.genesis_steps += 1
			if not avatar_sim_early_end and self.check_avatar_sim_early_end():
				avatar_sim_early_end = True
				gs.logger.info(f"After {_ + 1} frames, all agents finished action, end avatar simulation early.")
				if not self.enable_robots:
					self.genesis_steps = simulate_to_genesis_step
					break

		self.steps += 1
		self.seconds += self.sec_per_step
		self.curr_time += timedelta(seconds=self.sec_per_step)

		self.update_config()
		self.get_obs(agent_idx_to_return)

		if self.genesis_steps % self.traffic_control_freq:
			if self.traffic_manager is not None:
				self.traffic_manager.step()
			self.traffic_manager.bus.step(self.curr_time)
			if avatar_sim_early_end and not self.traffic_manager.bus.stop_at_this_step:
				bus_next_pose = self.traffic_manager.bus.update_at_time(self.curr_time)
				self.traffic_manager.bus.reset(np.array(bus_next_pose[:3], dtype=np.float64), geom_utils.euler_to_R(
					np.degrees(np.array(bus_next_pose[3:], dtype=np.float64))))
				# to update the position of agents in bus
				self.scene_step(avatar_sim_early_end=avatar_sim_early_end)
			self.events.clear()
		self.fps_tracker.step()
		return self.obs, 0, False, {}

	def reset(self):
		self.scene.reset()
		self.traffic_manager.reset()
		for i, agent in enumerate(self.agents):
			agent.reset(np.array(self.config['agent_poses'][i][:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(self.config['agent_poses'][i][3:], dtype=np.float64))))
			if self.config['agent_infos'][i]['current_vehicle'] == 'bus':
				agent.enter_bus(self.traffic_manager.bus.bus)
				gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()}")
			elif self.config['agent_infos'][i]['current_vehicle'] == 'bicycle':
				bike_idx = self.enter_bike(i) # how to resume previous cost?
				if bike_idx is None:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter bike because no bike is available. This is not normal!")
				else:
					gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bicycle {bike_idx} at {self.traffic_manager.shared_bicycles.bicycles[bike_idx].get_global_pose().tolist()}")

		self.scene_step()
		self.steps = self.config['step']
		self.seconds = self.steps * self.sec_per_step
		self.curr_time = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
		self.get_obs()
		self.fps_tracker = FPSTracker(0)
		return self.obs

	def set_curr_time(self, curr_time):
		gs.logger.warning(f"Set current time to {curr_time}")
		self.curr_time = curr_time
		self.config['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		atomic_save(os.path.join(self.config_path, 'config.json'), json.dumps(self.config, indent=4, default=json_converter))
		# bus_next_pose = self.traffic_manager.bus.update_at_time(self.curr_time)
		# self.traffic_manager.bus.reset(np.array(bus_next_pose[:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(bus_next_pose[3:], dtype=np.float64))))
		self.scene_step()
		self.get_obs()
		return self.obs

	def get_obs(self, agent_idx_list=None):
		if self.batch_renderer:
			self.rgbs, self.depths, self.segmentations, self.normals = self.scene.render_all_cameras(rgb=True, depth=True, segmentation=self.enable_gt_segmentation, normal=True)
			self.rgbs = [tensor_to_array(rgb) for rgb in self.rgbs]
			self.depths = [tensor_to_array(depth) for depth in self.depths]
			self.segmentations = [tensor_to_array(seg) for seg in self.segmentations]
			self.normals = [tensor_to_array(normal) for normal in self.normals]
		if agent_idx_list is not None:
			self.obs['agent_list_to_update'] = agent_idx_list
		else:
			self.obs['agent_list_to_update'] = [i for i in range(self.num_agents)]
		
		if self.traffic_manager is not None and self.enable_tm_debug and self.seconds % self.save_per_seconds == 0:
			for i, avatar in enumerate(self.traffic_manager.avatars):
				if avatar.avatar.ego_view is not None:
					third_person_rgb = avatar.avatar.get_third_person_camera_rgb()
					Image.fromarray(third_person_rgb).save(os.path.join(self.output_dir, 'traffic_ego', avatar.name, f"rgb_avatar{str(i)}_{avatar.name}_{self.steps:06d}.png"))
			for i, vehicle in enumerate(self.traffic_manager.vehicles):
				if vehicle.vehicle.ego_view is not None:
					if self.batch_renderer:
						rgb = self.rgbs[vehicle.vehicle.ego_view.idx]
					else:
						rgb, _, _, _, _ = vehicle.vehicle.render_ego_view(depth=True, segmentation=self.enable_gt_segmentation)
					Image.fromarray(rgb).save(os.path.join(self.output_dir, 'traffic_ego', vehicle.vehicle.name, f"rgb_vehicle{str(i)}_{vehicle.vehicle.name}_{self.steps:06d}.png"))

		for i, agent in enumerate(self.agents):
			self.obs[i]['rgb'] = None
			if agent_idx_list is not None and i not in agent_idx_list:
				continue
			if self.enable_third_person_cameras and self.seconds % self.save_per_seconds == 0:
				indoor = self.agent_infos[i]['current_building'] != 'open space'
				if self.batch_renderer:
					third_person_rgb = self.rgbs[agent.third_person_camera.idx]
				else:
					third_person_rgb = agent.get_third_person_camera_rgb(indoor)
				if third_person_rgb is not None:
					Image.fromarray(third_person_rgb).save(
						os.path.join(self.output_dir, 'tp', self.agent_names[i], f"rgb_{self.steps:06d}.png"))
			self.obs[i]['pose'] = self.config['agent_poses'][i]

			if self.genesis_steps % self.agent_visual_obs_freq[i] == 0:
				if self.batch_renderer:
					self.obs[i]['rgb'], self.obs[i]['depth'], self.obs[i]['segmentation'], self.obs[i]['extrinsics'] = self.rgbs[agent.ego_view.idx], self.depths[agent.ego_view.idx], self.segmentations[agent.ego_view.idx], agent.ego_view.extrinsics
				else:
					self.obs[i]['rgb'], self.obs[i]['depth'], self.obs[i]['segmentation'], self.obs[i]['extrinsics'] = agent.render_ego_view(depth=True, segmentation=self.enable_gt_segmentation)
				if self.seconds % self.save_per_seconds == 0:
					if self.obs[i]['rgb'] is not None:
						Image.fromarray(self.obs[i]['rgb']).save(
							os.path.join(self.output_dir, 'ego', self.agent_names[i], f"rgb_{self.steps:06d}.png"))
					if self.debug:
						if self.obs[i]['segmentation'] is not None:
							Image.fromarray(self.entity_idx_to_color[self.obs[i]['segmentation'] + 1]).save(
								os.path.join(self.output_dir, 'ego', self.agent_names[i], f"seg_{self.steps:06d}.png"))
				if self.enable_gt_segmentation:
					self.obs[i]["gt_seg_idxc_to_info"] = self.idxc_to_info

				self.obs[i]['events'] = self.events.get(
					ref_pos=(self.obs[i]['pose'][:3])) if self.challenge != 'commute' else []

				if not self.enable_gt_segmentation:
					for event in self.obs[i]['events']:
						event['subject'] = None
						event['predicate'] = None
						event['object'] = None
				self.obs[i]['curr_time'] = self.curr_time
				self.obs[i]['steps'] = self.steps
				self.obs[i]['accessible_places'] = get_accessible_places(self.building_metadata, self.place_metadata,
																		 agent.get_global_pose(),
																		 self.agent_infos[i]["current_building"],
																		 self.agent_infos[i]["current_place"])
				self.obs[i]['action_status'] = agent.action_status().value
				self.obs[i]['current_building'] = self.agent_infos[i]["current_building"]
				self.obs[i]['current_place'] = self.agent_infos[i]["current_place"]
				self.obs[i]['cash'] = self.agent_infos[i]["cash"]
				self.obs[i]['held_objects'] = self.agent_infos[i]["held_objects"] if "held_objects" in self.agent_infos[
					i] else [None, None]
				self.obs[i]['current_vehicle'] = self.agent_infos[i]["current_vehicle"]

				if self.obs[i]['current_vehicle'] == 'bus':
					if self.traffic_manager.bus.stop_at_this_step:
						self.obs[i]['accessible_places'] = [self.traffic_manager.bus.current_stop_name]
					else:
						self.obs[i]['accessible_places'] = []
				elif self.obs[i]['current_vehicle'] == 'bicycle':
					# remove the indoor places from accessible places
					self.obs[i]['accessible_places'] = [place for place in self.obs[i]['accessible_places'] if
														self.place_metadata[place]['building'] == 'open space']
				else:
					if self.traffic_manager.bus.stop_at_this_step and is_near_goal(self.obs[i]['pose'][0],
																				   self.obs[i]['pose'][1], None,
																				   self.traffic_manager.bus.bus.get_global_pose().tolist()[
																				   :2], threshold=8):
						self.obs[i]['accessible_places'].append('bus')
					nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_nearest_bicycle(
						agent.get_global_xy())
					if nearest_bicycle is not None:
						self.obs[i]['accessible_places'].append('bicycle')

			if i in self.robot_agent_id_list:
				self.obs[i]['robot_t'] = self.genesis_steps * self.config["dt_sim"]
				self.obs[i]['robot_obs'], self.obs[i]['robot_additional_obs'] = self.agents[i].get_observations()
		self.events.clear()

	def close(self):
		gs.logger.warning("Close ViCo environment")
		import gc
		self.scene = None
		gc.collect()
		gs.destroy()

	def load_indoor_scene(self, place_name):
		if not self.enable_indoor_scene or not place_name:
			return

		place = self.place_metadata.get(place_name)
		if not place or 'scene' not in place:
			gs.logger.error(f"Place {place_name} is not in the metadata or missing 'scene'. Skipping loading.")
			return

		if place.get('building') == 'open space' or not place.get('scene'):
			return

		if place_name in self.active_places_info:
			gs.logger.info(f"Place {place_name} is already loaded or tried to load. Skipping loading.")
			return

		gs.logger.info(f"Loading indoor scene for place {place_name}")
		
		if self.scene.is_built:
			gs.logger.warning("Scene is already built. Redirect to the default scene.")
			self.active_places_info[place_name] = self.active_places_info["default_room"]
			self.place_cameras[place_name] = self.place_cameras["default_room"]
			place['location'] = [1500, 1500, -50]
			return
		self.active_places_info[place_name], self.place_cameras[place_name] = \
				load_indoor_room(self, place['scene'], place['location'], place_name, self.enable_indoor_objects)

	@property
	def entities(self):
		"""All the entities in the scene."""
		return self.scene.entities


