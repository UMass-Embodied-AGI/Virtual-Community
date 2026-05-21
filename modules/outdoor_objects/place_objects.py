import os
import json
import numpy as np
import math
from typing import Optional

import genesis as gs

from .context import OutdoorObjectContext
from .context import compose_euler
from tools.utils import *

def place_mesh(
	env,
	name: str,
	path: str,
	location: np.ndarray = np.zeros(3),
	rotation: np.ndarray = np.zeros(3),
	scale: float = 1.0,
	collision: bool = False,
	fixed: bool = True,
	surface = None,
	visualize_contact: bool = False,
	vis_mode: Optional[str] = None,
):
	rotation = compose_euler(rotation, np.array([90.0, 0, 0]), unit='deg')
	if not os.path.isabs(path):
		path = os.path.join(get_assets_dir(), path)

	x = env.add_entity(
		type='object',
		name=name,
		material=gs.materials.Rigid(),
		morph=gs.morphs.Mesh(
			file=path,
			scale=scale,
			pos=(location[0], location[1], location[2]),
			euler=(rotation[0], rotation[1], rotation[2]),
			collision=collision,
			fixed=fixed
		),
		surface=surface,
		visualize_contact=visualize_contact,
		vis_mode=vis_mode,
	)

	if not hasattr(env, 'debug_positions'):
		env.debug_positions = []
	env.debug_positions.append(location)

	return x.idx

def place_urdf(
	env,
	name: str,
	path: str,
	location: np.ndarray = np.zeros(3),
	rotation: np.ndarray = np.zeros(3),
	scale: float = 1.0,
	collision: bool = True,
	fixed: bool = False,
	surface = None,
	visualize_contact: bool = False,
	vis_mode: Optional[str] = None,
):
	# rotation = np.array([0.0, 0, 0])

	x = env.add_entity(
		type='object',
		name=name,
		material=gs.materials.Rigid(),
		morph=gs.morphs.URDF(
			file=path,
			scale=scale,
			pos=(location[0], location[1], location[2]),
			euler=(rotation[0], rotation[1], rotation[2]),
			collision=collision,
			fixed=fixed
		),
		surface=surface,
		visualize_contact=visualize_contact,
		vis_mode=vis_mode,
	)

	return x.idx

def place_object(
	env, ctx,
	obj_path: str,
	obj_position: np.ndarray = np.zeros(2), # NOTE: 2D position
	obj_rotation: np.ndarray = np.zeros(3),
	obj_scale: float = 1.0,
	obj_tags: dict[str, str] = {},
	obj_name: Optional[str] = None,
):
	'''
	Place an object in the env.
	Use special tags to control the placement of the object.
	Note that we use 2D xy position for the object.
	'''
	obj_info = {
		'path': obj_path,
		'location': ctx.append_position_with_height(obj_position),
		'rotation': obj_rotation,
		'scale': obj_scale,
		'tags': obj_tags,
		'name': obj_name,
	}
	# align with nearby infrastructure (e.g. road, sidewalk, etc.)
	if 'align' in obj_info['tags']:
		align_params = obj_info['tags']['align']
		obj_info = ctx.align_with_road(
			obj_info,
			distance=align_params.get('distance', (0.0, math.inf)),
			angle=align_params.get('angle', (-math.pi, math.pi)),
		)
	if 'rescale' in obj_info['tags']:
		rescale_params = obj_info['tags']['rescale']
		obj_info = ctx.rescale_object(
			obj_info,
			height=rescale_params.get('height', (0.0, math.inf))
		)

	if obj_path.endswith('.urdf'):
		place_urdf(env, obj_info['name'], obj_info['path'], obj_info['location'], obj_info['rotation'], obj_info['scale'])
	else:
		place_mesh(env, obj_info['name'], obj_info['path'], obj_info['location'], obj_info['rotation'], obj_info['scale'])

def load_outdoor_objects(
	env,
	ctx: OutdoorObjectContext,
	transit_info: dict
):
	"""
	Load objects from the given object config directory and place them in the env.

	Args:
		env: VicoEnv
		ctx: OutdoorObjectContext
			Context for placing objects in outdoor scenes
	"""

	meta_cfg_path = os.path.join(ctx.objects_cfg_dir, 'meta.json')
	assert os.path.exists(meta_cfg_path), f'Meta config file does not exist: {meta_cfg_path}'
	with open(meta_cfg_path, 'r') as f:
		meta_cfg = json.load(f)

	groups = meta_cfg['groups']
	groups.pop('bus_stops', None)
	groups.pop('bikes', None)
	for group_id, group_cfg_path in groups.items():
		group_cfg_path = os.path.join(ctx.objects_cfg_dir, group_cfg_path)
		assert os.path.exists(group_cfg_path), f'Object group config file does not exist: {group_cfg_path}'
		with open(group_cfg_path, 'r') as f:
			group = json.load(f)
			objs_cfg = group['objects']

		objs_cfg = ctx.sample_objects(objs_cfg, group_id)
		for obj in objs_cfg:
			obj_position = obj.get('location', np.zeros(3))[:2]
			obj_rotation = obj.get('rotation', np.zeros(3))
			obj_scale = obj.get('scale', 1.0)
			obj_path = os.path.join(ctx.assets_dir, obj['asset_path'])
			obj_tags = obj.get('tags', {})
			obj_name = obj_tags.get('name', group_id)

			place_object(env, ctx, obj_path, obj_position, obj_rotation, obj_scale, obj_tags, obj_name)

	# Process transit objects (bus stops and bicycle stations)
	bus_stops = transit_info["bus"]["stops"]
	bicycle_stations = transit_info["bicycle"]["stations"]
	terrain_height_field = ctx.terrain_height_field

	#* Special handling for bus stops and bicycle stations
	for bus_stop_name in bus_stops.keys():
		bus_stop = bus_stops[bus_stop_name]
		obj_location = np.array([bus_stop["position"][0], bus_stop["position"][1], get_height_at(terrain_height_field, bus_stop["position"][0], bus_stop["position"][1])])
		obj_rotation = np.array([0.0, 0.0, bus_stop["target_rad"]])
		obj_scale = 1.0
		obj_path = os.path.join(ctx.assets_dir, "retrieved/mesh/354ceb04-5d52-4eb4-bb9b-c0da3ea49901.glb")
		obj_name = bus_stop_name
		place_mesh(env, f"Bus Stop {obj_name}", obj_path, obj_location, obj_rotation, obj_scale)

	for bicycle_station_name in bicycle_stations.keys():
		obj_location = np.array([bicycle_stations[bicycle_station_name][0], bicycle_stations[bicycle_station_name][1], get_height_at(terrain_height_field, bicycle_stations[bicycle_station_name][0], bicycle_stations[bicycle_station_name][1])])
		obj_rotation = np.array([0.0, 0.0, 0.0])
		obj_scale = 1.0
		obj_path = os.path.join(ctx.assets_dir, "generated/mesh/72d9f0dbed984b8da68bffd5f2638358.glb")
		obj_name = bicycle_station_name
		place_mesh(env, obj_name, obj_path, obj_location, obj_rotation, obj_scale)