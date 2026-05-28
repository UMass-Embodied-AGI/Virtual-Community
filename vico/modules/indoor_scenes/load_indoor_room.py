import os
import genesis as gs
import psutil
import json
import numpy as np
from ...tools.asset_utils import ensure_asset
from .usd_scene import place_usd_scene_with_ratio
from .architect_scene import load_indoor_scene
from ...tools.constants import ASSETS_PATH


def load_default_room(env, place_name='default_room'):
    indoor_scene = os.path.join(ASSETS_PATH, "indoor_scenes/accomodation-8.json")  # a default room
    
    with open(indoor_scene, 'r') as f:
        indoor_scene_place = json.load(f)

    active_places_info = indoor_scene_place
    try:
        place_cameras = load_indoor_scene(env,
                                            place=[indoor_scene_place],
                                            offset_x=1500,
                                            offset_y=1500,
                                            offset_z=-50,
                                            no_objects=True)[0][1]
    except Exception as e:
        gs.logger.error(f"Error loading indoor scene for place {place_name}: {e}")
    return active_places_info, place_cameras

def load_indoor_room(env, scene_path: str, offset: list[float], place_name: str, load_indoor_objects: bool):
    try:
        with open(os.path.join(ASSETS_PATH, scene_path), 'r') as f:
            indoor_scene = json.load(f)
        indoor_scene_type = indoor_scene["type"]
        indoor_scene_name = indoor_scene["name"]
        if indoor_scene_type == "glb":
            active_places_info = indoor_scene
            place_cameras = load_indoor_scene(env,
                                                place=[indoor_scene],
                                                offset_x=offset[0],
                                                offset_y=offset[1],
                                                offset_z=offset[2],
                                                no_objects=not load_indoor_objects)[0][1]
        elif indoor_scene_type == "usd":
            cameras = indoor_scene.get("cameras", [])
            place_cameras = []
            for cam in cameras:
                pos = cam['pos'] + offset
                lookat = cam['lookat'] + offset
                place_cameras.append(
                    (pos, lookat)
                )
            active_places_info = {
                'init_avatar_poses': indoor_scene.get("avatar_pos", [])
            }
            usd_file = ensure_asset(f"scene/commercial_scenes/scenes/{indoor_scene_name}_usd/start_result_raw.usd")
            place_usd_scene_with_ratio(usd_file,
                        env,
                        global_pos=offset,
                        load_objects=load_indoor_objects
            )
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            gs.logger.info(f"Memory used: {mem} MB")
        else:
            gs.logger.error(f"Invalid indoor scene type with scene path {scene_path}: {indoor_scene_type}")
            return None, None
        return active_places_info, place_cameras
    except Exception as e:
        gs.logger.error(f"Error loading indoor scene for place {place_name} with scene path {scene_path}: {e}")
        return None, None