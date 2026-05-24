import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import math
import genesis as gs

import pickle

from ...tools.utils import *
from ...tools.road_annotation.retrieve_nearest_road import point_to_road_distance

def euler_to_matrix(euler: np.ndarray, unit: str = 'rad') -> np.ndarray:
    '''Convert Euler angles to rotation matrix.
    Args:
        euler: np.ndarray
            Euler angles
    Returns:
        np.ndarray: Rotation matrix
    '''
    if unit == 'deg':
        euler = np.deg2rad(euler)
    R_x = np.array([[1, 0, 0], [0, np.cos(euler[0]), -np.sin(euler[0])], [0, np.sin(euler[0]), np.cos(euler[0])]])
    R_y = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])], [0, 1, 0], [-np.sin(euler[1]), 0, np.cos(euler[1])]])
    R_z = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0], [np.sin(euler[2]), np.cos(euler[2]), 0], [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))

def matrix_to_euler(matrix: np.ndarray, unit: str = 'rad') -> np.ndarray:
    '''Convert rotation matrix to Euler angles.
    Args:
        matrix: np.ndarray
            Rotation matrix
    Returns:
        np.ndarray: Euler angles
    '''
    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], sy)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], sy)
        z = 0
    if unit == 'deg':
        return np.rad2deg([x, y, z])
    else:
        return np.array([x, y, z])

def compose_euler(euler_a: np.ndarray, euler_b: np.ndarray, unit: str = 'rad') -> np.ndarray:
    '''Compose two Euler angles.
    Args:
        euler_a: np.ndarray
            First set of Euler angles
        euler_b: np.ndarray
            Second set of Euler angles
    Returns:
        np.ndarray: Composed Euler angles
    '''
    matrix_a = euler_to_matrix(euler_a, unit)
    matrix_b = euler_to_matrix(euler_b, unit)
    matrix_c = np.dot(matrix_b, matrix_a)
    return matrix_to_euler(matrix_c, unit)

@dataclass 
class OutdoorObjectContext:
    '''Context for placing objects in outdoor scenes.

    Attributes:
        scene_name: str
            Scene name
        objects_cfg_dir: str
            Path to the object config directory
        assets_dir: str
            Path to the object assets directory
        max_objects: int | Dict[str, int], optional
            Maximum number of objects to load for each group. 
            If int, the same number of objects will be loaded for all groups.
            If dict, the number of objects to load for each group is specified by the group id.
        seed: int, optional
            Random seed for shuffling objects in each group.
            By default, it is 0.
        street_cfg_path: str, optional
            Path to the street config file.
    '''

    scene_name: str = 'NY'
    assets_dir: str = 'ViCo/objects/outdoor_objects'
    objects_cfg_dir: str = 'assets/scene/v1/NY/objects'
    max_objects: Optional[Union[int, Dict[str, int]]] = None
    seed: int = 0
    terrain_height_field_path: str = ''
    road_info_path: str = ''

    def __post_init__(self):
        self.random_state = random.Random(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        self.objects_cfg_dir = os.path.join(get_assets_dir(), self.objects_cfg_dir)

        #* Load center location info
        self.center_location_info_path = os.path.join(
            os.path.dirname(self.objects_cfg_dir), 'center.txt'
        )
        assert os.path.exists(self.center_location_info_path), f'Center location info not found at {self.center_location_info_path}'
        with open(self.center_location_info_path, 'r') as f:
            self.base_lat, self.base_lon = map(float, f.readline().strip().split())

        #* Load terrain height field
        if not self.terrain_height_field_path:
            self.terrain_height_field_path = f'{self.assets_dir}/height_field.npz'
        self.terrain_height_field_path = os.path.join(get_assets_dir(), self.terrain_height_field_path)
        # assert os.path.exists(self.terrain_height_field_path), f'Terrain height field not found at {self.terrain_height_field_path}'
        self.terrain_height_field = None
        if os.path.exists(self.terrain_height_field_path):
            self.terrain_height_field = load_height_field(self.terrain_height_field_path)

        #* Load road info
        if not self.road_info_path:
            self.road_info_path = f'assets/scenes/{self.scene_name}/roads.pkl'
        # assert os.path.exists(self.road_info_path), f'Road info not found at {self.road_info_path}'
        self.road_info = None
        if os.path.exists(self.road_info_path):
            with open(self.road_info_path, 'rb') as f:
                self.road_info = pickle.load(f)

    def append_position_with_height(self, position: np.ndarray) -> np.ndarray:
        if self.terrain_height_field:
            position = np.array([position[0], position[1], get_height_at(self.terrain_height_field, position[0], position[1])])
        return position

    def max_objects_in(self, group_id: str) -> Optional[int]:
        if isinstance(self.max_objects, int):
            return self.max_objects
        elif isinstance(self.max_objects, dict):
            return self.max_objects.get(group_id, None)
        return None

    def sample_objects(self, objs, group_id: str, override: Optional[int] = None) -> List[Any]:
        if override is not None and isinstance(override, int):
            self.random_state.shuffle(objs)
            objs = objs[:override]
            return objs
        max_objects_in_group = self.max_objects_in(group_id)
        if max_objects_in_group and max_objects_in_group < len(objs):
            self.random_state.shuffle(objs)
            objs = objs[:max_objects_in_group]
        return objs

    def align_with_road(
        self, obj_info: Dict[str, Any],
        distance: Union[float, Tuple[float, float], List[float]] = (0.0, math.inf),
        angle: Union[float, Tuple[float, float], List[float]] = (-math.pi, math.pi),
    ) -> Dict[str, Any]:
        '''Align the object with the road.
        Args:
            obj_info: dict
                Information about the object
            distance: float | Tuple[float, float], optional
                Distance range from the road edge
            angle: float | Tuple[float, float], optional
                Angle range from the road. 
                Note that 0 is the direction that prependicularly points to the road.
        Returns:
            dict: Updated object information
        '''
        if self.road_info is None:
            return obj_info

        if isinstance(distance, list):
            distance = (distance[0], distance[1])
        elif isinstance(distance, float) or isinstance(distance, int):
            distance = (distance, distance)
        
        if isinstance(angle, list):
            angle = (angle[0], angle[1])
        elif isinstance(angle, float) or isinstance(angle, int):
            angle = (angle, angle)
        
        def find_closest_road(position, roads):
            min_distance = math.inf
            closest_road = None
            closest_point = None
            road_width = 0
            for road in roads:
                d, point, width = point_to_road_distance(position, road, self.base_lat, self.base_lon)
                if d < min_distance:
                    min_distance = d
                    closest_road = road
                    closest_point = point
                    road_width = width
            return min_distance, closest_point, road_width, closest_road

        obj_position = obj_info['location'][:2]
        obj_rotation = obj_info['rotation']
        obj_distance, cloest_point, road_width, _ = find_closest_road(obj_position, self.road_info[0])
        
        obj_direction = cloest_point - obj_position
        obj_direction /= np.linalg.norm(obj_direction)
        obj_orientation = math.atan2(obj_direction[1], obj_direction[0])

        aligned_obj_distance = np.clip(obj_distance, distance[0] + road_width, distance[1] + road_width)
        aligned_obj_orientation = np.clip(
            (-obj_orientation + math.pi) % (2 * math.pi) - math.pi,
            angle[0], angle[1]
        )

        obj_info['position'] = self.append_position_with_height(
            cloest_point - aligned_obj_distance * obj_direction
        )
        obj_info['rotation'] = compose_euler(
            obj_rotation, 
            np.array([0, 0, obj_orientation + aligned_obj_orientation])
        )
        return obj_info 

    def rescale_object(
        self, obj_info: Dict[str, Any],
        height: Union[float, Tuple[float, float], List[float]] = (0.0, math.inf),
    ) -> Dict[str, Any]:
        '''Rescale the object.
        Args:
            obj_info: dict
                Information about the object
        Returns:
            dict: Updated object information
        '''
        if isinstance(height, list):
            height = (height[0], height[1])
        elif isinstance(height, float) or isinstance(height, int):
            height = (height, height)
        
        import trimesh
        mesh = trimesh.load_mesh(os.path.join(get_assets_dir(), obj_info['path']))
        bbox = mesh.bounding_box.bounds
        obj_height = bbox[1][1] - bbox[0][1]
        rescaled_obj_height = np.clip(obj_height, height[0], height[1])
        rescale_ratio = rescaled_obj_height / obj_height
        obj_info['scale'] *= rescale_ratio

        return obj_info
