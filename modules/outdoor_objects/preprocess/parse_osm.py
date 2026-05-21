import math
import requests
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.utils import get_assets_dir
import argparse
import json
import yaml

import geojson
from geojson.utils import (
    map_coords as geo_map_coords, 
    map_tuples as geo_map_tuples, 
    coords as geo_coords,
)

import numpy as np
import osmium

from dataclasses import dataclass, field, replace

from typing import Any, List, Dict, Tuple, Optional, Mapping, Iterable, Callable, Type

@dataclass
class Bounds:
    maxlat: float = 0.
    maxlon: float = 0.
    minlat: float = 0.
    minlon: float = 0.

    @staticmethod
    def from_circle(lat: float, lon: float, r: float) -> 'Bounds':
        base_e, base_n, zone_number, zone_letter = utm.from_latlon(lat, lon)
        maxlat, maxlon = utm.to_latlon(base_e + r, base_n + r, zone_number, zone_letter)
        minlat, minlon = utm.to_latlon(base_e - r, base_n - r, zone_number, zone_letter)
        return Bounds(maxlat, maxlon, minlat, minlon)

    def __str__(self):
        return f'[n={self.maxlat:.6f}, e={self.maxlon:.6f}, s={self.minlat:.6f}, w={self.minlon:.6f}]'

@dataclass
class OsmObject:
    id: str = ''
    asset_path: str = ''
    geom: Optional[dict] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def obj_type(self) -> str:
        return self.geom.get('type', None) if self.geom else None

    def to_json(self):
        return {
            'id': self.id,
            'asset_path': self.asset_path,
            'location': self.geom['coordinates'],
            'rotation': (0.0, 0.0, 0.0),
            'tags': self.tags
        }
    
    def is_valid(self):
        for x in self.geom['coordinates']:
            if math.isnan(x):
                return False
        return True
        
    def map_coords(self, func):
        return replace(self, geom=geo_map_coords(func, self.geom))
    def map_tuples(self, func):
        return replace(self, geom=geo_map_tuples(func, self.geom))

@dataclass
class OsmObjectCollection:
    description: str = ''
    objs: List[OsmObject] = field(default_factory=list)

    def append(self, obj: OsmObject):
        self.objs.append(obj)
    def __iter__(self):
        return iter(self.objs)
    def __len__(self):
        return len(self.objs)

    def to_json(self):
        return {
            'description': self.description,
            'objects': [o.to_json() for o in self.objs if o.is_valid()]
        }
    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f)
    
    def map_coords(self, func):
        return replace(self, objs=[o.map_coords(func) for o in self.objs])
    def map_tuples(self, func):
        return replace(self, objs=[o.map_tuples(func) for o in self.objs])

class OsmHandler(osmium.SimpleHandler):
    def __init__(
        self, objs: Optional[OsmObjectCollection] = None,
        node_filter: Optional[Callable[[osmium.osm.Node], bool]] = None,
        way_filter: Optional[Callable[[osmium.osm.Way], bool]] = None,
        area_filter: Optional[Callable[[osmium.osm.Area], bool]] = None,
        asset_selector: Optional[Callable[[osmium.osm.OSMObject], str]] = None
    ):
        super().__init__()
        self.factory = osmium.geom.GeoJSONFactory()
        self.objs = objs if objs is not None else OsmObjectCollection() 
        self.node_filter = node_filter
        self.way_filter = way_filter
        self.area_filter = area_filter
        self.asset_selector = asset_selector

    def node(self, n):
        if self.node_filter and self.node_filter(n):
            asset_info = self.asset_selector(n) if self.asset_selector else {}
            osm_tags = {str(k): str(v) for k, v in n.tags}
            asset_tags = {k: v for k, v in asset_info.items() if k not in ['path']}
            geom = geojson.loads(self.factory.create_point(n))
            obj = OsmObject(
                id=n.id, 
                asset_path=asset_info.get('path', ''),
                geom=geom, 
                tags={**osm_tags, **asset_tags}
            )
            self.objs.append(obj)

    def way(self, w):
        if self.way_filter and self.way_filter(w):
            asset_info = self.asset_selector(w) if self.asset_selector else {}
            osm_tags = {str(k): str(v) for k, v in w.tags}
            asset_tags = {k: v for k, v in asset_info.items() if k not in ['path']}
            geom = geojson.loads(
                self.factory.create_linestring(w, use_nodes=osmium.geom.UNIQUE, direction=osmium.geom.FORWARD)
            )
            obj = OsmObject(
                id=w.id, 
                asset_path=asset_info.get('path', ''),
                geom=geom, 
                tags={**osm_tags, **asset_tags}
            )
            self.objs.append(obj)

    def area(self, a):
        if self.area_filter and self.area_filter(a):
            asset_info = self.asset_selector(a) if self.asset_selector else {}
            osm_tags = {str(k): str(v) for k, v in a.tags}
            asset_tags = {k: v for k, v in asset_info.items() if k not in ['path']}
            geom = geojson.loads(self.factory.create_multipolygon(a))
            obj = OsmObject(
                id=a.id, 
                asset_path=asset_info.get('path', ''),
                geom=geom, 
                tags={**osm_tags, **asset_tags}
            )
            self.objs.append(obj)

class OSMParser:
    api: str = 'http://overpass-api.de/api/map?bbox={w},{s},{e},{n}'
    
    def __init__(
        self, temp_dir: str = 'tmp/osm_cache',
    ) -> None:
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        # self._register('amenities', Amenity, AmenitiesHandler)
        # self._register('trees', Tree, TreesHandler)

    def fetch_osm(
        self, bounds: Bounds, 
        enable_cache: bool = True
    ) -> str:
        file_path = os.path.join(self.temp_dir, f'{bounds}.osm')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if enable_cache and os.path.exists(file_path):
            return file_path
        
        url = self.api.format(
            n=bounds.maxlat, e=bounds.maxlon, 
            s=bounds.minlat, w=bounds.minlon
        )
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return file_path

    def parse(self, bounds: Bounds, handler: OsmHandler) -> OsmObjectCollection:
        osm_file = self.fetch_osm(bounds)
        handler.apply_file(osm_file, locations=True)
        return handler.objs

def parse_osm(
    scene: str, scene_assets_dir: str,
    base_lat: Optional[float] = None,
    base_lon: Optional[float] = None,
    radius: float = 400.0,
    parser_cfg_path: str = 'parser_config.yaml',
    assets_cfg_path: str = 'assets_config.yaml',
    output_dir: str = 'NY_ok/objects',
    seed: int = 0
):  
    '''
    Parse OSM data and generate object placement configurations.

    Args:
        scene (str): Scene name.
        scene_assets_dir (str): Directory to store scene assets.
        base_lat (float): Latitude of the base location.
        base_lon (float): Longitude of the base location.
        radius (float): Radius of the area to parse.
        parser_cfg_path (str): Path to the parser configuration file, which describes which categories of objects to parse.
        assets_cfg_path (str): Path to the assets configuration file, which stores information of assets in categories.
        output_dir (str): Directory to save the generated object placement configurations
        seed (int): Random seed.
    '''

    np.random.seed(seed)
    
    center_file_path = os.path.join(scene_assets_dir, 'center.txt')
    if os.path.exists(center_file_path):
        with open(center_file_path, 'r') as f:
            base_lat, base_lon = map(float, f.readline().split())
    assert base_lat is not None and base_lon is not None, 'Base location is not provided'

    bounds = Bounds.from_circle(base_lat, base_lon, radius)
    base_e, base_n = utm.from_latlon(base_lat, base_lon)[:2]
    os.makedirs(output_dir, exist_ok=True)

    with open(assets_cfg_path, 'r') as f:
        asset_paths = yaml.load(f, Loader=yaml.FullLoader)
    with open(parser_cfg_path, 'r') as f:
        groups = yaml.load(f, Loader=yaml.FullLoader)
    
    def latlon_to_xy(location):
        lon, lat = location
        e, n = utm.from_latlon(lat, lon)[:2]
        return (e - base_e, n - base_n)

    meta = {
        'scene': scene, 
        'groups': {}
    }
    selection_helper = {}

    parser = OSMParser()
    for group_id, group_info in groups.items():
        objs = OsmObjectCollection()
        for osm_tag, asset_info in group_info.items():
            tag_key, tag_value = osm_tag.split('=')
            asset_category, asset_selector_mode = asset_info.split(':')
            
            # by default, randomly select an asset from the category
            if asset_selector_mode == 'random':
                asset_selector = lambda n: np.random.choice(asset_paths[asset_category])
            # for some cases, it's better to select a same asset for all objects in the group
            elif asset_selector_mode == 'same':
                if asset_category not in selection_helper:
                    selection_helper[asset_category] = np.random.choice(asset_paths[asset_category])
                asset_selector = lambda n: selection_helper[asset_category]
            else:
                raise ValueError(f'Invalid asset selector mode: {asset_selector_mode}')

            handler = OsmHandler(
                objs=objs,
                node_filter=lambda n: n.tags.get(tag_key) == tag_value,
                asset_selector=asset_selector
            )
            objs = parser.parse(bounds, handler)
        
        objs = objs.map_tuples(latlon_to_xy)        
        objs.save(os.path.join(output_dir, f'{group_id}.json'))
        meta['groups'][group_id] = f'{group_id}.json'

    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse OSM data')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--scene', type=str, required=True, help='Scene name')
    parser.add_argument('--parser_cfg_path', type=str, default='modules/outdoor_objects/parser_config.yaml', help='Path to the groups configuration file')
    parser.add_argument('--assets_cfg_path', type=str, default='modules/outdoor_objects/assets_config.yaml', help='Path to the assets configuration file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the generated object placement configurations')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(get_assets_dir(), f"ViCo/scene/v1/{args.scene}/objects")
    scene_assets_dir = os.path.join(get_assets_dir(), "ViCo/scene/v1", args.scene)
    parse_osm(
        scene=args.scene, 
        scene_assets_dir=scene_assets_dir,
        parser_cfg_path=args.parser_cfg_path, 
        assets_cfg_path=args.assets_cfg_path, 
        output_dir=args.output_dir,
        seed=args.seed
    )