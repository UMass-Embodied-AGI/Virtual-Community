from vico.tools.constants import ASSETS_PATH
import json
import os
import pdb
import pickle
import osmnx as ox
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
import math
import requests
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
import time

import carla
from pathlib import Path

def fetch_buildings(lat, lng, rad):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"](around:{rad},{lat},{lng});
      relation["building"](around:{rad},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data


# Step 2: Parse the building data and create Shapely polygons
def parse_buildings(data):
    # Step 1: Create a dictionary for nodes
    nodes = {element['id']: (element['lat'], element['lon']) for element in data['elements'] if
             element['type'] == 'node'}
    nodes_1 = {element['id']: element for element in data['elements'] if
             element['type'] != 'node'}
    buildings = []

    # Step 2: Create polygons for ways
    for element in data['elements']:
        if element['type'] == 'way' and 'members' not in element and len(element['nodes']) > 3:
            try:
                coords = [(nodes[node_id][1], nodes[node_id][0]) for node_id in element['nodes']]
                if len(coords) < 4:
                    pdb.set_trace()
                buildings.append(Polygon(coords))
            except KeyError:
                # Handle the case where node_id is not found in nodes
                continue

        # Step 3: Handle relations (multipolygon)
        elif element['type'] == 'relation' or 'members' in element:
            outer_coords = []
            for member in element['members']:
                if member['role'] == 'outer' and member['type'] == 'way':
                    way_id = member['ref']
                    if way_id in [way['id'] for way in data['elements'] if way['type'] == 'way']:
                        way = next(way for way in data['elements'] if way['id'] == way_id)
                        outer_coords.extend([(nodes[node_id][1], nodes[node_id][0]) for node_id in way['nodes']])

            if outer_coords:
                outer_polygon = Polygon(outer_coords)
                buildings.append(outer_polygon)

    return buildings


def reorder_points(points_list, nodes_list):
    nodes_table = {}
    for idx, nodes in enumerate(nodes_list):
        assert len(nodes) > 1
        for node in [nodes[0], nodes[-1]]:
            if node not in nodes_table:
                nodes_table[node] = []
            nodes_table[node].append(idx)

    ordered_points = []
    ordered_nodes = []
    starting_nodes = []
    for k, v in nodes_table.items():
        if len(v) % 2 == 1:
            starting_nodes.append((k, v[0]))
            nodes_table[k] = v[1:]
    current_node = starting_nodes[0][0]
    current_idx = starting_nodes[0][1]
    ordered_points.append(points_list[current_idx])
    ordered_nodes.append(nodes_list[current_idx])
    processed_starting_nodes = [current_node]
    counter = 0
    while len(processed_starting_nodes) < len(starting_nodes):
        counter += 1
        start_n, end_n = nodes_list[current_idx][0], nodes_list[current_idx][-1]
        current_node = start_n if end_n == current_node else end_n
        if len(nodes_table[current_node]) == 2:
            current_idx = nodes_table[current_node][0] if nodes_table[current_node][1] == current_idx else nodes_table[current_node][1]
        elif len(nodes_table[current_node]) == 0:
            processed_starting_nodes.append(current_node)
            if len(processed_starting_nodes) < len(starting_nodes):
                for node in starting_nodes:
                    if node[0] not in processed_starting_nodes:
                        current_node = node[0]
                        current_idx = node[1]
                        break
                continue
        else:
            for j, idx in enumerate(nodes_table[current_node]):
                if idx == current_idx:
                    nodes_table[current_node].remove(current_idx)
                    current_idx = nodes_table[current_node][0]
                    nodes_table[current_node].remove(current_idx)
                    break
        if counter > 1000:
            assert "Error processing OSM roads!"
        ordered_points.append(points_list[current_idx])
        ordered_nodes.append(nodes_list[current_idx])
    ordered_points = sum(ordered_points, [])
    ordered_nodes = sum(ordered_nodes, [])
    return np.array(ordered_points), np.array(ordered_nodes)


def sample_points_on_way(way_nodes, num_points=100):
    lats = [node['lat'] for node in way_nodes]
    lons = [node['lon'] for node in way_nodes]

    lat_samples = np.interp(np.linspace(0, len(lats) - 1, num_points), np.arange(len(lats)), lats)
    lon_samples = np.interp(np.linspace(0, len(lons) - 1, num_points), np.arange(len(lons)), lons)

    return np.column_stack((lat_samples, lon_samples))


def get_roads(lat, lng, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (
      way["highway"](around:{radius},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def get_osm_data(lat, lng, radius):
    # Construct the Overpass API query
    query = f"""
    [out:xml];
    (
      node(around:{radius},{lat},{lng});
      way(around:{radius},{lat},{lng});
      relation(around:{radius},{lat},{lng});
    );
    out body;
    """
    
    # Send request to Overpass API
    response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
    
    if response.status_code == 200:
        return response.text  # Return the raw XML data
    else:
        raise Exception("Error fetching data from Overpass API")



def get_ground_areas(lat, lng, radius):
    location_point = (lat, lng)
    tags = {
        'leisure': ['park', 'pitch', 'playground'],
        'landuse': ['grass', 'meadow', 'recreation_ground'],
        'natural': ['grassland', 'meadow'],
        'amenity': ['parking']
    }
    gdf = ox.geometries.geometries_from_point(location_point, tags, dist=radius)
    gdf_polygons = gdf[gdf.geometry.type == 'Polygon']
    return gdf_polygons


def fetch_osm(lat, lng, rad, scene_name):
    if os.path.exists(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.pkl"):
        road_data=pickle.load(open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.pkl","rb"))
    else:
        road_data = get_roads(lat, lng, rad)
        pickle.dump(road_data,open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.pkl","wb"))
    if os.path.exists(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.osm"):
        with open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.osm","r", encoding="utf-8") as file:
            osm_data=file.read()
    else:
        osm_data = get_osm_data(lat, lng, rad)
        with open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.osm","w", encoding="utf-8") as file:
            file.write(osm_data)
    try:
        # Define the desired settings. In this case, default values.
        settings = carla.Osm2OdrSettings()
        settings.proj_string='+proj=merc'
        # Set OSM road types to export to OpenDRIVE
        settings.set_osm_way_types(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"])
        # Convert to .xodr
        xodr_data = carla.Osm2Odr.convert(osm_data, settings)
        with open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/road_data.xodr","w", encoding='utf-8') as file:
            file.write(xodr_data)
    except Exception as e:
        pass

    ground_data = None#get_ground_areas(lat, lng, rad)
    data = fetch_buildings(lat, lng, rad)
    buildings = parse_buildings(data)

    if not road_data and not ground_data:
        print("No data found.")
        return
    else:
        print("Data loaded")

    # Dictionary to store the result
    roads_dict = {}
    ground_polygons = []
    types = set()
    if road_data:
        # Create a dictionary of node ID to node coordinates
        node_dict = {node['id']: {'lat': node['lat'], 'lon': node['lon']} for node in road_data['elements'] if
                     node['type'] == 'node'}

        for element in road_data['elements']:
            types.update({element['type']})
            if element['type'] == 'way' and 'tags' in element:
                name = element['tags'].get('name', f"unnamed road {element['id']}")
                highway = element['tags'].get('highway', 'footway')
                covered = element['tags'].get('covered', False)
                tunnel = element['tags'].get('tunnel', '')
                layer = element['tags'].get('layer', 0)
                covered = covered or tunnel != '' or str(layer).startswith('-')

                # Get the node coordinates for this way
                way_nodes = [{'lat': node_dict[node_id]['lat'], 'lon': node_dict[node_id]['lon']} for node_id in
                             element['nodes'] if node_id in node_dict]

                # Sample points on this way
                sampled_points = sample_points_on_way(way_nodes, num_points=300)

                # Store the sampled points in the dictionary
                if name in roads_dict:
                    roads_dict[name][1].append(sampled_points.tolist())
                    roads_dict[name][3].append(element['nodes'])
                else:
                    roads_dict[name] = [highway, [sampled_points.tolist()], covered, [element['nodes']]]

    # for name in tqdm(roads_dict):
    #     if len(roads_dict[name][1]) > 1:
    #         roads_dict[name][1], roads_dict[name][3] = reorder_points(roads_dict[name][1], roads_dict[name][3])
    #     else:
    #         roads_dict[name][1] = roads_dict[name][1][0]
    #         roads_dict[name][3] = roads_dict[name][3][0]
    return road_data, roads_dict, node_dict


def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    earth_radius = 6378137  # in meters
    meters_per_degree_lat = 111139  # Approximate meters per degree latitude
    
    # Calculate meters per degree longitude based on reference latitude
    meters_per_degree_lon = 111139 * math.cos(math.radians(ref_lat))
    
    # Convert lat/lon to x/y
    x = (lon - ref_lon) * meters_per_degree_lon
    y = (lat - ref_lat) * meters_per_degree_lat
    
    return [x, y]

def get_road_from_latlng(lat, lng, api_key):
    time.sleep(1)
    GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key
    }
    response = requests.get(GEOCODING_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            # formatted_address = data["results"][0]["formatted_address"]
            address_components = data["results"][0]["address_components"]
            road_name = None
            for component in address_components:
                if "route" in component["types"]:
                    road_name = component["long_name"]
                    break
            return road_name
        else:
            print(f"Error: {data['status']}")
    else:
        print(f"Request failed with status code {response.status_code}")
    return None

def draw_line_segment(start, end, width, color='white'):
    # Calculate the direction vector from start to end
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    
    # Normalize the direction vector
    if length == 0:
        return  # Avoid division by zero if start and end are the same
    direction /= length
    
    # Calculate perpendicular vector
    perp_direction = np.array([-direction[1], direction[0]])  # Rotate 90 degrees
    
    # Calculate offset for width
    offset = (width / 2) * perp_direction
    
    # Define the four corners of the line segment rectangle
    corner1 = start + offset
    corner2 = start - offset
    corner3 = end - offset
    corner4 = end + offset
    
    # Create a polygon (rectangle) representing the line segment with width
    line_segment = np.array([corner1, corner2, corner3, corner4])
    
    # Plotting
    plt.fill(line_segment[:, 0], line_segment[:, 1], color=color)  # Draw filled rectangle for width
    plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)  # Draw main line

def get_road_area(road_data, ref_lat, ref_lon, api_key=None):
    '''This function is for getting all roads and their corresponding nodes that forming their reference lines.'''
    roads=[]
    nodes = {node['id']: {'lat': node['lat'], 'lon': node['lon'], 'connected_roads': []} for node in road_data['elements'] if node['type'] == 'node'}
    for i in nodes:
        x, y = lat_lon_to_xy(nodes[i]['lat'],nodes[i]['lon'], ref_lat, ref_lon)
        nodes[i]['x'], nodes[i]['y']=x, y

    for element in road_data['elements']:
        if element['type'] == 'way' and 'tags' in element:
            # Get the node coordinates for this way
            way_nodes = [{'lat': nodes[node_id]['lat'], 'lon': nodes[node_id]['lon'], 'id': node_id, 'x':nodes[node_id]['x'], 'y': nodes[node_id]['y']} for node_id in element['nodes'] if node_id in nodes]
            assert len(way_nodes)>1
            # Get Road infos
            name = element['tags'].get('name', get_road_from_latlng((way_nodes[0]['lat']+way_nodes[1]['lat'])/2,(way_nodes[0]['lon']+way_nodes[1]['lon'])/2, api_key) if api_key is not None else f"unnamed road {element['id']}")
            highway = element['tags'].get('highway', 'footway')
            covered = element['tags'].get('covered', False)
            tunnel = element['tags'].get('tunnel', '')
            layer = element['tags'].get('layer', 0)
            covered = covered or tunnel != '' or str(layer).startswith('-')
            oneway = element['tags'].get('oneway', 'no')
            lanes = int(element['tags'].get('lanes',0))

            # Get Width
            width = [float(''.join(element['tags'].get('width','0').split()[0].split(';')[0].split('\'')[0])),'width']
            if width[0] == 0:
                # print(type(element['tags'].get('lanes',0)), element['tags'].get('lanes',0))
                width = [lanes * 3.5,'lanes']
            if width[0] == 0:
                try:
                    width = [{'primary': 30, 'secondary': 20, 'tertiary': 12, 'residential': 8, 'cycleway':8, 'pedestrian': 2, 'footway': 2.8, 'service': 8, 'unclassified': 8, 'steps': 8, 'elevator':8, 'construction': 8, 'path': 8}[highway], 'highway']
                except Exception as e:
                    print(highway)
            for i in range(1,len(way_nodes)):
                # every segment between nodes is counted as an independent road
                id = len(roads)
                start = way_nodes[i-1]
                end = way_nodes[i]
                nodes[start['id']]['connected_roads'].append(id)
                roads.append({'id': id, 'name': name, 'start':start, 'end':end, 'highway': highway, 'width': width[0], 'lanes':lanes, 'oneway': oneway})
    return roads, nodes

def draw_roads(roads, nodes, ref_lat, ref_lon):
    color_dict={'primary': 'red', 'secondary': 'orangered', 'tertiary': 'orange', 'residential': 'gold', 'cycleway':'green', 'pedestrian': 'blue', 'footway': 'deepskyblue', 'service': 'peru', 'unclassified': 'm', 'steps': 'blue', 'elevator':'violet', 'living_street':'pink', 'construction': 'orchid'}
    for road in roads:
        # print(road)
        highway = road['highway']
        color = 'grey'
        if highway in color_dict:
            color = color_dict[highway]
        # color = {'yes': 'red', 'no': 'blue'}[road['oneway']]
        start_x, start_y = road['start']['x'],road['start']['y']
        end_x, end_y = road['end']['x'],road['end']['y']
        # print(road['width'])
        # assert not isinstance(road['width'],str)
        draw_line_segment([start_x, start_y], [end_x, end_y] ,road['width'], color)
    # for node in nodes.values():
    #     plt.scatter(node['x'], node['y'], color='lime', s=2)

# python tools/fetch_osm_roads.py -s newyork --ref_lat 40.748998486718186 --ref_lon -73.9882893780644 --radius 400
# python tools/fetch_osm_roads.py -s detroit --ref_lat 42.33165461030516 --ref_lon -83.0480662316049 --radius 400
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    # parser.add_argument("--ref_lat", type=float, required=True)
    # parser.add_argument("--ref_lon", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    args = parser.parse_args()
    if os.path.exists(f"{ASSETS_PATH}/scenes/{args.scene}"):
        print("Scene exists")
        Path(f"{ASSETS_PATH}/scenes/{args.scene}/road_data").mkdir(parents=True, exist_ok=True)
        with open(f"{ASSETS_PATH}/scenes/{args.scene}/raw/center.txt", "r") as f:
            ref_lat, ref_lon=f.readline().split()
            args.ref_lat, args.ref_lon=float(ref_lat), float(ref_lon)
            print(f"retrieved coordinates from raw file. lat: {args.ref_lat}, lon: {args.ref_lon}")
    else:
        print(f"Scene not exist: assets/scenes/{args.scene}")
        exit()
    road_data, roads_dict, node_dict = fetch_osm(lat=args.ref_lat,
                                      lng=args.ref_lon,
                                      rad=400, scene_name=args.scene)
    # pickle.dump((roads_dict, node_dict), open("road_ny.pkl", 'wb'))
    # json.dump(width_dict, open("road_ny.json","w"))

    # Set up the plot area
    plt.figure(figsize=(8, 8))
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()

    roads, nodes=get_road_area(road_data, args.ref_lat, args.ref_lon, api_key=None)
    pickle.dump((roads, nodes), open(f"{ASSETS_PATH}/scenes/{args.scene}/road_data/roads.pkl", 'wb'))
    draw_roads(roads, nodes, args.ref_lat, args.ref_lon)

    # Show the plot
    plt.title(f'Roads with Width - {args.scene}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()