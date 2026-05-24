from vico.tools.constants import ASSETS_PATH
import os
import copy
import json
import glob
import math
import random
import pickle
import argparse
import numpy as np
import pymap3d as pm
from sklearn.cluster import KMeans
from tqdm import tqdm
from math import factorial
from itertools import permutations
from collections import defaultdict, deque
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import datetime

from road_annotation.retrieve_nearest_road import point_to_road_distance

def draw_line_segment(start, end, width, fill_color='white', line_color='white'):
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction /= length
    perp_direction = np.array([-direction[1], direction[0]])
    offset = (width / 2) * perp_direction
    corner1 = start + offset
    corner2 = start - offset
    corner3 = end - offset
    corner4 = end + offset
    line_segment = np.array([corner1, corner2, corner3, corner4])
    plt.fill(line_segment[:, 0], line_segment[:, 1], color=fill_color)
    plt.plot([start[0], end[0]], [start[1], end[1]], color=line_color, linewidth=1)

def draw_roads(roads):
    for road in roads:
        draw_line_segment(road['start']['pos'],road['end']['pos'],road['width'],fill_color='lightgray', line_color='lightgray')

def are_points_close(p1, p2, epsilon=1):
    return np.linalg.norm(np.array(p1) - np.array(p2)) <= epsilon

def build_adjacency(roads, bus_allowed_highway_types):
    adjacency = defaultdict(list)
    for road in roads:
        highway = road['highway']
        if highway in bus_allowed_highway_types:
            start_node = road['start']['id']
            end_node = road['end']['id']
            start_pos = tuple(road['start']['pos'])
            end_pos = tuple(road['end']['pos'])
            adjacency[start_node].append({
                'start_node': start_node,
                'end_node': end_node,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'road': road
            })
            if road['oneway'] != 'yes':
                adjacency[end_node].append({
                    'start_node': end_node,
                    'end_node': start_node,
                    'start_pos': end_pos,
                    'end_pos': start_pos,
                    'road': road
                })
    return adjacency

def bfs_find_path(adjacency, start_node, end_node):
    # print(adjacency)
    visited = {start_node: (None, None)}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # print("current node:", current_node)
        if current_node == end_node:
            return backtrack_path(visited, current_node)
        for next_road in adjacency[current_node]:
            next_node = next_road['end_node']
            # print("next node:", next_node)
            if next_node not in visited:
                visited[next_node] = (current_node, next_road)
                queue.append(next_node)
    return None

def backtrack_path(visited, end_pos):
    path = []
    current = end_pos
    while True:
        prev_pos, road_info = visited[current]
        if prev_pos is None or road_info is None:
            break
        path.append(road_info)
        current = prev_pos
    path.reverse()
    return path

def convert_segments_to_waypoints(segments):
    if not segments:
        return []
    waypoints = []
    first_start = segments[0]['start_pos']
    waypoints.append(first_start)
    for seg in segments:
        seg_end = seg['end_pos']
        waypoints.append(seg_end)
    return waypoints

def calculate_route_length(waypoints):
    if not waypoints or len(waypoints) < 2:
        return 0
    total_length = 0
    for i in range(1, len(waypoints)):
        total_length += np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i-1]))
    return total_length

def find_bus_route(roads, bus_stop_roads, bus_allowed_highway_types):
    adjacency = build_adjacency(roads, bus_allowed_highway_types)
    route_segments = []
    for i in range(len(bus_stop_roads)):
        current_stop_road = {
            'start_node': bus_stop_roads[i]['start']['id'],
            'end_node': bus_stop_roads[i]['end']['id'],
            'start_pos': tuple(bus_stop_roads[i]['start']['pos']),
            'end_pos': tuple(bus_stop_roads[i]['end']['pos']),
            'road': bus_stop_roads[i]
        }
        route_segments.append(current_stop_road)
        if i < len(bus_stop_roads) - 1:
            start_node = bus_stop_roads[i]['end']['id']
            next_stop_road = bus_stop_roads[i+1]
            end_node = next_stop_road['start']['id']
            path_between = bfs_find_path(adjacency, start_node, end_node)
            if path_between is None:
                # print("No route found between bus stop roads.")
                return None
            for seg in path_between:
                route_segments.append(seg)
    waypoints = convert_segments_to_waypoints(route_segments)
    return waypoints

def change_waypoints_sep(waypoints, separation=5):
    waypoints = np.array(waypoints)
    sep_waypoints = [waypoints[0].tolist()]
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]
        segment_length = np.linalg.norm(end_point - start_point)
        num_points = int(segment_length // separation)
        for j in range(1, num_points + 1):
            new_point = start_point + (end_point - start_point) * (j * separation / segment_length)
            sep_waypoints.append(new_point.tolist())
    sep_waypoints.append(waypoints[-1].tolist())
    return np.array(sep_waypoints)

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norm_product if norm_product != 0 else 1
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_sparse_points(dense_points, stops, min_distance=100, turn_angle_threshold=20):
    dense_points = np.array(dense_points)
    query_points = [stop["position"] if "position" in stop else stop for stop in stops]
    query_points = np.array(query_points)
    sparse_points = []
    dense_tree = cKDTree(dense_points)
    _, closest_indices = dense_tree.query(query_points)
    closest_indices = sorted(set(closest_indices))
    last_index = closest_indices[0]
    sparse_points.append(tuple(dense_points[last_index]))
    for i in range(last_index + 1, len(dense_points)):
        include_point = False
        if i in closest_indices:
            include_point = True
        elif np.linalg.norm(dense_points[i] - dense_points[last_index]) > min_distance:
            include_point = True
        elif i > 0 and i < len(dense_points) - 1:
            angle = calculate_angle(dense_points[i - 1], dense_points[i], dense_points[i + 1])
            if angle < 180 - turn_angle_threshold and angle > turn_angle_threshold:
                include_point = True
        if include_point:
            sparse_points.append(tuple(dense_points[i]))
            last_index = i
    return sparse_points

def sample_bus_stop_from_waypoint(x0, y0):
    theta = random.uniform(0, 2*math.pi)
    x = x0 + 5*math.cos(theta)
    y = y0 + 5*math.sin(theta)
    return [x, y]

def get_bus_stop_from_road_info_and_waypoint(road_info, x0, y0):
    starting_position = road_info['start']['pos']
    ending_position = road_info['end']['pos']
    road_width = road_info['width']
    # calculate the bus stop position by extending the waypoint towards the normal direction of the road by half of the width (+0.5m by default, if it collides with the building, consider removing this) so that the bus stop is on the road side, not middle.
    road_vector = np.array(ending_position) - np.array(starting_position)
    road_vector /= np.linalg.norm(road_vector)
    normal_vector = np.array([-road_vector[1], road_vector[0]])
    bus_stop_position = np.array([x0, y0]) + (road_width / 2 + 0.5) * normal_vector
    return bus_stop_position.tolist()

def find_closest_cluster_from_waypoint(waypoint, clusters):
    min_distance = float('inf')
    closest_cluster = None
    for cluster in clusters:
        distance = np.linalg.norm(np.array(cluster) - np.array(waypoint))
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster
    return closest_cluster

def find_closest_waypoint_to_cluster_from_waypoints(waypoints, cluster):
    min_distance = float('inf')
    closest_waypoint = None
    for waypoint in waypoints:
        distance = np.linalg.norm(np.array(waypoint) - np.array(cluster))
        if distance < min_distance:
            min_distance = distance
            closest_waypoint = waypoint
    return closest_waypoint

def get_sparse_points_from_dense_and_stops(dense_points, stops, min_distance=100, turn_angle_threshold=5):
    dense_points = np.array(dense_points)
    query_points = [stop["position"] if "position" in stop else stop for stop in stops]
    query_points = np.array(query_points)
    sparse_points = []
    dense_tree = cKDTree(dense_points)
    _, closest_indices = dense_tree.query(query_points)
    closest_indices = sorted(set(closest_indices))
    last_index = closest_indices[0]
    sparse_points.append(tuple(dense_points[last_index]))
    for i in range(last_index + 1, len(dense_points)):
        include_point = False
        if i in closest_indices:
            include_point = True
        elif np.linalg.norm(dense_points[i] - dense_points[last_index]) > min_distance:
            include_point = True
        elif i > 0 and i < len(dense_points) - 1:
            angle = calculate_angle(dense_points[i - 1], dense_points[i], dense_points[i + 1])
            if angle < 180 - turn_angle_threshold and angle > turn_angle_threshold:
                include_point = True
        if include_point:
            sparse_points.append(tuple(dense_points[i]))
            last_index = i
    return sparse_points

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def remove_very_close_waypoints(waypoints, filter_distance=5):
    filtered_waypoints = [waypoints[0]]
    for i in range(1, len(waypoints)):
        last_waypoint = filtered_waypoints[-1]
        current_waypoint = waypoints[i]
        distance = euclidean_distance(last_waypoint, current_waypoint)
        if distance >= filter_distance:
            filtered_waypoints.append(current_waypoint)
    return filtered_waypoints

def find_closest_waypoints_near_bus_stops(dense_points, stops):
    dense_points = np.array(dense_points)
    query_points = [stop["position"] if "position" in stop else stop for stop in stops]
    query_points = np.array(query_points)
    dense_tree = cKDTree(dense_points)
    _, closest_indices = dense_tree.query(query_points)
    return closest_indices

def calculate_time_between_stops(d, start, end):
    if start == end:
        return 0
    total_time = 0
    if start < end:
        current_stop = start
        while current_stop < end:
            next_stop = current_stop + 1
            route = f"{current_stop}->{next_stop}"
            total_time += d.get(route)
            current_stop = next_stop
    else:
        current_stop = start
        while current_stop > end:
            next_stop = current_stop - 1
            route = f"{current_stop}->{next_stop}"
            total_time += d.get(route)
            current_stop = next_stop
    return total_time

def calculate_travel_time(bus_forward_speed, bus_route, bus_stop_indices):
    travel_time = {}
    cur_trans = bus_route[0]
    for idx, next_trans in enumerate(bus_route[1:]):
        distance = np.linalg.norm(np.array(next_trans) - np.array(cur_trans))
        travel_time[f"{idx}->{idx+1}"] = distance / bus_forward_speed + 1 # + 1 because we guarantee a turn of 1s everytime
        cur_trans = next_trans
    cur_trans = bus_route[-1]
    for idx, next_trans in enumerate(reversed(bus_route[:-1])):
        distance = np.linalg.norm(np.array(next_trans) - np.array(cur_trans))
        travel_time[f"{len(bus_route[:-1])-idx}->{len(bus_route[:-1])-idx-1}"] = distance / bus_forward_speed + 1 # + 1 because we guarantee a turn of 1s everytime
        cur_trans = next_trans
    curr_stop_index = 0
    for next_stop_index in bus_stop_indices[1:]:
        travel_time[f"{curr_stop_index}->{next_stop_index}"] = calculate_time_between_stops(travel_time, curr_stop_index, next_stop_index)
        curr_stop_index = next_stop_index
    curr_stop_index = bus_stop_indices[-1]
    for next_stop_index in reversed(bus_stop_indices[:-1]):
        travel_time[f"{curr_stop_index}->{next_stop_index}"] = calculate_time_between_stops(travel_time, curr_stop_index, next_stop_index)
        curr_stop_index = next_stop_index
    return travel_time

def get_schedule(bus_travel_time, bus_start_time, bus_end_time, bus_arrival_interval, bus_stop_indices, bus_stop_names, bus_stop_positions, bus_stop_rads, stop_time=30):
    start_time = datetime.datetime.strptime(bus_start_time, "%H:%M:%S")
    end_time = datetime.datetime.strptime(bus_end_time, "%H:%M:%S")
    arrival_interval = datetime.timedelta(minutes=bus_arrival_interval)
    schedule = []
    schedule_reversed = []
    for i, stop_index in enumerate(bus_stop_indices):
        stop_schedule = {
            'stop_name': bus_stop_names[i],
            'stop_pos': bus_stop_positions[i],
            'stop_rot': bus_stop_rads[i],
            'arrival_times': [],
            'departure_times': []
        }
        schedule.append(copy.deepcopy(stop_schedule))
        schedule_reversed.append(copy.deepcopy(stop_schedule))
    current_arrival_time = start_time
    while current_arrival_time <= end_time:
        this_time = current_arrival_time
        for i in range(len(bus_stop_indices)-1):
            schedule[i]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
            this_time += datetime.timedelta(seconds=stop_time)
            schedule[i]['departure_times'].append(this_time.strftime("%H:%M:%S"))
            this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i]}->{bus_stop_indices[i+1]}"]))
        schedule[len(bus_stop_indices)-1]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
        schedule_reversed[len(bus_stop_indices)-1]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
        this_time += datetime.timedelta(seconds=stop_time)
        schedule[len(bus_stop_indices)-1]['departure_times'].append(this_time.strftime("%H:%M:%S"))
        schedule_reversed[len(bus_stop_indices)-1]['departure_times'].append(this_time.strftime("%H:%M:%S"))
        for i in range(len(bus_stop_indices)-2, 0, -1):
            this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i+1]}->{bus_stop_indices[i]}"]))
            schedule_reversed[i]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
            this_time += datetime.timedelta(seconds=stop_time)
            schedule_reversed[i]['departure_times'].append(this_time.strftime("%H:%M:%S"))
        this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[1]}->{bus_stop_indices[0]}"]))
        schedule_reversed[0]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
        current_arrival_time += arrival_interval
    return schedule, schedule_reversed

def get_simplified_schedule(bus_travel_time, bus_start_time, bus_end_time, bus_arrival_interval, bus_stop_indices, bus_stop_names, stop_time=30):
    start_time = datetime.datetime.strptime(bus_start_time, "%H:%M:%S")
    end_time = datetime.datetime.strptime(bus_end_time, "%H:%M:%S")
    arrival_interval = datetime.timedelta(minutes=bus_arrival_interval)
    schedule = []
    schedule_reversed = []
    schedule_departure_times = []
    schedule_reversed_departure_times = []
    schedule_arrival_times = []
    schedule_reversed_arrival_times = []

    for i in range(len(bus_stop_indices)):
        stop_schedule = {
            'stop_name': bus_stop_names[i],
        }
        schedule.append(copy.deepcopy(stop_schedule))
        schedule_reversed.append(copy.deepcopy(stop_schedule))
        schedule_departure_times.append([])
        schedule_reversed_departure_times.append([])
        schedule_arrival_times.append([])
        schedule_reversed_arrival_times.append([])
    current_arrival_time = start_time
    while current_arrival_time <= end_time:
        this_time = current_arrival_time
        for i in range(len(bus_stop_indices)-1):
            schedule_arrival_times[i].append(this_time.strftime("%H:%M"))
            this_time += datetime.timedelta(seconds=stop_time)
            schedule_departure_times[i].append(this_time.strftime("%H:%M"))
            this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i]}->{bus_stop_indices[i+1]}"]))
        schedule_arrival_times[len(bus_stop_indices)-1].append(this_time.strftime("%H:%M"))
        schedule_reversed_arrival_times[len(bus_stop_indices)-1].append(this_time.strftime("%H:%M"))
        this_time += datetime.timedelta(seconds=stop_time)
        schedule_departure_times[len(bus_stop_indices)-1].append(this_time.strftime("%H:%M"))
        schedule_reversed_departure_times[len(bus_stop_indices)-1].append(this_time.strftime("%H:%M"))
        for i in range(len(bus_stop_indices)-2, 0, -1):
            this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i+1]}->{bus_stop_indices[i]}"]))
            schedule_reversed_arrival_times[i].append(this_time.strftime("%H:%M"))
            this_time += datetime.timedelta(seconds=stop_time)
            schedule_reversed_departure_times[i].append(this_time.strftime("%H:%M"))
        this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[1]}->{bus_stop_indices[0]}"]))
        schedule_reversed_arrival_times[0].append(this_time.strftime("%H:%M"))
        current_arrival_time += arrival_interval
    # print(schedule_departure_times)
    # print(schedule_reversed_departure_times)
    for i in range(len(bus_stop_indices)):
        schedule[i]["frequency"] = bus_arrival_interval
        schedule[i]["first_bus"] = schedule_arrival_times[i][0]
        schedule[i]["last_bus"] = schedule_arrival_times[i][-1]
        schedule_reversed[i]["frequency"] = bus_arrival_interval
        schedule_reversed[i]["first_bus"] = schedule_reversed_arrival_times[i][0]
        schedule_reversed[i]["last_bus"] = schedule_reversed_arrival_times[i][-1]
    return schedule, schedule_reversed, schedule_arrival_times, schedule_reversed_arrival_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--num_bus_clusters", type=int, default=3)
    parser.add_argument("--num_bicycle_clusters", type=int, default=7)
    parser.add_argument("--init_bus_road_min_length", type=float, default=100.0)
    parser.add_argument("--remove_bus_road_constraint", action="store_true")
    parser.add_argument("--reduce_bicycle_road_constraint", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--only_update_metadata", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    if not args.only_update_metadata:
    
        transit = {}
        
        with open(f"{ASSETS_PATH}/scenes/{args.scene}/road_data/roads.pkl", 'rb') as file:
            roads_geodetic, nodes = pickle.load(file)
        scene_range_meta = {}
        with open(f"{ASSETS_PATH}/scenes/{args.scene}/raw/center.txt") as f:
            scene_range_meta["lat"], scene_range_meta["lng"] = map(float, f.read().strip().split(' '))
        
        roads = []
        for road in roads_geodetic:
            temp_road = copy.deepcopy(road)
            temp_road["start"]["pos"] = list(pm.geodetic2enu(temp_road["start"]["lat"], temp_road["start"]["lon"], 0, scene_range_meta["lat"], scene_range_meta["lng"], 0))[:2]
            temp_road["end"]["pos"] = list(pm.geodetic2enu(temp_road["end"]["lat"], temp_road["end"]["lon"], 0, scene_range_meta["lat"], scene_range_meta["lng"], 0))[:2]
            roads.append(temp_road)
        # print(roads[0])

        all_highway_types = set()
        highway_types_stats = defaultdict(int)
        for i, road in enumerate(roads):
            all_highway_types.add(road['highway'])
            highway_types_stats[road['highway']] += 1

        print("Highway Type Stats:", highway_types_stats)

        place_metadata_path = f"{ASSETS_PATH}/scenes/{args.scene}/place_metadata.json"
        place_metadata = json.load(open(place_metadata_path, 'r'))

        all_places = []
        for place_name in place_metadata:
            all_places.append(place_metadata[place_name]["location"][:2])
        
        # First find known places clusters for bus route (find around 3 clusters, you can tune this number)
        places_posinates = np.array(all_places)
        kmeans_for_bus_route = KMeans(n_clusters=args.num_bus_clusters, random_state=args.seed).fit(places_posinates)
        clusters_for_bus_route = list(kmeans_for_bus_route.cluster_centers_)

        # bus_allowed_highway_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'service', 'living_street', 'path']
        bus_not_allowed_highway_types = ['pedestrian', 'footway', 'cycleway']
        bus_allowed_highway_types = list(all_highway_types - set(bus_not_allowed_highway_types))
        if args.remove_bus_road_constraint:
            bus_allowed_highway_types = all_highway_types
        if args.reduce_bicycle_road_constraint:
            bus_not_allowed_highway_types = ['pedestrian', 'footway', 'cycleway', 'residential', 'service']

        # Then find the closest road to each cluster center
        cluster_roads_for_bus_route = []
        past_road_names = []
        for cluster in clusters_for_bus_route:
            min_distance = float('inf')
            closest_road = None
            for road in roads:
                if not road['name'] in past_road_names:
                    distance, closest_point, width = point_to_road_distance(cluster, road, scene_range_meta["lat"], scene_range_meta["lng"])
                    if distance < min_distance and road['highway'] in bus_allowed_highway_types and np.linalg.norm(np.array(road['start']['pos']) - np.array(road['end']['pos'])) > args.init_bus_road_min_length:
                        min_distance = distance
                        closest_road = road
            if closest_road is None:
                print("No road found for cluster. Exiting... Try change the number of clusters for bus route, lower init_bus_road_min_length, or add remove_bus_road_constraint.")
                draw_roads(roads)
                plt.plot(np.array(clusters_for_bus_route)[:, 0], np.array(clusters_for_bus_route)[:, 1], 'ro', markersize=3)
                for idx, cluster_road in enumerate(cluster_roads_for_bus_route):
                    draw_line_segment(cluster_road['start']['pos'], cluster_road['end']['pos'], cluster_road['width'], fill_color='green', line_color='green')
                plt.show()
                exit()
            cluster_roads_for_bus_route.append(closest_road)
            past_road_names.append(closest_road['name'])
        # print(cluster_roads)
        # print(len(cluster_roads))

        # Lastly, connect each cluster road to form a route
        
        # Try finding a complete bus route
        best_route = None
        best_permutation_bus_roads = None
        min_route_length = int(1e10)
        for perm_roads in tqdm(permutations(cluster_roads_for_bus_route), total=factorial(len(cluster_roads_for_bus_route))):
            route = find_bus_route(roads, perm_roads, bus_allowed_highway_types)
            route_length = calculate_route_length(route)
            if route is not None and route_length < min_route_length:
                best_route = route
                best_permutation_bus_roads = perm_roads
                min_route_length = route_length

        if best_route is None:
            print("No valid bus route found. Exiting... Try change the number of clusters for bus route, lower init_bus_road_min_length, or add remove_bus_road_constraint.")
            draw_roads(roads)
            plt.plot(np.array(clusters_for_bus_route)[:, 0], np.array(clusters_for_bus_route)[:, 1], 'ro', markersize=3)
            for idx, cluster_road in enumerate(cluster_roads_for_bus_route):
                draw_line_segment(cluster_road['start']['pos'], cluster_road['end']['pos'], cluster_road['width'], fill_color='green', line_color='green')
            plt.show()
            exit()

        # print("best_permutation_bus_roads:", best_permutation_bus_roads)

        best_permutation_indices = [cluster_roads_for_bus_route.index(road) for road in best_permutation_bus_roads]
        clusters_for_bus_route = [clusters_for_bus_route[i] for i in best_permutation_indices]

        bus_stop_positions = []
        waypoints_sep_low = change_waypoints_sep(best_route, separation=1)
        starting_point = waypoints_sep_low[0]
        ending_point = waypoints_sep_low[-1]
        # plt.plot(ending_point[0], ending_point[1], 'b*')
        starting_bus_stop_position = get_bus_stop_from_road_info_and_waypoint(best_permutation_bus_roads[0], starting_point[0], starting_point[1])
        ending_bus_stop_position = get_bus_stop_from_road_info_and_waypoint(best_permutation_bus_roads[-1], ending_point[0], ending_point[1])
        bus_stop_positions.append(starting_bus_stop_position)
        starting_cluster = find_closest_cluster_from_waypoint(starting_bus_stop_position, clusters_for_bus_route)
        ending_cluster = find_closest_cluster_from_waypoint(ending_bus_stop_position, clusters_for_bus_route)
        other_clusters = [list(cluster) for cluster in clusters_for_bus_route if not (cluster==starting_cluster).all() and not (cluster==ending_cluster).all()]
        for idx, cluster in enumerate(other_clusters):
            closest_waypoint_to_cluster = find_closest_waypoint_to_cluster_from_waypoints(waypoints_sep_low, cluster)
            bus_stop_position = get_bus_stop_from_road_info_and_waypoint(best_permutation_bus_roads[idx+1], closest_waypoint_to_cluster[0], closest_waypoint_to_cluster[1])
            bus_stop_positions.append(bus_stop_position)
        bus_stop_positions.append(ending_bus_stop_position)
        waypoints_sep_high_including_bus_stops = get_sparse_points_from_dense_and_stops(waypoints_sep_low, bus_stop_positions, min_distance=100)
        refined_waypoints = remove_very_close_waypoints(waypoints_sep_high_including_bus_stops, filter_distance=10)
        transit["bus"] = {}
        transit["bus"]["refined_waypoints"] = refined_waypoints
        bus_stops_indices = find_closest_waypoints_near_bus_stops(refined_waypoints, bus_stop_positions).tolist()
        transit["bus"]["stop_indices"] = bus_stops_indices
        bus_stops = {}
        for i, bus_stop_position in enumerate(bus_stop_positions):
            # Here, we made sure that bus_stop_positions are in-order, if anything goes wrong on the stop name, investigate here
            # print(best_permutation_bus_roads[i])
            bus_stop_name = best_permutation_bus_roads[i]["name"] + " Bus Stop"
            bus_stops[bus_stop_name] = {}
            stop_position_2d = bus_stop_position
            corresponding_bus_waypoint_2d = refined_waypoints[bus_stops_indices[i]]
            target_rad = np.arctan2(corresponding_bus_waypoint_2d[1] - stop_position_2d[1], corresponding_bus_waypoint_2d[0] - stop_position_2d[0])
            bus_stops[bus_stop_name] = {"position": stop_position_2d, "target_rad": target_rad}
        transit["bus"]["stops"] = bus_stops
        transit["bus"]["start_time"] = "06:00:00"
        transit["bus"]["end_time"] = "22:00:00"
        transit["bus"]["forward_speed"] = 10 # 10m/s
        transit["bus"]["rotation_speed"] = 360 # 360 degrees/s
        transit["bus"]["frequency"] = 15 # must be in minutes
        transit["bus"]["travel_time"] = calculate_travel_time(transit["bus"]["forward_speed"], 
                                                                transit["bus"]["refined_waypoints"], 
                                                                transit["bus"]["stop_indices"])
        schedule, schedule_reversed, _, _ = get_simplified_schedule(transit["bus"]["travel_time"], 
                    transit["bus"]["start_time"], 
                    transit["bus"]["end_time"], 
                    transit["bus"]["frequency"], 
                    transit["bus"]["stop_indices"], 
                    list(transit["bus"]["stops"].keys()))
        
        # transit["bus"]["schedule"] = schedule
        # transit["bus"]["schedule_reversed"] = schedule_reversed

        transit["bus"]["schedule"] = {}
        transit["bus"]["schedule"][f"To {schedule[-1]['stop_name'][:-9]}"] = schedule
        transit["bus"]["schedule"][f"To {schedule[0]['stop_name'][:-9]}"] = schedule_reversed[::-1]

        # Find bicycle stations on roads of bus_not_allowed_highway_types
        bicycle_station_roads = []
        for road in roads:
            if road['id'] not in [iter_road['id'] for iter_road in best_permutation_bus_roads] and road['highway'] in bus_not_allowed_highway_types:
                bicycle_station_roads.append(road)
        if len(bicycle_station_roads) == 0:
            print("No bicycle_station roads found. Exiting...")
            exit()
        
        kmeans_for_bicycle_stations = KMeans(n_clusters=args.num_bicycle_clusters, random_state=args.seed).fit(places_posinates)
        clusters_for_bicycle_stations = list(kmeans_for_bicycle_stations.cluster_centers_)

        # remove clusters that are too close to bus stops
        # clusters_for_bicycle_stations = [cluster for cluster in clusters_for_bicycle_stations if all([np.linalg.norm(np.array(cluster) - np.array(bus_stop_position)) > 100 for bus_stop_position in bus_stop_positions])]

        bicycle_stations = {}
        for cluster in clusters_for_bicycle_stations:
            min_distance = float('inf')
            closest_closest_point = None
            closest_road = None
            for road in bicycle_station_roads:
                distance, closest_point, width = point_to_road_distance(cluster, road, scene_range_meta["lat"], scene_range_meta["lng"])
                if distance < min_distance:
                    min_distance = distance
                    closest_closest_point = closest_point
                    closest_road = road
            
            # do the same thing as get_bus_stop_from_road_info_and_waypoint:
            # calculate the bicycle station position by extending the waypoint towards the normal direction of the road by half of the width (+0.5m by default, if it collides with the building, consider removing this) so that the bus stop is on the road side, not middle.
            road_vector = np.array(closest_road['end']['pos']) - np.array(closest_road['start']['pos'])
            road_vector /= np.linalg.norm(road_vector)
            normal_vector = np.array([-road_vector[1], road_vector[0]])
            new_bicycle_station_position = np.array(closest_closest_point) + (width / 2 + 0.5) * normal_vector

            # if too close to previous bicycle station, skip, or too close to previous bus stop, skip
            if all([np.linalg.norm(np.array(new_bicycle_station_position) - np.array(bicycle_station_position)) > 50 for bicycle_station_position in bicycle_stations.values()]) and \
                all([np.linalg.norm(np.array(new_bicycle_station_position) - np.array(bus_stop_position)) > 50 for bus_stop_position in bus_stop_positions]):
                bicycle_stations[f"Bicycle Sharing Station {len(bicycle_stations) + 1 if bicycle_stations else 1}"] = new_bicycle_station_position.tolist()
        
        transit["bicycle"] = {}
        transit["bicycle"]["stations"] = bicycle_stations

        json.dump(transit, open(f"{ASSETS_PATH}/scenes/{args.scene}/transit.json", 'w'), indent=4)

        draw_roads(roads)
        # for position in all_places:
        #     plt.plot(position[0], position[1], 'ko', markersize=2)
        # plt.plot(np.array(clusters_for_bus_route)[:, 0], np.array(clusters_for_bus_route)[:, 1], 'ro', markersize=3)
        # plt.plot(np.array(clusters_for_bicycle_stations)[:, 0], np.array(clusters_for_bicycle_stations)[:, 1], 'yo', markersize=3)
        # for idx, cluster_road in enumerate(cluster_roads_for_bus_route):
        #     draw_line_segment(cluster_road['start']['pos'], cluster_road['end']['pos'], cluster_road['width'], fill_color='green', line_color='green')
        for bus_stop_position in bus_stop_positions:
            plt.plot(bus_stop_position[0], bus_stop_position[1], 'bo', markersize=3, label='Bus Stop')
        for bicycle_station_position in bicycle_stations.values():
            plt.plot(bicycle_station_position[0], bicycle_station_position[1], 'mo', markersize=3, label='Bicycle Station')
        
        if not args.animation:
            plt.plot(np.array(transit["bus"]["refined_waypoints"])[:, 0], np.array(transit["bus"]["refined_waypoints"])[:, 1], 'b-', label='Bus Route')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            if args.remove_bus_road_constraint:
                plt.title('Transit Annotation Mode: No Bus Road Constraint')
            else:
                plt.title('Transit Annotation Mode: Normal')
            plt.savefig(f"{ASSETS_PATH}/scenes/{args.scene}/transit.png")
            plt.show()
        else:
            # base_path = f"{ASSETS_PATH}/scenes/{args.scene}"
            # os.makedirs(f'{base_path}/animation', exist_ok=True)
            for i in range(0, len(refined_waypoints)):
                temp_way_points_enu = np.array(refined_waypoints[:i+1])
                plt.plot(temp_way_points_enu[:, 0], temp_way_points_enu[:, 1], 'b-', label='Bus Route')
                plt.axis('off')
                plt.pause(1)
                plt.title('Vis')
                # plt.legend()
                plt.draw()
                # plt.savefig(f'{base_path}/animation/frame_{i:04d}.png')
        
            
            # with imageio.get_writer(f'{base_path}/animation/frame_*.png', fps=10) as writer:
            #     for filename in sorted(glob.glob(f'{base_path}/animation/frame_*.png')):
            #         image = imageio.imread(filename)
            #         writer.append_data(image)
    else:   
        transit = json.load(open(f"{ASSETS_PATH}/scenes/{args.scene}/transit.json", 'r'))
        place_metadata_path = f"{ASSETS_PATH}/scenes/{args.scene}/place_metadata.json"
        place_metadata = json.load(open(place_metadata_path, 'r'))

    # Update place metadata and building metadata
    building_metadata_path = f"{ASSETS_PATH}/scenes/{args.scene}/building_metadata.json"
    building_metadata = json.load(open(building_metadata_path, 'r'))

    # clean previous transit data
    for place_name in place_metadata.copy():
        if place_metadata[place_name]["coarse_type"] == "transit":
            del place_metadata[place_name]
    if "open space" not in building_metadata:
        building_metadata["open space"] = {}
        building_metadata["open space"]["bounding_box"] = None
        building_metadata["open space"]["places"] = []
    for place in building_metadata["open space"]["places"].copy():
        if place["coarse_type"] == "transit":
            building_metadata["open space"]["places"].remove(place)
    
    for bus_stop in transit["bus"]["stops"]:
        place_metadata[bus_stop] = {
            "building": "open space",
            "coarse_type": "transit",
            "fine_type": "bus stop",
            "location": transit["bus"]["stops"][bus_stop]["position"],
            "scene": None,
        }
        building_metadata["open space"]["places"].append({
            "name": bus_stop,
            "coarse_type": "transit",
            "fine_type": "bus stop",
            "location": transit["bus"]["stops"][bus_stop]["position"],
        })
    
    for bicycle_station in transit["bicycle"]["stations"]:
        place_metadata[bicycle_station] = {
            "building": "open space",
            "coarse_type": "transit",
            "fine_type": "bicycle station",
            "location": transit["bicycle"]["stations"][bicycle_station],
            "scene": None,
        }
        building_metadata["open space"]["places"].append({
            "name": bicycle_station,
            "coarse_type": "transit",
            "fine_type": "bicycle station",
            "location": transit["bicycle"]["stations"][bicycle_station],
        })

    json.dump(place_metadata, open(place_metadata_path, 'w'), indent=4)
    json.dump(building_metadata, open(building_metadata_path, 'w'), indent=4)
    print(f"[Transit Annotation] Place metadata and building metadata updated for {args.scene}.")