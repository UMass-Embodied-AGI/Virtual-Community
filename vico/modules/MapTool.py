import ast
import os
import random
import copy
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import re
from enum import Enum
import time
import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from PIL import Image
import argparse

from ..tools.road_annotation.visualize_osm_roads import draw_roads
from ..tools.constants import ASSETS_PATH

if __name__ != "__main__" :
    from ..tools.utils import *
    from . import *

def is_point_enclosed_MapTool(grid, point, resolution, min_x, min_y, nx, ny):
    from collections import deque

    i = int((point[0] - min_x) / resolution)
    j = int((point[1] - min_y) / resolution)

    if i < 0 or i >= nx or j < 0 or j >= ny:
        # print("Point is out of bounds")
        return True, (i, j) # not valid

    if grid[i, j] == 1:
        # print("Point is inside an obstacle")
        return True, (i, j) # not valid
    else:
        return False, (i, j)

    visited = np.zeros_like(grid, dtype=bool)
    queue = deque()
    queue.append((i, j))
    visited[i, j] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if x == 0 or x == nx - 1 or y == 0 or y == ny - 1:
            # print("Point is not enclosed")
            return False, (i, j)
        for dx, dy in directions:
            nx_ = x + dx
            ny_ = y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                if not visited[nx_, ny_] and grid[nx_, ny_] == 0:
                    visited[nx_, ny_] = True
                    queue.append((nx_, ny_))
    # print("Point is enclosed")
    return True, (i, j)

@dataclass
class Waypoints:
    id: int
    name: str | None = None
    location: list[float, float] | None = None
    belong: str | None = None
    predecessor: list = field(default_factory=list)
    successor: list = field(default_factory=list)
    property: dict = field(default_factory=dict)

    def is_a_bus_stop(self):
        return "bus_stop_id" in self.property
    
@dataclass
class RouteNode:
    location: list[float, float] | None = None
    transit: str | None = None
    eta: datetime | None = None

    def to_dict(self):
        return {
            "location": self.location,
            "transit": self.transit,
            "eta": datetime.strftime(self.eta, '%H:%M:%S')
        }

class Route:
    def __init__(self, nodes=None, impossible=False):
        '''
        waypoints: list(np.array([int,int]))
        '''
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        self.impossible = impossible

    def __getitem__(self, key):
        if isinstance(key, int):  # Single index
            return self.nodes[key]
        elif isinstance(key, slice):  # Slice → return a new Route
            return Route(self.nodes[key])
        else:
            raise TypeError("Invalid argument type.")
    
    def __len__(self):
        return len(self.nodes)
    
    def empty(self):
        return not self.nodes
    
    def append(self, node: RouteNode):
        self.nodes.append(node)

    def extend(self, route):
        self.nodes.extend(route.nodes)

    def pop(self, idx):
        self.nodes.pop(idx)

    def reverse(self):
        self.nodes.reverse()

    def calc_time(self, pose=None):
        if self.impossible:
            return 24*60*60-1
        if pose is not None:
            ret=np.linalg.norm(np.array(self.nodes[0].location[:2])-np.array(pose[:2]))/(5.0 if self.nodes[0].transit=='bus' else 1.0)
        else:
            ret=0
        for i in range(1, len(self.nodes)):
            ret+=np.linalg.norm(np.array(self.nodes[i].location[:2])-np.array(self.nodes[i-1].location[:2]))/(5.0 if self.nodes[i].transit=='bus' else 1.0)
        return ret*2 # for turning
    
    def to_dict(self):
        return [node.to_dict() for node in self.nodes]
    
    def simplify(self):
        new_nodes=[]
        i=0
        while i<len(self.nodes):
            j=len(self.nodes)
            while j>i:
                j-=1
                if np.linalg.norm(np.array(self.nodes[j].location)-np.array(self.nodes[i].location))<=7:
                    break
            new_nodes.append(self.nodes[j])
            i=j+1
        self.nodes=new_nodes
    
def find_next_bus_times(current_stop, current_time, schedule, schedule_reversed):
    """
    Given the current stop and time, return the nearest reachable times
    for each stop (same index across stops), checking both schedule directions.
    
    Returns:
        dict of stop -> arrival time OR None if no buses available.
    """

    def get_time_from_str(string):
        return datetime.combine(current_time.date(), datetime.strptime(string, "%H:%M:%S").time())

    def get_next_index(schedule_variant):
        """Return (index, schedule_variant) of next bus if available, else (None, None)."""
        arrivals = schedule_variant[current_stop]["departure_times"]
        for i, t_str in enumerate(arrivals):
            t = get_time_from_str(t_str)
            if t >= current_time:
                return i, schedule_variant
        return None, None

    # Check forward and reverse
    idx_fwd, sch_fwd = get_next_index(schedule)
    idx_rev, sch_rev = get_next_index(schedule_reversed)

    # Pick the earlier valid bus (if both exist)
    chosen_index, chosen_schedule = None, None
    if idx_fwd is not None and idx_rev is not None:
        t_fwd = get_time_from_str(schedule[current_stop]["arrival_times"][idx_fwd])
        # if t_fwd<current_time:
        #     if idx_fwd+1<len(schedule[current_stop]["arrival_times"]):
        #         t_fwd = datetime.strptime(schedule[current_stop]["arrival_times"][idx_fwd+1], "%H:%M:%S").time()
        #     else:
        #         t_fwd = None
        t_rev = get_time_from_str(schedule_reversed[current_stop]["arrival_times"][idx_rev])
        # if t_rev<current_time:
        #     if idx_rev+1<len(schedule[current_stop]["arrival_times"]):
        #         t_rev = datetime.strptime(schedule[current_stop]["arrival_times"][idx_rev+1], "%H:%M:%S").time()
        #     else:
        #         t_rev = None
        if t_fwd <= t_rev:
            chosen_index, chosen_schedule = idx_fwd, schedule
        else:
            chosen_index, chosen_schedule = idx_rev, schedule_reversed
    elif idx_fwd is not None:
        chosen_index, chosen_schedule = idx_fwd, schedule
    elif idx_rev is not None:
        chosen_index, chosen_schedule = idx_rev, schedule_reversed
    else:
        # No buses left today
        return {stop: None for stop in range(len(schedule))}

    # Build result
    result = {}
    for stop, times in enumerate(chosen_schedule):
        result[stop] = get_time_from_str(times["arrival_times"][chosen_index])
        if result[stop]<current_time:
            if stop==current_stop:
                result[stop]=current_time
            else:
                if chosen_schedule==schedule:
                    result[stop]=get_time_from_str(schedule_reversed[stop]["arrival_times"][chosen_index])
                else:
                    result[stop]=get_time_from_str(schedule[stop]["arrival_times"][chosen_index+1])

    return result


class MapTool:
    '''walkers only'''
    def __init__(self, scene_name=None, pose=None, place_metadata=None, building_metadata=None, bus=None, waypoints_dis=7., logger=None):
        init_time = time.perf_counter()
        self.scene_name=scene_name
        self.pose=pose
        self.covered_length=0.
        self.place_metadata=deepcopy(place_metadata)
        self.building_metadata=deepcopy(building_metadata)
        self.waypoints_dis=waypoints_dis

        self.roads, self.nodes = pickle.load(open(f"{ASSETS_PATH}/scenes/{scene_name}/road_data/roads.pkl", 'rb'))
        # Paths
        img_path = f"{ASSETS_PATH}/scenes/{self.scene_name}/global.png"
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Aerial image not found: {img_path}")
        self.global_image = Image.open(img_path).convert("RGB")

        obstacle_grid_save = pickle.load(open(f"{ASSETS_PATH}/scenes/{scene_name}/obstacle_grid.pkl", 'rb'))
        self.obstacle_grid = obstacle_grid_save["grid"]
        self.obstacle_grid_parameters = obstacle_grid_save["parameters"]

        self.waypoints = []
        self.road2waypoint_ids = {}
        self.spawn_waypoints()

        self.logger = logger
        self.grid_map = None
        self.clipped_grid_map = None
        init_time = time.perf_counter() - init_time
        if self.logger is not None:
            self.logger.info(f"MapTool initialized in {init_time}s")
        else:
            print(f"MapTool initialized in {init_time}s")
        
    def reset(self, pose):
        self.pose=pose
        self.covered_length=0.

    def is_point_invalid(self, point, lim=None):
        if lim is None:
            lim = self.waypoints_dis
        return all([is_point_enclosed_MapTool(grid=self.obstacle_grid, point=point+shift, resolution=self.obstacle_grid_parameters["resolution"], min_x=self.obstacle_grid_parameters["min_x"], min_y=self.obstacle_grid_parameters["min_y"], nx=self.obstacle_grid_parameters["nx"], ny=self.obstacle_grid_parameters["ny"])[0] for shift in [np.array([i, j]) for i in range(-int(lim),int(lim)+1) for j in range(-int(lim),int(lim)+1)]])

    def spawn_waypoints(self):
        for node in self.nodes:
            # if the point is invalid
            if self.is_point_invalid([self.nodes[node]['x'], self.nodes[node]['y']]): continue
            # processing node
            for road in self.nodes[node]["connected_roads"]:
                if road not in self.road2waypoint_ids:
                    self.road2waypoint_ids[road]=[]
                self.road2waypoint_ids[road].append(len(self.waypoints))
            self.nodes[node]["2wp"]=len(self.waypoints)
            self.waypoints.append(Waypoints(id=len(self.waypoints), location=[self.nodes[node]['x'], self.nodes[node]['y']], belong=None if not self.nodes[node]["connected_roads"] else self.nodes[node]["connected_roads"][0]))
        for road in self.roads:
            # spawn points on road
            start_x, start_y = road['start']['x'],road['start']['y']
            end_x, end_y = road['end']['x'],road['end']['y']
            length = np.linalg.norm(np.array([start_x-end_x, start_y-end_y]))
            s=self.waypoints_dis
            # if the start point is invalid
            if '2wp' not in self.nodes[road['start']['id']]:
                last_waypoint = None
            else:
                last_waypoint=self.waypoints[self.nodes[road['start']['id']]['2wp']]
            while s<length:
                p = np.array([start_x, start_y]) - s/length*np.array([start_x-end_x, start_y-end_y])
                # self.nodes[f"new_node_{len(self.nodes)}"]={"x": p[0], "y": p[1], "connected_roads": road["id"]}
                if self.is_point_invalid(p):
                    last_waypoint = None
                    s+=self.waypoints_dis
                    continue
                new_wp=(Waypoints(id=len(self.waypoints), location=p, belong=road["id"]))
                self.waypoints.append(new_wp)
                if road['id'] not in self.road2waypoint_ids:
                    self.road2waypoint_ids[road['id']]=[]
                self.road2waypoint_ids[road['id']].append(len(self.waypoints))
                if last_waypoint is not None:
                    last_waypoint.successor.append(new_wp.id)
                last_waypoint = new_wp
                s+=self.waypoints_dis
            if '2wp' in self.nodes[road['end']['id']] and last_waypoint is not None:
                last_waypoint.successor.append(self.waypoints[self.nodes[road['end']['id']]['2wp']].id)
        for waypoint in self.waypoints:
            for successor in waypoint.successor:
                self.waypoints[successor].predecessor.append(waypoint.id)
        # for low-connected waypoints, search its neighbour
        for idx, waypoint in enumerate(self.waypoints):
            # if len(waypoint.successor)+len(waypoint.predecessor)>2: continue
            for jdx, n_wp in enumerate(self.waypoints):
                if jdx==idx:continue
                # if len(waypoint.successor)+len(waypoint.predecessor)>2: break
                if np.linalg.norm(np.array(waypoint.location)-np.array(n_wp.location))<self.waypoints_dis:
                    self.waypoints[idx].successor.append(jdx)
                    self.waypoints[jdx].predecessor.append(idx)

    def initiate_transit(self, bus):
        self.bus_stop_to_waypoint=dict()
        self.bus=bus
        for bus_wp_id, r_wp in enumerate(self.bus.route):
            new_wp=Waypoints(id=len(self.waypoints), location=r_wp, belong=None, predecessor=[], successor=[], property={})
            self.waypoints.append(new_wp)
            if bus_wp_id in self.bus.stop_indices:
                new_wp.property["bus_stop_id"]=self.bus.stop_indices.index(bus_wp_id)
                self.bus_stop_to_waypoint[self.bus.stop_names[new_wp.property["bus_stop_id"]]]=new_wp.id
            for jdx, n_wp in enumerate(self.waypoints):
                if jdx==new_wp.id:continue
                if np.linalg.norm(np.array(new_wp.location)-np.array(n_wp.location))<self.waypoints_dis:
                    self.waypoints[new_wp.id].successor.append(jdx)
                    self.waypoints[jdx].predecessor.append(new_wp.id)

    def get_pose(self):
        return self.pose
    
    def reset_covered_length(self):
        self.covered_length=0.

    def set_pose(self, pose):
        self.covered_length+=np.linalg.norm(pose[:2]-self.pose[:2])
        self.pose=pose

    def query_place(self, place_name):
        # one place at one time to simulate time cost
        if place_name not in self.place_metadata: return None
        knowledge = copy.deepcopy(self.place_metadata[place_name])
        knowledge["bounding_box"]=self.building_metadata[knowledge['building']]['bounding_box']
        knowledge_items={place_name: knowledge}
        return knowledge_items
    
    def query_nearby(self, target_pos, threshold=30):
        assert type(threshold) != str
        places_list=[]
        for place in self.place_metadata:
            if is_near_goal(target_pos[0], target_pos[1], self.building_metadata[self.place_metadata[place]['building']]['bounding_box'], self.place_metadata[place]['location'], threshold=threshold):
                places_list.append(place)
        return places_list
    
    def get_nearest_waypoints(self, curr_trans):
        """
        Find and return several nearest waypoint ids from the given curr_trans.
        """
        ret=[]
        start_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
        )
        min_dis2s = np.linalg.norm(np.array(self.waypoints[start_wp_id].location) - np.array(curr_trans[:2]))
        for i in range(len(self.waypoints)):
            if np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2])) <= min_dis2s+self.waypoints_dis:
                ret.append(i)
        return ret
    
    def query_route(self, curr_trans, goal_place=None, goal_trans=None, curr_time=datetime.strptime("6:00:00","%H:%M:%S")):
        """
        Find a route from current pose to the goal_place using waypoint graph.
        
        Args:
            curr_trans (list | np.ndarray): Current [x, y, ...] position of agent
            goal_place (str): Name of the destination place
            goal_trans (list | np.ndarray): The goal position [x, y, ...] in the same coordinate system as curr_trans

        Returns:
            list[Waypoints]: Ordered list of waypoints from current pose to goal
        """
        if not self.waypoints:
            raise ValueError("No waypoints available. Call spawn_waypoints() first.")

        # Get goal location
        if goal_place is not None and goal_place not in self.place_metadata:
            print(f"[MapTool.query_route] Unknown place: {goal_place!r}; returning None so the caller can handle it.")
            return None

        if goal_place is not None:
            goal_bbox = self.building_metadata[self.place_metadata[goal_place]['building']]['bounding_box']
            goal_pos = self.place_metadata[goal_place]['location'][:2]  # [x, y]
        else:
            goal_bbox = None
            goal_pos = goal_trans
        curr_trans=copy.deepcopy(curr_trans)
        if goal_pos[0]>500 or goal_pos[1]>500:
            goal_pos[0], goal_pos[1]=goal_pos[0]-1000, goal_pos[1]-1000
        if curr_trans[0]>500 or curr_trans[1]>500:
            curr_trans[0], curr_trans[1]=curr_trans[0]-1000, curr_trans[1]-1000

        # 1. Find nearest waypoint to current pose
        start_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
        )
        min_dis2s = np.linalg.norm(np.array(self.waypoints[start_wp_id].location) - np.array(curr_trans[:2]))

        # 2. Find nearest waypoint to goal location
        goal_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos))
        )
        min_dis2t = np.linalg.norm(np.array(self.waypoints[goal_wp_id].location) - np.array(goal_pos))

        # 3. Pathfinding: Dijkstra (or BFS if uniform cost) over waypoint graph
        # Using Dijkstra with distance as edge cost
        inf_time = datetime.combine(curr_time.date(), datetime.strptime("23:59:59", "%H:%M:%S").time())
        dist = {i: inf_time for i in range(len(self.waypoints))}
        prev = {i: None for i in range(len(self.waypoints))}
        heap = []
        for i in range(len(self.waypoints)):
            if np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2])) <= min_dis2s+self.waypoints_dis:
                dist[i] = curr_time+timedelta(seconds=np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2])))
                heapq.heappush(heap, (dist[i], i))

        while heap:
            d, wp_id = heapq.heappop(heap)
            if d > dist[wp_id]:
                continue
            if wp_id == goal_wp_id:
                break

            current_wp = self.waypoints[wp_id]
            potential = current_wp.successor + current_wp.predecessor
            for succ_id in potential:
                if succ_id >= len(self.waypoints):
                    continue
                succ_wp = self.waypoints[succ_id]
                cost = np.linalg.norm(
                    np.array(current_wp.location) - np.array(succ_wp.location)
                )
                new_dist = dist[wp_id] + timedelta(seconds=cost)
                if new_dist < dist[succ_id]:
                    dist[succ_id] = new_dist
                    prev[succ_id] = [wp_id, 'walk']
                    heapq.heappush(heap, (new_dist, succ_id))
            if current_wp.is_a_bus_stop():
                next_bus_time=find_next_bus_times(current_wp.property["bus_stop_id"], curr_time, self.bus.schedule, self.bus.schedule_reversed)
                for i in range(len(self.bus.stop_names)):
                    bus_wp_id=self.bus_stop_to_waypoint[self.bus.stop_names[i]]
                    if bus_wp_id == wp_id: continue
                    new_dist=next_bus_time[i]
                    if dist[wp_id]<new_dist<dist[bus_wp_id]:
                        dist[bus_wp_id]=new_dist
                        prev[bus_wp_id] = [wp_id, 'bus']
                        heapq.heappush(heap, (new_dist, bus_wp_id))

        # 4. Reconstruct path
        goal_wp_pair=(dist[goal_wp_id]+timedelta(seconds=min_dis2t), goal_wp_id)
        for i in range(len(self.waypoints)):
            if is_near_goal(self.waypoints[i].location[0], self.waypoints[i].location[1], goal_bbox, goal_pos):
                goal_wp_pair=min((dist[i], i), goal_wp_pair)
            elif np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos)) <= min_dis2t+self.waypoints_dis:
                goal_wp_pair=min((dist[i]+timedelta(seconds=np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos))), i), goal_wp_pair)
        if self.logger is not None:
            self.logger.info(f"found goal_wp_pair is {goal_wp_pair}")
        else:
            print(f"found goal_wp_pair is {goal_wp_pair}")
        if goal_wp_pair[0] >= inf_time:
            self.logger.warning(f"{self.scene_name}: No path found from {curr_trans[:2]} to {goal_place} at {goal_pos}")
            return Route(impossible=True)
        path = Route()
        curr = goal_wp_pair[1]
        while curr is not None:
            if prev[curr] is None:
                path.append(RouteNode(list(self.waypoints[curr].location), 'walk', dist[curr]))
                curr = None
            else:
                path.append(RouteNode(list(self.waypoints[curr].location), prev[curr][1], dist[curr]))
                curr = prev[curr][0]
        path.reverse()

        if not path:
            self.logger.warning(f"{self.scene_name}: No valid route found from {curr_trans[:2]} to {goal_place} at {goal_pos}")
            return Route(impossible=True)
        
        path.append(RouteNode(list(goal_pos), 'walk', goal_wp_pair[0]))
        return path
    
    def get_connected_waypoints(self, waypoint_id):
        """
        Find all waypoints connected to the given waypoint_id via successor/predecessor links.
        Performs BFS to collect all reachable waypoints in the graph.

        Returns:
            List[int]: List of waypoint IDs that are connected (including the start).
        """
        if waypoint_id >= len(self.waypoints) or waypoint_id < 0:
            return []

        visited = set()
        queue = [waypoint_id]
        visited.add(waypoint_id)

        while queue:
            current_id = queue.pop(0)
            current_wp = self.waypoints[current_id]

            # Traverse both successors and predecessors for full connectivity
            neighbors = current_wp.successor + current_wp.predecessor
            for neighbor_id in neighbors:
                if neighbor_id < len(self.waypoints) and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        return sorted(list(visited))
    
    def get_zoomed_scene_metadata(self, x_min, y_min, x_max, y_max):
        """
        Return a dict that contains all roads and buildings metadata within the area.
        """
        ret_road = dict()
        ret_building = dict()

        # As for roads
        for road in self.roads:
            start_x, start_y = road['start']['x'],road['start']['y']
            end_x, end_y = road['end']['x'],road['end']['y']
            if (x_min <= start_x <= x_max and y_min <= start_y <=y_max) or (x_min <= end_x <= x_max and y_min <= end_y <=y_max):
                ret_road[f"{road['name']} {road['id']}"]={'start': [road['start']['x'], road['start']['y']], 'end': [road['end']['x'], road['end']['y']]}

        # As for buildings
        for building in self.building_metadata:
            print(self.building_metadata[building])
            if building == 'open space':
                for place in self.building_metadata[building]['places']:
                    if (x_min <= place['location'][0] <= x_max and y_min <= place['location'][1] <=y_max):
                        ret_building[place['name']] = place
            else:
                p = self.building_metadata[building]['outdoor_xy']
                if (x_min <= p[0] <= x_max and y_min <= p[1] <=y_max):
                    ret_building[building]=self.building_metadata[building]
        
        return ret_road, ret_building
    
    def get_zoomed_scene_image(self, x_min, y_min, x_max, y_max, alpha=1.0):
        """
        Return a zoomed-in part of the whole scene image with road annotations
        as a PIL.Image.Image object.

        Args:
            scene_name (str): Scene name (e.g., "newyork", "detroit").
            x_min, y_min, x_max, y_max (float): Bounding box coordinates in world units
                (matching the coordinate extent used when displaying the image, e.g. [-512, 512]).
            ref_lat, ref_lon (float, optional): Reference latitude and longitude for projection.

        Returns:
            PIL.Image.Image: Cropped and annotated subimage.
        """
        # Paths
        img_path = f"{ASSETS_PATH}/scenes/{self.scene_name}/global.png"

        # Check assets
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Aerial image not found: {img_path}")

        # Load aerial image
        img = self.global_image
        width, height = img.size
        world_min, world_max = -512, 512

        # Create a Matplotlib figure with the zoomed-in extent
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        # Display the cropped portion of the aerial image
        ax.imshow(
            img,
            extent=[world_min, world_max, world_min, world_max],
            origin="upper"
        )

        # Draw roads
        draw_roads(self.roads, alpha=alpha)

        color_dict = {'primary': 'red', 'secondary': 'orangered', 'tertiary': 'orange', 'residential': 'gold', 
              'cycleway':'green', 'pedestrian': 'blue', 'footway': 'deepskyblue', 'service': 'peru', 
              'unclassified': 'm', 'steps': 'blue', 'elevator':'violet', 'living_street':'pink', 'construction': 'orchid'}

        legend_handles = [mpatches.Patch(color=color, label=road_type) for road_type, color in color_dict.items()]

        ax.legend(handles=legend_handles, title="Road Types", loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0, fontsize='small', title_fontsize='medium')
        plt.tight_layout(pad=0., rect=[0, 0, 1, 1])

        # Turn off axes and layout
        # ax.axis("off")
        # plt.tight_layout(pad=0)

        # Render figure to a PIL Image
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = buf.shape
        zoomed_img = buf[:, :, :3]  # drop alpha
        zoomed_img = Image.fromarray(zoomed_img)

        plt.close(fig)
        return zoomed_img
    
    def query_grid_map(self):
        if getattr(self, "grid_map", None) is not None:
            return self.grid_map
        # Define grid size and coordinate range
        grid_size = 50
        cell_size = 20
        coord_min, coord_max = -490, 490

        # Initialize with open space ('.')
        grid_map = np.full((grid_size, grid_size), '.', dtype=str)

        # Precompute coordinate offset
        offset = coord_min // cell_size  # = 49 for [-495, 495]

        # Iterate through coordinates
        for x in range(coord_min, coord_max + 1, cell_size):
            for y in range(coord_min, coord_max + 1, cell_size):
                i = (y // cell_size) - offset
                j = (x // cell_size) - offset
                # Skip if out of bounds
                if not (0 <= i < grid_size and 0 <= j < grid_size):
                    continue
                # Mark obstacle
                if self.is_point_invalid([x, y], lim=10):
                    grid_map[i, j] = 'X'

        self.grid_map = grid_map.tolist()
        return self.grid_map
    
    def get_grid_map_image(self, route_coords=None, circle_coords=None, agent_coords=None, target_coords=None, new_route_coords=None):
        return self.visualize_grid_with_objects(grid=self.obstacle_grid, resolution=self.obstacle_grid_parameters["resolution"], min_x=self.obstacle_grid_parameters["min_x"], min_y=self.obstacle_grid_parameters["min_y"], route_coords=route_coords, circle_coords=circle_coords, agent_coords=agent_coords, target_coords=target_coords, new_route_coords=new_route_coords)

    def visualize_grid_with_objects(self, grid, route_coords=None, circle_coords=None, agent_coords=None, target_coords=None, new_route_coords=None,
                                resolution=1.0, min_x=0, min_y=0):
        """
        Visualize occupancy grid with multiple types of external coordinates.

        Args:
            grid (np.ndarray): 2D array, 1 for obstacle (black), 0 for free (white)
            route_coords (list[(x, y)]): sequence of points forming a route
            circle_coords (list[(x, y)]): centers of circle markers
            agent_coords (list[(x, y)]): isolated points
            target_coords (list[(x, y)]): isolated points
            resolution (float): meters per grid cell
            min_x, min_y (float): world coordinates of grid[0, 0]
        """
        if self.clipped_grid_map is None:
            grid = deepcopy(grid)
            grid = grid[int((-400 - min_x)/resolution):int((400 - min_x)/resolution), int((-400 - min_y)/resolution):int((400 - min_y)/resolution)].T
            self.clipped_grid_map = grid
            min_x, min_y = -400, -400
            h, w = grid.shape
        else:
            grid = self.clipped_grid_map
            min_x, min_y = -400, -400
            h, w = grid.shape

        # --- Plot background grid ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap='gray_r', origin='lower',
                extent=[min_x, min_x + w * resolution,
                        min_y, min_y + h * resolution])

        # Helper: test if a coordinate is in free space
        def is_free(x, y):
            gx = int((x - min_x) / resolution)
            gy = int((y - min_y) / resolution)
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                return False
            return grid[gy, gx] == 0

        # --- 2️⃣ Circle-like coordinates ---
        if circle_coords is not None:
            for (x, y) in circle_coords:
                # if is_free(x, y):
                circle = Circle((x, y), radius=20, edgecolor='red',
                                facecolor='red', linewidth=1.5)
                ax.add_patch(circle)

        # --- 1️⃣ Route-like coordinates ---
        if route_coords is not None and len(route_coords) > 0:
            # self.logger.info(f"getting grid image, route_coords are {route_coords}")
            route_coords = np.array([p for p in route_coords if is_free(p[0], p[1])])
            if len(route_coords) > 0:
                ax.plot(route_coords[:, 0], route_coords[:, 1], c='blue', linewidth=1.5)
                ax.scatter(route_coords[:, 0], route_coords[:, 1],
                        s=12, c='blue', edgecolors='k', linewidths=0.3, zorder=3)

        # --- 1️⃣ New Route-like coordinates ---
        if new_route_coords is not None and len(new_route_coords) > 0:
            new_route_coords = np.array([p for p in new_route_coords if is_free(p[0], p[1])])
            if len(new_route_coords) > 0:
                ax.plot(new_route_coords[:, 0], new_route_coords[:, 1], c='orange', linewidth=1.5)
                ax.scatter(new_route_coords[:, 0], new_route_coords[:, 1],
                        s=12, c='orange', edgecolors='k', linewidths=0.3, zorder=3)

        # --- 3️⃣ Point-like coordinates ---
        if agent_coords is not None:
            pts = np.array([p for p in agent_coords if is_free(p[0], p[1])])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=12, c='green', zorder=3)

        # --- 3️⃣ Point-like coordinates ---
        if target_coords is not None:
            pts = np.array([p for p in target_coords if is_free(p[0], p[1])])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=12, c='purple', zorder=3)

        # --- Display setup ---
        ax.set_xlabel("X (world units)")
        ax.set_ylabel("Y (world units)")
        ax.set_title("Occupancy Grid with External Elements")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.axis('equal')
        plt.tight_layout(pad=0., rect=[0, 0, 1, 1])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = buf.shape
        img = buf[:, :, :3]  # drop alpha
        if img.shape[0]>800:
            img = img[::2, ::2, :]
        print(img.shape)
        plt.close(fig)
        return img.tolist()
        img = Image.fromarray(img)

        return img
    
    def refine_route(self, route, curr_time, route_coords=None, circle_coords=None, agent_coords=None, target_coords=None):
        ret=Route()
        for wp in route:
            nwp_loc = self.waypoints[self.get_nearest_waypoints(wp)[0]].location
            if len(ret)==0:
                ret.append(RouteNode(list(nwp_loc), 'walk', datetime.combine(curr_time.date(), datetime.strptime("23:59:59", "%H:%M:%S").time())))
            else:
                new_route=self.query_route(ret[-1].location, goal_place=None, goal_trans=nwp_loc, curr_time=curr_time)
                if new_route.impossible == 0: ret.extend(new_route)
        if target_coords is not None:
            if len(ret)==0:
                ret.append(RouteNode(list(target_coords[0]), 'walk', datetime.combine(curr_time.date(), datetime.strptime("23:59:59", "%H:%M:%S").time())))
            else:
                new_route=self.query_route(ret[-1].location, goal_place=None, goal_trans=target_coords[0], curr_time=curr_time)
                if new_route.impossible == 0: ret.extend(new_route)
        ret.simplify()
        return {"refined_route": ret, "grid_map_image": self.get_grid_map_image(route_coords=route_coords, circle_coords=circle_coords, agent_coords=agent_coords, target_coords=target_coords, new_route_coords=[list(wp.location) for wp in ret])}


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", '-s', type=str, required=True)
    args = parser.parse_args()
    scene_dir = f"{ASSETS_PATH}/scenes/{args.scene}"
    if not os.path.exists(f"{scene_dir}/road_data/road_data.pkl"):
        print(f"{scene_dir}/road_data/road_data.pkl not exist!")
        exit()
    with open(f"{scene_dir}/raw/center.txt", "r") as file:
        for line in file:
            ref_lat, ref_lon = line.strip().split()
        ref_lat, ref_lon = float(ref_lat), float(ref_lon)
    building_metadata = json.load(open(os.path.join(f"{scene_dir}/agents_num_15", "building_metadata.json"), 'r'))
    place_metadata = json.load(open(os.path.join(f"{scene_dir}/agents_num_15", "place_metadata.json"), 'r'))
    map_tool=MapTool(scene_name=args.scene, building_metadata=building_metadata, place_metadata=place_metadata)
    wps=[wp.location for wp in map_tool.waypoints]
    xs, ys = zip(*wps)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, 'bo', markersize=3)
    n_wps=map_tool.get_nearest_waypoints([285.16, -196.96])
    c_wps=set()
    for wp in n_wps:
        c_wps.update(map_tool.get_connected_waypoints(wp))
    c_wps=[wp.location for wp in map_tool.waypoints if wp.id in c_wps]
    c_xs, c_ys = zip(*c_wps)
    plt.plot(c_xs, c_ys, 'ro', markersize=3)
    plt.title(f'MapTool Waypoints Visualization - {args.scene}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.axis('equal')
    plt.show()
