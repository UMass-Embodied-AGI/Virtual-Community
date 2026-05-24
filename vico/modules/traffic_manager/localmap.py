import pyxodr
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from pathlib import Path
import pickle
import random
import pyproj

def cubic_func(t, shape):
    u = shape['aU']+shape['bU']*t+shape['cU']*(t**2)+shape['dU']*(t**3)  # Cubic function for u
    v = shape['aV']+shape['bV']*t+shape['cV']*(t**2)+shape['dV']*(t**3)  # Quadratic function for v
    return u, v
def cubic_func_d(t, shape):
    u = shape['bU']+2*shape['cU']*(t)+3*shape['dU']*(t**2)  # Cubic function for u
    v = shape['bV']+2*shape['cV']*(t)+3*shape['dV']*(t**2)  # Quadratic function for v
    return u, v
def cubic_func_d2(t, shape):
    u = 2*shape['cU']+6*shape['dU']*(t)  # Cubic function for u
    v = 2*shape['cV']+6*shape['dV']*(t)  # Quadratic function for v
    return u, v
def cubic_curve(t, shape, hdg):
    u, v = cubic_func(t, shape)
    rotation_matrix = np.array([[np.cos(hdg), -np.sin(hdg)],
                             [np.sin(hdg), np.cos(hdg)]])
    uv_rotated = rotation_matrix @ np.vstack((u, v))
    u_rotated = uv_rotated[0, :]
    v_rotated = uv_rotated[1, :]
    return u_rotated, v_rotated
def load_opendrive_file(file_path):
    # Create an instance of the OpenDRIVE parser
    opendrive_parser = pyxodr.OpenDrive()
    
    # Load the OpenDRIVE file
    opendrive_parser.load(file_path)
    
    # Accessing road objects and their coordinates
    road_objects = opendrive_parser.get_road_objects()
    
    # Create a data structure to store the parsed data
    roads_data = []
    
    for road in road_objects:
        road_info = {
            'id': road.id,
            'length': road.length,
            'geometry': []
        }
        
        # Extract geometry points (x, y, z)
        for geometry in road.geometry:
            points = geometry.get_points()
            road_info['geometry'].extend(points)
        
        roads_data.append(road_info)
    
    return roads_data

def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    earth_radius = 6378137  # in meters
    meters_per_degree_lat = 111139  # Approximate meters per degree latitude
    
    # Calculate meters per degree longitude based on reference latitude
    meters_per_degree_lon = 111139 * math.cos(math.radians(ref_lat))
    
    # Convert lat/lon to x/y
    x = (lon - ref_lon) * meters_per_degree_lon
    y = (lat - ref_lat) * meters_per_degree_lat
    
    return [x, y]

def corr_xy(x, y, ref_lat, ref_lon, offset_x, offset_y):
    inProj = pyproj.Proj("+proj=merc")
    outProj = pyproj.Proj(init='epsg:4326')  # WGS84
    lon, lat = pyproj.transform(inProj, outProj, x-offset_x, y-offset_y)
    x, y = lat_lon_to_xy(lat, lon, ref_lat, ref_lon)
    return x, y

import xml.etree.ElementTree as ET

class TarjanSCC:
    def __init__(self, graph):
        self.graph = graph
        self.index = 0
        self.stack = []
        self.lowlink = {}
        self.index_mapping = {}
        self.sccs = []
        self.ined = {}

    def strongconnect(self, v):
        # Set the depth index for v to the smallest unused index
        self.index_mapping[v] = self.index
        self.lowlink[v] = self.index
        self.index += 1
        self.stack.append(v)

        # Consider successors of v
        for w in self.graph[v]:
            if w not in self.index_mapping:
                # Successor w has not yet been visited; recurse on it
                self.strongconnect(w)
                # Check if the lowlink needs to be updated
                self.lowlink[v] = min(self.lowlink[v], self.lowlink[w])
            elif w in self.stack:
                # Successor w is in stack S and hence in the current SCC
                self.lowlink[v] = min(self.lowlink[v], self.index_mapping[w])

        # If v is a root node, pop the stack and generate an SCC
        if self.lowlink[v] == self.index_mapping[v]:
            current_scc = []
            while True:
                w = self.stack.pop()
                current_scc.append(w)
                self.ined[w]=len(self.sccs)
                if w == v:
                    break
            # Store the found SCC
            self.sccs.append(current_scc)

    def find_sccs(self):
        for v in self.graph:
            if v not in self.index_mapping:
                self.strongconnect(v)
        return self.sccs, self.ined

class Waypoint:
    def __init__(self, id, x, y, previous_waypoints, next_waypoints, next_left_waypoint, next_right_waypoint):
        self.id = id
        self.x = x
        self.y = y
        self.pre = previous_waypoints
        self.nxt = next_waypoints
        self.left = next_left_waypoint
        self.right = next_right_waypoint

# Unified map for both vehicles and pedestrians
class BaseMap:
    def __init__(self, file_path, ref_lat, ref_lon):
        assert os.path.exists(file_path)
        tree = ET.parse(file_path)
        self.root = tree.getroot()
        self.offset_x = float(self.root.find('header/offset').get('x'))
        self.offset_y = float(self.root.find('header/offset').get('y'))
        self.inProj = pyproj.Proj("+proj=merc")
        self.outProj = pyproj.Proj(init='epsg:4326')  # WGS84
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon

        self.roads = {}
        self.junctions = {}
        self.main_roads = []
        self.vehicle_paths = {}
        self.pedestrian_paths = {}

        for road in self.root.findall('road'):
            name = road.get('name')
            length = float(road.get('length'))
            road_id = road.get('id')
            junction = road.get('junction')
            
            if junction == "-1":
                self.main_roads.append(road_id)

            predecessor = road.find('link/predecessor')
            successor = road.find('link/successor')
            geometry = [
                {
                    's': float(geo.get('s')),
                    'x': float(geo.get('x')),
                    'y': float(geo.get('y')),
                    'hdg': float(geo.get('hdg')),
                    'length': float(geo.get('length'))
                }
                for geo in road.findall('planView/geometry')
            ]
            
            lanes = self.parse_lanes(road)
            
            self.roads[road_id] = {
                'name': name,
                'length': length,
                'id': road_id,
                'junction': junction,
                'predecessor': predecessor,
                'successor': successor,
                'geometry': geometry,
                'lanes': lanes,
            }
            
        self.process_lane_types()
        self.visualize_all_roads()
        
    def parse_lanes(self, road):
        """Extract lane information, including types, from a road."""
        lanes = []
        for lane_section in road.findall('lanes/laneSection'):
            # Handle left lanes
            left_lanes = lane_section.find('left')
            if left_lanes is not None:
                for lane in left_lanes.findall('lane'):
                    lane_id = lane.get('id')
                    lane_type = lane.get('type', 'unknown')
                    #print(f"Left Lane ID: {lane_id}, Type: {lane_type}")
                    lanes.append({'id': lane_id, 'type': lane_type, 'side': 'left'})
    
            # Handle right lanes
            right_lanes = lane_section.find('right')
            if right_lanes is not None:
                for lane in right_lanes.findall('lane'):
                    lane_id = lane.get('id')
                    lane_type = lane.get('type', 'unknown')
                    #print(f"Right Lane ID: {lane_id}, Type: {lane_type}")
                    lanes.append({'id': lane_id, 'type': lane_type, 'side': 'right'})
    
        return lanes

        
    def process_lane_types(self):
        """Separate vehicle and pedestrian paths based on lane types."""
        self.vehicle_paths = {}
        self.pedestrian_paths = {}
        
        for road_id, road_data in self.roads.items():
            lanes = road_data.get('lanes', [])
            has_driving = any(lane['type'] == 'driving' for lane in lanes)
            has_sidewalk = any(lane['type'] == 'sidewalk' for lane in lanes)
    
            if has_driving:
                self.vehicle_paths[road_id] = road_data
    
            if has_sidewalk:
                self.pedestrian_paths[road_id] = road_data
                
        print(f"Number of vehicle paths: {len(self.vehicle_paths)}")
        print(f"Number of pedestrian paths: {len(self.pedestrian_paths)}")
    
        # Process junctions for connectivity
        for road in self.root.findall('road'):
            road_id = road.get('id')
            lanes = road.find('lanes')
            if not lanes:
                continue
    
            # Parse lane sections
            road_lanes = []
            for lane_section in lanes.findall('laneSection'):
                for lane_type in ['left', 'right']:
                    lane_group = lane_section.find(lane_type)
                    if lane_group is not None:
                        for lane in lane_group.findall('lane'):
                            lane_id = lane.get('id')
                            lane_type = lane.find('type').text if lane.find('type') is not None else None
                            road_lanes.append({'id': lane_id, 'type': lane_type})
    
            # Assign lanes to the road data
            self.roads[road_id]['lanes'] = road_lanes
    
        # Junction processing
        for junc in self.root.findall('junction'):
            name = junc.get('name')
            junction_id = junc.get('id')
            connections = [
                {
                    'incomingRoad': conn.get('incomingRoad'),
                    'connectingRoad': conn.get('connectingRoad'),
                    'contactPoint': conn.get('contactPoint'),
                }
                for conn in junc.findall('connection')
            ]
            self.junctions[junction_id] = {
                'name': name,
                'id': junction_id,
                'connections': connections,
            }
        
        # for road_id, road_data in self.roads.items():
        #     has_driving = any(lane['type'] == 'driving' for lane in road_data['lanes'])
        #     has_sidewalk = any(lane['type'] == 'sidewalk' for lane in road_data['lanes'])

        #     if has_driving:
        #         self.vehicle_paths[road_id] = road_data

        #     if has_sidewalk:
        #         self.pedestrian_paths[road_id] = road_data

        # for junc in root.findall('junction'):
        #     name = junc.get('name')
        #     junction_id = junc.get('id')
        #     connections = [conn for conn in junc.findall('connection')]
        #     self.junctions[junction_id] = {
        #         'name': name,
        #         'id': junction_id,
        #         'connections': connections,
        #     }

    def get_roads(self):
        return self.roads
        
    def get_main_roads(self):
        return self.main_roads

    def get_length(self, road_id):
        return self.roads[road_id]['length']
        
    def get_pos(self, road_id, s):
        road = self.roads[road_id]
        
        # Ensure `s` is within the valid range
        assert s <= road['length'], f"s ({s}) exceeds road length ({road['length']}) for road {road_id}"
        if s == road['length']:
            s -= 1e-6  # Adjust slightly to stay within the road
    
        # Validate geometry
        if not road['geometry']:
            raise ValueError(f"Road {road_id} has no valid geometry.")
        if any('x' not in geo or 'y' not in geo or 'hdg' not in geo or 'length' not in geo for geo in road['geometry']):
            raise ValueError(f"Road {road_id} has invalid geometry: {road['geometry']}")
    
        # Find the correct geometry segment
        i = 0
        while i < len(road['geometry']):
            if s <= road['geometry'][i]['length']:
                break
            s -= road['geometry'][i]['length']
            i += 1
    
        # Fallback to the last segment if `i` exceeds bounds
        if i >= len(road['geometry']):
            i = len(road['geometry']) - 1
    
        # Calculate position
        end_x = road['geometry'][i]['x'] + s * math.cos(road['geometry'][i]['hdg'])
        end_y = road['geometry'][i]['y'] + s * math.sin(road['geometry'][i]['hdg'])
        
        # Transform to lat/lon and back to xy
        end_lon, end_lat = pyproj.transform(
            self.inProj, self.outProj, end_x - self.offset_x, end_y - self.offset_y
        )
        end_x, end_y = lat_lon_to_xy(end_lat, end_lon, self.ref_lat, self.ref_lon)
        
        return (end_x, end_y)


    # def get_pos(self, road_id, s):
    #     road = self.roads[road_id]
    #     assert s < road['length']
    #     i = 0
    #     while i < len(road['geometry']):
    #         if s <= road['geometry'][i]['length']:
    #             break
    #         s -= road['geometry'][i]['length']
    #         i += 1

    #     end_x = road['geometry'][i]['x'] + s * math.cos(road['geometry'][i]['hdg'])
    #     end_y = road['geometry'][i]['y'] + s * math.sin(road['geometry'][i]['hdg'])
    #     end_lon, end_lat = pyproj.transform(
    #         self.inProj, self.outProj, end_x - self.offset_x, end_y - self.offset_y
    #     )
    #     end_x, end_y = lat_lon_to_xy(end_lat, end_lon, self.ref_lat, self.ref_lon)
    #     return (end_x, end_y)
        
    def get_successor(self, road_id):
        return self.roads[road_id]['successor']

    def get_next_roads(self, road_id):
        if self.roads[road_id]['successor'] is None:
            return []
        if self.roads[road_id]['successor'].get('elementType') == 'road':
            return [self.roads[road_id]['successor'].get('elementId')]
        else:
            if self.roads[road_id]['successor'].get('elementId') not in self.junctions:
                return []
            junc = self.junctions[self.roads[road_id]['successor'].get('elementId')]
            return [
                conn.get('connectingRoad')
                for conn in junc['connections']
                if conn.get('incomingRoad') == road_id
            ]

    def get_next_road(self, road_id):
        next_roads = self.get_next_roads(road_id)
        if not next_roads:
            return None
        return random.choice(next_roads)

    def get_junction(self, road_id):
        return self.roads[road_id]['junction']
        
    def visualize_all_roads(self):
        """Visualize all roads (vehicle and pedestrian) without filtering."""
        plt.figure(figsize=(10, 10))
        for road_id, road_data in self.roads.items():
            for geometry in road_data["geometry"]:
                x = geometry["x"]
                y = geometry["y"]
                length = geometry["length"]
                hdg = geometry["hdg"]

                end_x = x + length * np.cos(hdg)
                end_y = y + length * np.sin(hdg)

                x, y = corr_xy(x, y, self.ref_lat, self.ref_lon, self.offset_x, self.offset_y)
                end_x, end_y = corr_xy(end_x, end_y, self.ref_lat, self.ref_lon, self.offset_x, self.offset_y)

                plt.plot([x, end_x], [y, end_y], color="blue")

        plt.title("Visualization of All Roads (Vehicle and Pedestrian)")
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.grid()
        
        # Save the plot
        output_file="all_roads_visualization.png"
        plt.savefig(output_file)
        
        plt.show()

class LocalMap:
    def __init__(self, file_path, terrain_height_path, ref_lat, ref_lon):
        assert os.path.exists(file_path)
        tree = ET.parse(file_path)
        root = tree.getroot()
        self.offset_x = float(root.find('header/offset').get('x'))
        self.offset_y = float(root.find('header/offset').get('y'))
        self.inProj = pyproj.Proj("+proj=merc")
        self.outProj = pyproj.Proj(init='epsg:4326')  # WGS84
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        # if terrain_height_path:
        #     terrain_height_path = os.path.join(get_assets_dir(), terrain_height_path)
        #     self.terrain_height_field = load_height_field(terrain_height_path)
        
        self.roads = {}
        self.junctions = {}
        self.main_roads = [] # roads not in any junction
        self.pedestrian_main_roads = {}

        for road in root.findall('road'):
            name = road.get('name')
            length = float(road.get('length'))
            id = road.get('id')
            junction = road.get('junction')
            if junction=="-1": self.main_roads.append(id)
            predecessor = road.find('link/predecessor')
            successor = road.find('link/successor')
            # get road geometry
            geometry = [
                {
                    's': float(geo.get('s')), 
                    'x': float(geo.get('x')), 
                    'y': float(geo.get('y')), 
                    'hdg': float(geo.get('hdg')), 
                    'length': float(geo.get('length'))
                }
                for geo in road.findall('planView/geometry')
            ]
            for idx, geo in enumerate(road.findall('planView/geometry')):
                if geo.find('line') is not None:
                    geometry[idx]['type']='line'
                elif geo.find('paramPoly3') is not None:
                    geometry[idx]['type']='paramPoly3'
                    param=geo.find('paramPoly3')
                    shape={'aU': float(param.get('aU')), 'bU': float(param.get('bU')), 'cU': float(param.get('cU')), 'dU': float(param.get('dU')), 'aV': float(param.get('aV')), 'bV': float(param.get('bV')), 'cV': float(param.get('cV')), 'dV': float(param.get('dV')), 'pRange': param.get('pRange')}
                    geometry[idx]['shape']=shape
                else:
                    geometry[idx]['type']='unknown'
            # get lane information
            lanes = [
                {
                    'id': int(lane.get('id')), 
                    'type': lane.get('type'), 
                    'level': lane.get('level'), 
                    'width': float(lane.find('width').get('a'))+float(lane.find('roadMark').get('width'))
                }
                for lane in road.findall('lanes/laneSection/right/lane')
            ]
            lane_offset = float(road.find('lanes/laneOffset').get('a')) if road.find('lanes/laneOffset') is not None else 0.
            self.roads[id]={
                'name': name, 
                'length': length, 
                'id': id, 
                'junction': junction, 
                'predecessor': predecessor, 
                'successor': successor, 
                'geometry': geometry, 
                'lanes':lanes, 
                'lane_offset': lane_offset
            }

        for junc in root.findall('junction'):
            name = junc.get('name')
            id = junc.get('id')
            connections = [
                {
                    'incomingRoad': conn.get('incomingRoad'),
                    'connectingRoad': conn.get('connectingRoad'),
                    'contactPoint': conn.get('contactPoint'),
                }
                for conn in junc.findall('connection')
            ]
            self.junctions[id]={'name':name, 'id': id, 'connections': connections}
        
        self.valid_road=self.cut_branch()
        self.main_roads=[road for road in self.main_roads if road in self.valid_road]
        for road_id, road_data in self.roads.items():
            if road_id not in self.valid_road or road_id not in self.main_roads:continue
            if self.has_sidewalk(road_id):
                self.pedestrian_main_roads[road_id]=road_data

    def has_sidewalk(self, road_id):
        return any(lane['type']=='sidewalk' for lane in self.roads[road_id]['lanes'])

    def cut_branch(self, pedestrian_only=False):
        '''
        Cut dead end roads. Pedestrian_only feature is not needed because pedestrians view roads as bidirectional.
        '''
        graph={}
        road_set=self.pedestrian_main_roads if pedestrian_only else self.roads
        for road in road_set:
            graph[road]=self.get_next_roads(road, pedestrian_only=pedestrian_only)
        sccs, ined=TarjanSCC(graph).find_sccs()

        queue=np.zeros(len(sccs), dtype=int)
        rd=np.zeros(len(sccs), dtype=int)
        head, tail = 0, 0
        for i in range(len(sccs)):
            for road in sccs[i]:
                next_roads=self.get_next_roads(road, pedestrian_only=pedestrian_only)
                for e in next_roads:
                    if ined[e]!=ined[road]:
                        rd[ined[e]]+=1
        for i in range(len(sccs)):
            if rd[i]==0:
                queue[tail]=i
                tail+=1
        while head < tail:
            scc=sccs[queue[head]]
            head+=1
            for road in scc:
                next_roads=self.get_next_roads(road, pedestrian_only=pedestrian_only)
                for e in next_roads:
                    if ined[e]!=ined[road]:
                        rd[ined[e]]-=1
                        if rd[ined[e]]==0:
                            queue[tail]=ined[e]
                            tail+=1
        valid_road=[] # All roads can lead to some scc containing more than 1 roads should be valid.
        while tail>0:
            tail-=1
            if len(sccs[tail])>1:
                for road in sccs[tail]:
                    valid_road.append(road)
            else:
                road=sccs[tail][0]
                next_roads=self.get_next_roads(road, pedestrian_only=pedestrian_only)
                for e in next_roads:
                    if e in valid_road:
                        valid_road.append(road)
                        break
        return valid_road


    def get_roads(self):
        return self.roads
    
    def get_main_roads(self):
        return self.main_roads
    
    def get_length(self, road_id):
        return self.roads[road_id]['length']
    
    def calc_lane_scaler(self, road_id, s, lane_id=0, direction=0):
        '''
        the ratio of offset on lanes to offset on reference line
        '''
        road=self.roads[road_id]
        assert s<road['length']
        i=0
        # find correct road section i
        while i<len(road['geometry']):
            if s<=float(road['geometry'][i]['length']):
                break
            s-=float(road['geometry'][i]['length'])
            i+=1
        # calculate the distance to reference line
        dis_to_ref, j = 0.13-road['lane_offset'], 0
        if lane_id<0: lane_id+=len(road['lanes'])
        while j<=lane_id:
            if j==lane_id:
                if direction==0:
                    dis_to_ref+=road['lanes'][j]['width']/2
                if direction==1:
                    dis_to_ref+=road['lanes'][j]['width']*4/5
                if direction==2:
                    dis_to_ref+=road['lanes'][j]['width']*1/5
                break
            dis_to_ref+=road['lanes'][j]['width']
            j+=1
        # calculate correct direction hdg
        hdg=road['geometry'][i]['hdg']
        if road['geometry'][i]['type']=='paramPoly3':
            sp=s/float(road['geometry'][i]['length'])
            du, dv = cubic_func_d(sp, road['geometry'][i]['shape'])
            d2u, d2v = cubic_func_d2(sp, road['geometry'][i]['shape'])
            duv = math.sqrt(du**2 + dv**2)
            # up, vp = (fu)+ (dv/duv)*dis_to_ref, (fv)- (du/duv)*dis_to_ref
            dup, dvp = (du)+(d2v/duv)*dis_to_ref, (dv)-(d2u/duv)*dis_to_ref
            duvp = math.sqrt(dup**2+dvp**2)
            scaler = duvp/duv
        else:
            scaler = 1.
        return scaler
    
    def get_pos(self, road_id, s, lane_id=0, direction=0, std=0.):
        '''
        direction=0: driving, in the middle
        direction=1: walk along, in the right
        direction=2: walk back, in the left
        '''
        road=self.roads[road_id]
        assert s<road['length']
        i=0
        # find correct road section i
        while i<len(road['geometry']):
            if s<=float(road['geometry'][i]['length']):
                break
            s-=float(road['geometry'][i]['length'])
            i+=1
        # calculate correct direction hdg
        hdg=road['geometry'][i]['hdg']
        if road['geometry'][i]['type']=='paramPoly3':
            end_x, end_y = cubic_curve(np.array([s/float(road['geometry'][i]['length'])]), road['geometry'][i]['shape'], road['geometry'][i]['hdg'])
            del_x, del_y = cubic_curve(np.array([s/float(road['geometry'][i]['length'])])-0.0000001, road['geometry'][i]['shape'], road['geometry'][i]['hdg'])
            del_x, del_y = end_x-del_x, end_y-del_y
            hdg = math.atan2(del_y, del_x)
            end_x, end_y = end_x[0]+road['geometry'][i]['x'], end_y[0]+road['geometry'][i]['y']
        else:
            end_x = road['geometry'][i]['x'] + s * math.cos(road['geometry'][i]['hdg'])
            end_y = road['geometry'][i]['y'] + s * math.sin(road['geometry'][i]['hdg'])
        # calculate the distance to reference line
        dis_to_ref, j = 0.13-road['lane_offset'], 0
        noise = min(max(random.normalvariate(0., std), -0.8), 0.8)
        dis_to_ref += noise
        if lane_id<0: lane_id+=len(road['lanes'])
        while j<=lane_id:
            if j==lane_id:
                if direction==0:
                    dis_to_ref+=road['lanes'][j]['width']/2
                if direction==1:
                    dis_to_ref+=road['lanes'][j]['width']*5/7 # 2.0
                if direction==2:
                    dis_to_ref+=road['lanes'][j]['width']*2/7 # 0.8
                break
            dis_to_ref+=road['lanes'][j]['width']
            j+=1
        end_x = end_x + dis_to_ref * math.cos(hdg-np.pi/2)
        end_y = end_y + dis_to_ref * math.sin(hdg-np.pi/2)
        # get true coordinates
        end_lon, end_lat = pyproj.transform(self.inProj, self.outProj, end_x-self.offset_x, end_y-self.offset_y)
        end_x, end_y = lat_lon_to_xy(end_lat, end_lon, self.ref_lat, self.ref_lon)
        return [end_x, end_y]
    
    def get_rot(self, road_id, s, direction=0):
        road=self.roads[road_id]
        assert s<road['length']
        i=0
        # print(len(road['geometry']))
        while i<len(road['geometry']):
            if s<=float(road['geometry'][i]['length']):
                break
            s-=float(road['geometry'][i]['length'])
            i+=1
        # calculate correct direction hdg
        hdg=road['geometry'][i]['hdg']
        if road['geometry'][i]['type']=='paramPoly3':
            end_x, end_y = cubic_curve(np.array([s/float(road['geometry'][i]['length'])]), road['geometry'][i]['shape'], road['geometry'][i]['hdg'])
            del_x, del_y = cubic_curve(np.array([s/float(road['geometry'][i]['length'])])-0.0000001, road['geometry'][i]['shape'], road['geometry'][i]['hdg'])
            del_x, del_y = end_x-del_x, end_y-del_y
            hdg = math.atan2(del_y, del_x)
        if direction==2:
            hdg-=math.pi
            if hdg<0:
                hdg+=2*math.pi
        return np.array([0.0,0.0,hdg], dtype=np.float64)
    
    def get_successor(self, road_id):
        return self.roads[road_id]['successor']
    
    def get_previous_roads(self, road_id, valid=False, pedestrian_only=False):
        '''
        todo
        '''
        if self.roads[road_id]['predecessor'] is None: return []
        if self.roads[road_id]['predecessor'].get('elementType')=='road':
            prev_roads = [self.roads[road_id]['predecessor'].get('elementId')]
        else:
            if self.roads[road_id]['predecessor'].get('elementId') not in self.junctions: return []
            junc = self.junctions[self.roads[road_id]['predecessor'].get('elementId')]
            prev_roads = [conn['connectingRoad'] for conn in junc['connections'] if road_id in self.get_next_roads(conn['connectingRoad'])] # junction info don't include this information so...
        if valid:
            prev_roads = [road for road in prev_roads if road in self.valid_road]
        if pedestrian_only:
            prev_roads = [road for road in prev_roads if self.has_sidewalk(road)]
        return prev_roads
    
    def get_next_roads(self, road_id, valid=False, pedestrian_only=False):
        '''
        valid controls whether consider road validity. Return road ids.
        '''
        if self.roads[road_id]['successor'] is None: return []
        if self.roads[road_id]['successor'].get('elementType')=='road':
            next_roads = [self.roads[road_id]['successor'].get('elementId')]
        else:
            if self.roads[road_id]['successor'].get('elementId') not in self.junctions: return []
            junc = self.junctions[self.roads[road_id]['successor'].get('elementId')]
            next_roads = [conn['connectingRoad'] for conn in junc['connections'] if conn['incomingRoad'] == road_id]
        if valid:
            next_roads = [road for road in next_roads if road in self.valid_road]
        if pedestrian_only:
            next_roads = [road for road in next_roads if self.has_sidewalk(road)]
        return next_roads
        
    def get_previous_road(self, road_id, valid=False, pedestrian_only=False):
        '''
        valid controls whether consider road validity. Return a road id if possible.
        '''
        prev_roads = self.get_previous_roads(road_id, valid=valid, pedestrian_only=pedestrian_only)
        if not prev_roads:
            return None
        return random.choice(prev_roads)
    
    def get_next_road(self, road_id, valid=False, pedestrian_only=False):
        '''
        valid controls whether consider road validity. Return a road id if possible.
        '''
        next_roads = self.get_next_roads(road_id, valid=valid, pedestrian_only=pedestrian_only)
        if not next_roads:
            return None
        return random.choice(next_roads)
        
    def get_junction(self, road_id):
        return self.roads[road_id]['junction']

# PedestrianMap
class PedestrianMap(BaseMap):
    def __init__(self, file_path, ref_lat, ref_lon):
        super().__init__(file_path, ref_lat, ref_lon)
        
    def get_pedestrian_paths(self):
        """Return all paths designated for pedestrians."""
        return self.pedestrian_paths