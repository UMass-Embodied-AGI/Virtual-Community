import random
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

from .. import *

from .localmap import *
from .simple_vehicle import *
from .simple_agent import *
from ...tools.utils import get_assets_dir

class TrafficManager:
    def __init__(self, env, scene_name, vehicle_number=0, pedestrian_number=0, dry_run=False, enable_tm_debug=False, logger=None, debug=False):
        self.env = env
        self.logger = logger
        self.debug = debug
        self.enable_tm_debug = enable_tm_debug
        self.dry_run = dry_run
        if pedestrian_number>0 or vehicle_number>0:
            self.enable_traffic=True
        else:
            self.enable_traffic=False
        if self.enable_traffic and not os.path.exists(f"{get_assets_dir()}/scenes/{scene_name}/road_data/road_data.xodr"):
            logger.error(f"{get_assets_dir()}/scenes/{scene_name}/road_data/road_data.xodr not exist!")
            exit()
        if self.enable_traffic:
            with open(f'{self.env.scene_assets_dir}/center.txt', "r") as file:
                for line in file:
                    ref_lat, ref_lon = line.strip().split()
                ref_lat, ref_lon = float(ref_lat), float(ref_lon)
            self.map = LocalMap(file_path=f"{get_assets_dir()}/scenes/{scene_name}/road_data/road_data.xodr",
                                  terrain_height_path=None if self.dry_run else f"{self.env.scene_assets_dir}/height_field.npz",
                                  ref_lat=ref_lat, ref_lon=ref_lon)
            self.sampled_roads = self.map.get_main_roads()
            self.vehicles_on_road = {}
            for road_id in self.map.get_roads().keys():
                self.vehicles_on_road[road_id]=list()
        else:
            self.map = None
            self.sampled_roads = []
            self.vehicles_on_road = {}
        self.vehicle_number = vehicle_number
        self.pedestrian_number = pedestrian_number
        
        self.vehicles = []
        self.simple_vehicles = []
        self.simple_avatars = []
        self.avatars = []
        self.vehicle_action = [None] * vehicle_number
        self.uniform_speed=5.
        self.min_dis=2.

        # Simple Pedestrian and Vehicle Initialization
        self.create_pedestrians(pedestrian_number)
        for i in range(self.vehicle_number):
            self.add_simple_vehicle()
        if self.dry_run:
            return

        for i in range(self.vehicle_number):
            simple_vehicle = self.simple_vehicles[i]
            self.add_vehicle(
                vehicle=self.env.add_vehicle(
                    name=f"auto_vehicle_{i}",
                    vehicle_asset_path="ViCo/cars/Car/OldCar.glb",
                    ego_view_options={
                        "res": (self.env.resolution, self.env.resolution),
                        "fov": 90,
                        "GUI": False,
                    } if self.enable_tm_debug else None,
                    position=np.array(simple_vehicle.get_pos() + [79.59], dtype=np.float64),
                    rotation=np.array(simple_vehicle.get_rot(), dtype=np.float64),
                    dt=1e-2,  # self.env.dt,
                    forward_speed_m_per_s=20,
                    angular_speed_deg_per_s=60,
                    terrain_height_path=f"{self.env.scene_assets_dir}/height_field.npz",
                ),
                simple_vehicle=simple_vehicle,
                debug=self.debug,
                logger=self.logger
            )
            if self.enable_tm_debug:
                os.makedirs(os.path.join(self.env.output_dir, 'traffic_ego', f"auto_vehicle_{i}"), exist_ok=True)

        self.bus = Bus(
            bus=self.env.add_vehicle(
                name="bus",
                vehicle_asset_path="ViCo/cars/Car/kozak_i_van2.glb",
                ego_view_options={
                    "res": (self.env.resolution, self.env.resolution),
                    "fov": 90,
                    "GUI": False,
                },
                dt= self.env.sec_per_step if self.env.skip_avatar_animation else 1e-2,
                terrain_height_path=f"{self.env.scene_assets_dir}/height_field.npz",
            ),
            forward_speed=self.env.transit_info["bus"]["forward_speed"],
            rotation_speed=self.env.transit_info["bus"]["rotation_speed"],
            current_time=self.env.curr_time,
            route=self.env.transit_info["bus"]["refined_waypoints"],
            stop_names=list(self.env.transit_info["bus"]["stops"].keys()),
            stop_indices=self.env.transit_info["bus"]["stop_indices"],
            travel_time=self.env.transit_info["bus"]["travel_time"],
            bus_start_time=self.env.transit_info["bus"]["start_time"],
			bus_end_time=self.env.transit_info["bus"]["end_time"],
            frequency=self.env.transit_info["bus"]["frequency"],
            debug=self.debug,
            logger=self.logger
        )
        self.shared_bicycles = SharedBicycles(
            self.env,
            self.env.transit_info["bicycle"]["stations"],
            self.env.env_other_meta["price"]["per_minute"]["transit"]["bike"],
            num_bicycles = len(self.env.transit_info["bicycle"]["stations"]),
            terrain_height_path=f"{self.env.scene_assets_dir}/height_field.npz",
            debug=self.debug, logger=self.logger
        )

        for i in range(self.pedestrian_number):
            simple_avatar = self.simple_avatars[i]
            self.add_avatar(
                vehicle=self.env.add_avatar(name=f"auto_avatar_{i}",
                    motion_data_path='ViCo/avatars/motions/motion.pkl',
                    skin_options={
                        'glb_path': self.env.config['agent_skins'][0],
                        'euler': (-90, 0, 90),
                        'pos': (0.0, 0.0, -0.959008030)
                    },
                    ego_view_options={
                        "res": (self.env.resolution, self.env.resolution),
                        "fov": 90,
                        "GUI": False,
                    } if self.enable_tm_debug else None, frame_ratio=5,
                    terrain_height_path=f"{self.env.scene_assets_dir}/height_field.npz",
                    third_person_camera_resolution=128 if self.env.enable_third_person_cameras else None,
                    enable_collision=False
                    ),
                simple_vehicle=simple_avatar,
                debug=self.debug,
                logger=self.logger,
                name=f"auto_avatar_{i}"
            )
            if self.enable_tm_debug:
                os.makedirs(os.path.join(self.env.output_dir, 'traffic_ego', f"auto_avatar_{i}"), exist_ok=True)

    # Function to create pedestrians
    def create_pedestrians(self, pedestrian_number):
        for _ in range(pedestrian_number):
            road_id = random.choice(list(self.map.pedestrian_main_roads.keys()))
            start_s = random.uniform(0, self.map.get_length(road_id))
            pos = self.map.get_pos(road_id = road_id, s = start_s, lane_id=-1)
            rot = self.map.get_rot(road_id = road_id, s = start_s)
            avatar=SimpleVehicle(id = _, init_pos = pos, init_rot = rot, init_road = road_id, init_s = start_s, std_speed = random.uniform(0.8, 1.4))
            self.simple_avatars.append(avatar)
        
    # Function to move pedestrians
    def move_pedestrians(self, dt=1.0, min_distance=0.5):
        for ped_id, ped in enumerate(self.simple_avatars):
            if self.dry_run==False and self.avatars[ped_id].avatar.action_status()=="ONGOING":
                continue
            road_id, s = ped.get_loc()
            road_length = self.map.get_length(road_id)
            scaler = self.map.calc_lane_scaler(road_id, s, lane_id=-1, direction=1 if ped.std_speed>0 else 2)
            s += ped.std_speed * dt / scaler  # Update position along the road
            ped.set_loc(road_id, s)
            #print(f"Pedestrian road_id: {road_id}, s: {ped['s']}, road_length: {road_length}")

            
            if s >= road_length:  # Check if pedestrian exceeds road length
                next_road_id = self.map.get_next_road(road_id, valid=True, pedestrian_only=True)
                if next_road_id:
                    if next_road_id in self.junction_hazzard:
                        pass
                    else:
                        # self.logger.info(f"Pedestrian moving to next road: {next_road_id}, s={ped['s']}")
                        ped.set_loc(next_road_id, 0.)  # Move to the next road
                else:
                    # self.logger.info(f"No next road for pedestrian on road {road_id}, s={ped['s']}")
                    ped.set_loc(road_id, road_length - 0.01)  # Stop at the end of the current road
                    ped.std_speed = -ped.std_speed # And turn around and walk back

            
            if s < 0:  # Check if pedestrian exceeds road length
                prev_road_id = self.map.get_previous_road(road_id, valid=True, pedestrian_only=True)
                if prev_road_id:
                    if prev_road_id in self.junction_hazzard:
                        pass
                    else:
                        # self.logger.info(f"Pedestrian moving to next road: {next_road_id}, s={ped['s']}")
                        ped.set_loc(prev_road_id, self.map.get_length(prev_road_id) - 0.01)  # Move to the next road
                else:
                    # self.logger.info(f"No next road for pedestrian on road {road_id}, s={ped['s']}")
                    ped.set_loc(road_id, 0.)  # Stop at the end of the current road
                    ped.std_speed = -ped.std_speed # And turn around and walk back
    
            # Update pedestrian position
            try:
                ped.set_pos(self.map.get_pos(ped.road, ped.s, lane_id=-1, direction=1 if ped.std_speed>0 else 2, std=0.4))
                ped.set_rot(self.map.get_rot(ped.road, ped.s, direction=1 if ped.std_speed>0 else 2))
            except AssertionError as e:
                print(f"Error in get_pos: road_id={ped.road}, s={ped.s}, road_length={road_length}")
                raise e

        # Handle pedestrian collisions
        self.detect_and_resolve_collisions(self.simple_avatars, min_distance)

        
    # Function to detect and resolve collisions
    def detect_and_resolve_collisions(self, pedestrians, min_distance):
        for i, ped1 in enumerate(pedestrians):
            for j, ped2 in enumerate(pedestrians):
                if i >= j:
                    continue
                dist = np.linalg.norm(np.array(ped1.get_pos()) - np.array(ped2.get_pos()))
                if dist < min_distance:
                    direction = (np.array(ped1.get_pos()) - np.array(ped2.get_pos())) / dist
                    adjustment = 0.5 * (min_distance - dist) * direction
                    ped1.pos += adjustment
                    ped2.pos -= adjustment

    def add_simple_vehicle(self):
        '''
        Firstly create a simple vehicle object, which is an abstract in traffic manager. Then env will call add_vehicle() to create an auto vehicle in the simulator.
        '''
        sampled_road = random.sample(self.sampled_roads, 1)[0]
        self.sampled_roads.remove(sampled_road)
        pos = self.map.get_pos(road_id = sampled_road, s = 0.)
        rot = self.map.get_rot(road_id = sampled_road, s = 0.)
        simple_vehicle=SimpleVehicle(id = len(self.simple_vehicles), init_pos = pos, init_rot = rot, init_road = sampled_road, init_s = 0.)
        self.simple_vehicles.append(simple_vehicle)
        self.vehicles_on_road[sampled_road].append((0., simple_vehicle.id))
        return simple_vehicle

    def add_vehicle(self, vehicle, simple_vehicle, debug=False, logger=None):
        '''
        Create an AutoVehicle object. Just like bus class.
        '''
        vehicle = AutoVehicle(vehicle, simple_vehicle, debug=debug, logger=logger)
        self.vehicles.append(vehicle)

    def add_avatar(self, vehicle, simple_vehicle, debug=False, logger=None, name=None):
        '''
        Create an AutoAvatar object. Just like bus class.
        '''
        avatar = AutoAvatar(vehicle, simple_vehicle, debug=debug, logger=logger, name=name)
        self.avatars.append(avatar)
    
    def plan(self):
        self.junction_hazzard=dict()
        # filling each junctions with vehicles already in it
        for rid, road in enumerate(self.vehicles_on_road):
            junc_id = self.map.get_junction(road)
            if len(self.vehicles_on_road[road])==0 or junc_id=="-1": continue
            sorted_vehicles = sorted(self.vehicles_on_road[road])
            for s, vehicle_id in sorted_vehicles:
                assert junc_id not in self.junction_hazzard
                self.junction_hazzard[junc_id]=vehicle_id
        for i, ped in enumerate(self.simple_avatars):
            road_id, s = ped.get_loc()
            junc_id = self.map.get_junction(road_id)
            if junc_id=="-1": continue
            if junc_id not in self.junction_hazzard:
                self.junction_hazzard[junc_id]=f"ped_{i}"
        for rid, road in enumerate(self.vehicles_on_road):
            if len(self.vehicles_on_road[road])==0: continue
            sorted_vehicles = sorted(self.vehicles_on_road[road])
            for vid, sorted_vehicle in enumerate(sorted_vehicles):
                s, vehicle_id = sorted_vehicle
                # speed = np.random.normal(loc=self.uniform_speed, scale=0.1)
                speed = self.uniform_speed
                if self.dry_run==False:
                    if self.vehicles[vehicle_id].spare()==False:
                        self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'stop', 'next_road': road, 'arg1': "vehicle is still moving or turning."}
                        continue
                # halt if there is a collision ahead
                if vid+1<len(sorted_vehicles):
                    if s+speed+self.min_dis>=sorted_vehicles[vid+1][0]:
                        self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'stop', 'next_road': road, 'arg1': f"collision with {sorted_vehicles[vid+1][1]} ahead on the same road"}
                        continue
                # if the vehicle is going into the next road...
                if s+speed>self.map.get_length(road):
                    next_road=self.smart_get_next_road(road)
                    if next_road is None:
                        self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'stop', 'next_road': road, 'arg1': "no road ahead"}
                        continue
                    # print(f"road: {road}, next: {next_road}")
                    assert next_road!=road
                    junc_id = self.map.get_junction(next_road)
                    if junc_id!="-1":
                        if junc_id in self.junction_hazzard:
                            self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'stop', 'next_road': road, "arg1": f"junction_hazzard with {self.junction_hazzard[junc_id]}"}
                            continue
                        self.junction_hazzard[junc_id]=vehicle_id
                        self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'move_forward', 'arg1': self.map.get_length(road)-s, 'next_road': next_road}
                        # no need of checking vehicles in next road since its in a junction
                        continue
                    else:
                        tmp_list=sorted(self.vehicles_on_road[next_road])# nearest vehicle on the next road
                        if len(tmp_list)>0 and tmp_list[0][0]<self.min_dis:
                            self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'stop', 'next_road': road, "arg1": f"{tmp_list[0][1]} on next road"}
                        else:
                            self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'move_forward', 'arg1': self.map.get_length(road)-s, 'next_road': next_road}
                        continue
                self.vehicle_action[vehicle_id]={'id': vehicle_id, 'type': 'move_forward', 'arg1': speed, 'next_road': road}

    def act(self):
        # update self.vehicles_on_road
        for road_id in self.map.get_roads().keys():
            self.vehicles_on_road[road_id]=list()
        # update vehicles
        for i in range(self.vehicle_number):
            next_road_id=self.vehicle_action[i]['next_road']
            road_id, s=self.simple_vehicles[i].get_loc()
            # print(f"road_id: {road_id}, s: {s}")
            if self.vehicle_action[i]['type']=='move_forward':
                s+=self.vehicle_action[i]['arg1']
                if road_id != next_road_id:
                    s=0.
                self.simple_vehicles[i].set_loc(next_road_id,s)
            if self.vehicle_action[i]['type']=='stop':
                pass
            self.vehicles_on_road[next_road_id].append((s, i))
            # print(f"next_road_id: {next_road_id}, s: {s}")
            self.simple_vehicles[i].set_pos(self.map.get_pos(next_road_id,s))
            self.simple_vehicles[i].set_rot(self.map.get_rot(next_road_id,s))
    
    def schedule(self):
        self.plan()
        self.move_pedestrians()
        self.act()

    def spare(self):
        return all([vehicle.spare() for vehicle in self.vehicles]) and all([avatar.spare() for avatar in self.avatars])

    def step(self):
        if self.enable_traffic:
            self.schedule()
        # if self.enable_tm_debug:
        #     self.logger.info(self.vehicle_action)
        for vehicle in self.vehicles:
            vehicle.step()
        for avatar in self.avatars:
            avatar.step()

    def get_vehicles_pos(self):
        return [vehicle.get_pos() for vehicle in self.simple_vehicles]

    def get_pedestrians_pos(self):
        return [ped.get_pos() for ped in self.simple_avatars]
    
    def is_full(self, road):
        tmp_list=sorted(self.vehicles_on_road[road])# nearest vehicle on the next road
        if len(tmp_list)>0 and tmp_list[0][0]<self.min_dis:
            return True
        else:
            return False
        
    def smart_get_next_road(self, road):
        next_roads = self.map.get_next_roads(road, valid=True)
        next_roads = [next_road for next_road in next_roads if not self.is_full(road)]
        if len(next_roads)==0:
            return self.map.get_next_road(road)
        if len(next_roads)==1:
            return next_roads[0]
        if len(next_roads)>1:
            next_roads = [next_road for next_road in next_roads if len([nnr for nnr in self.map.get_next_roads(next_road, valid=True) if not self.is_full(nnr)])>0]
            if len(next_roads)>0:
                return random.choice(next_roads)
            else:
                return self.map.get_next_road(road)

    def init_post_scene_build(self):
        bus_pose = self.bus.update_at_time(self.env.curr_time)
        self.bus.reset(np.array(bus_pose[:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(bus_pose[3:], dtype=np.float64))))
        if "bicycle_poses" not in self.env.config:
            self.env.config["bicycle_poses"] = []
            bicycle_station_locations = list(self.env.transit_info["bicycle"]["stations"].values())
            for i in range(0, len(bicycle_station_locations)):
                bicycle_sampled_pos_xy = bicycle_station_locations[i] + np.random.uniform(-1, 1, 2)
                self.env.config["bicycle_poses"].append([bicycle_sampled_pos_xy[0], bicycle_sampled_pos_xy[1], 0.26*1.7, 0, 0, 0])
        for i in range(0, len(self.env.config["bicycle_poses"])):
            self.shared_bicycles.bicycles[i].reset(np.array(self.env.config['bicycle_poses'][i][:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(self.env.config['bicycle_poses'][i][3:], dtype=np.float64))))

    def reset(self):
        # todo: @xiangye @zheyuan
        for vehicle in self.vehicles:
            simple_vehicle = vehicle.simple_vehicle
            vehicle.reset(np.array(np.array(simple_vehicle.get_pos() + [3.0], dtype=np.float64), dtype=np.float64),
                          geom_utils.euler_to_R(np.degrees(np.array(simple_vehicle.get_rot(), dtype=np.float64))))
        for i, avatar in enumerate(self.avatars):
            simple_avatar = avatar.simple_avatar
            avatar.reset(np.array(np.array(simple_avatar.get_pos() + [3.0], dtype=np.float64), dtype=np.float64),
                         geom_utils.euler_to_R(np.degrees(np.array(simple_avatar.get_rot(), dtype=np.float64))))
        bus_pose = self.bus.update_at_time(self.env.curr_time)
        self.bus.reset(np.array(bus_pose[:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(bus_pose[3:], dtype=np.float64))))

def update(frame, traffic_manager):
    
    # Update vehicle positions (simple linear movement to the right)
    vehicle_positions = traffic_manager.get_vehicles_pos()
    pedestrian_positions = traffic_manager.get_pedestrians_pos()
    
    xs, ys = zip(*vehicle_positions)
    traffic_manager.schedule()
    
    # Update vehicle positions on the plot
    vehicles.set_data(xs,ys)
    pedestrians.set_data(zip(*pedestrian_positions))
    
    
    return vehicles, pedestrians, 



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", '-s', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(f"{get_assets_dir()}/scenes/{args.scene}/road_data/road_data.xodr"):
        print(f"{get_assets_dir()}/scenes/{args.scene}/road_data/road_data.xodr not exist!")
        exit()
    with open(f'assets/scenes/{args.scene}/raw/center.txt', "r") as file:
        for line in file:
            ref_lat, ref_lon = line.strip().split()
        ref_lat, ref_lon = float(ref_lat), float(ref_lon)
    local_map=LocalMap(file_path=f"{get_assets_dir()}/scenes/{args.scene}/road_data/road_data.xodr", terrain_height_path=None, ref_lat=ref_lat, ref_lon=ref_lon)
    traffic_manager = TrafficManager(None, args.scene, vehicle_number=60, dry_run=True, pedestrian_number=60)

    fig, ax = plt.subplots(figsize=(10, 6))
    aerial_view=mpimg.imread(f"{get_assets_dir()}/scenes/{args.scene}/global.png")
    plt.imshow(aerial_view, extent=[-512, 512, -512, 512])# left, right, bottom, top
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    road_data=local_map.get_roads()
    for road_id, road in road_data.items():
        color='blue'
        if road['junction']!="-1": color='cyan'
        if road_id in local_map.pedestrian_main_roads: color='yellow'
        for geometry in road['geometry']:
            s = float(geometry.get('s'))
            x = float(geometry.get('x'))
            y = float(geometry.get('y'))
            hdg = float(geometry.get('hdg'))
            length = float(geometry.get('length'))
            if geometry['type']=='paramPoly3':
                t = np.linspace(0, 1, 5)
                u_rotated, v_rotated = cubic_curve(t, geometry['shape'], hdg)
                u_rotated, v_rotated = u_rotated+x, v_rotated+y
                for i in range(u_rotated.shape[0]):
                    u_rotated[i], v_rotated[i] = corr_xy(u_rotated[i], v_rotated[i], ref_lat, ref_lon, local_map.offset_x, local_map.offset_y)
                ax.plot(u_rotated, v_rotated, color=color)
            else:
                end_x = x + length * np.cos(hdg)
                end_y = y + length * np.sin(hdg)
                x, y = corr_xy(x, y, ref_lat, ref_lon, local_map.offset_x, local_map.offset_y)
                end_x, end_y = corr_xy(end_x, end_y, ref_lat, ref_lon, local_map.offset_x, local_map.offset_y)
                ax.plot([x, end_x], [y, end_y], color=color)
    vehicle_positions = traffic_manager.get_vehicles_pos()
    pedestrian_positions = traffic_manager.get_pedestrians_pos()
    xs, ys = zip(*vehicle_positions)
    pxs, pys = zip(*pedestrian_positions)
    vehicles, = ax.plot(xs, ys, 'ro', markersize=5)
    pedestrians, = ax.plot(xs, ys, 'go', markersize=3)
    ani = animation.FuncAnimation(fig, update, frames=150, fargs=(traffic_manager,), repeat=False, interval=100)

    plt.title(f'Traffic Visualization - {args.scene}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    # plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    ani.save('traffic_visualization.gif', writer='pillow', fps=10)
    plt.show()