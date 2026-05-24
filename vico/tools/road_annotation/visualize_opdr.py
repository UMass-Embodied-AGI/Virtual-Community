from vico.tools.constants import ASSETS_PATH
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import argparse
import numpy as np
import pyproj
import math

def parse_opendrive(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    offset_x = float(root.find('header/offset').get('x'))
    offset_y = float(root.find('header/offset').get('y'))
    roads = []
    
    # Extract road geometries
    for road in root.findall('road'):
        for geometry in road.find('planView').findall('geometry'):
            s = float(geometry.get('s'))
            x = float(geometry.get('x'))
            y = float(geometry.get('y'))
            hdg = float(geometry.get('hdg'))
            length = float(geometry.get('length'))
            roads.append((s, x, y, hdg, length))
    
    return roads, offset_x, offset_y

def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    earth_radius = 6378137  # in meters
    meters_per_degree_lat = 111139  # Approximate meters per degree latitude
    
    # Calculate meters per degree longitude based on reference latitude
    meters_per_degree_lon = 111139 * math.cos(math.radians(ref_lat))
    
    # Convert lat/lon to x/y
    x = (lon - ref_lon) * meters_per_degree_lon
    y = (lat - ref_lat) * meters_per_degree_lat
    
    return [x, y]

def visualize_roads(roads, ref_lat=0., ref_lon=0., offset_x=0., offset_y=0.):
    
    inProj = pyproj.Proj("+proj=merc")
    outProj = pyproj.Proj(init='epsg:4326')  # WGS84
    lon, lat = pyproj.transform(inProj, outProj, -offset_x, -offset_y)
    print(f"Latitude: {lat}, Longitude: {lon}")
    plt.figure(figsize=(10, 6))
    for s,x,y,hdg,length in roads:
        end_x = x + length * np.cos(hdg)
        end_y = y + length * np.sin(hdg)
        lon, lat = pyproj.transform(inProj, outProj, x-offset_x, y-offset_y)
        end_lon, end_lat = pyproj.transform(inProj, outProj, end_x-offset_x, end_y-offset_y)
        x, y = lat_lon_to_xy(lat, lon, ref_lat, ref_lon)
        end_x, end_y = lat_lon_to_xy(end_lat, end_lon, ref_lat, ref_lon)
        # print(lat, lon)
        plt.plot([x, end_x], [y, end_y], color='blue')
    plt.title('OpenDRIVE Road Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    # plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    # plt.savefig("modules/trafficmanager/opdr.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    args = parser.parse_args()
    # Example usage
    file_path = f"{ASSETS_PATH}/scenes/{args.scene}/road_data/road_data.xodr"  # Replace with your OpenDRIVE file path
    print(f"visualizing {file_path}...")
    roads_data, offset_x, offset_y = parse_opendrive(file_path)

    # Define the Mercator projection parameters
    # inProj = pyproj.Proj("+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    inProj = pyproj.Proj("+proj=merc")
    outProj = pyproj.Proj(init='epsg:4326')  # WGS84

    # Example x-y coordinates (replace with actual values)
    x = -offset_x  # Replace with your x coordinate
    y = -offset_y  # Replace with your y coordinate

    ref_lat = 42.33165461030516
    ref_lon = -83.0480662316049

    # Convert from x-y to lat-lon
    lon, lat = pyproj.transform(inProj, outProj, x, y)
    ref_x, ref_y = pyproj.transform(outProj, inProj, ref_lon, ref_lat)
    # print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"ref_x: {ref_x}, ref_y: {ref_y}")
    print(f"del_x: {ref_x+offset_x}, del_y: {ref_y+offset_y}")

    visualize_roads(roads_data, ref_lat, ref_lon, offset_x, offset_y)