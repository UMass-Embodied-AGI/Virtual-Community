from vico.tools.constants import ASSETS_PATH
import numpy as np
import math
import pickle
import argparse

def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    earth_radius = 6378137  # in meters
    meters_per_degree_lat = 111139  # Approximate meters per degree latitude
    
    # Calculate meters per degree longitude based on reference latitude
    meters_per_degree_lon = 111139 * math.cos(math.radians(ref_lat))
    
    # Convert lat/lon to x/y
    x = (lon - ref_lon) * meters_per_degree_lon
    y = (lat - ref_lat) * meters_per_degree_lat
    
    return [x, y]

def point_to_road_distance(point, road, ref_lat, ref_lon):
    """Calculate the distance from a point to a line segment."""
    # Convert points to numpy arrays for easier calculations
    point = np.array(point)
    start = np.array(lat_lon_to_xy(road['start']['lat'],road['start']['lon'],ref_lat,ref_lon))
    end = np.array(lat_lon_to_xy(road['end']['lat'],road['end']['lon'],ref_lat,ref_lon))
    
    # Vector from start to end
    segment_vector = end - start
    # Vector from start to point
    point_vector = point - start
    
    # Project point_vector onto segment_vector
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    if segment_length_squared == 0:
        # The segment is a single point
        return np.linalg.norm(point - start), start  # Distance and closest point is start
    
    # Calculate projection scalar
    projection = np.dot(point_vector, segment_vector) / segment_length_squared
    
    if projection < 0:
        # Closest to start
        closest_point = start
    elif projection > 1:
        # Closest to end
        closest_point = end
    else:
        # Closest on the segment
        closest_point = start + projection * segment_vector
    
    # Calculate distance from point to closest point on segment
    distance = np.linalg.norm(point - closest_point)
    
    return distance, closest_point, road['width']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--ref_lat", type=float, required=True)
    parser.add_argument("--ref_lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()
    # lat, lon = 40.748998486718186, -73.9882893780644 # x, y
    [x,y]=lat_lon_to_xy(args.lat, args.lon, args.ref_lat, args.ref_lon)
    roads, nodes = pickle.load(open(f"{ASSETS_PATH}/scenes/{args.scene}/roads.pkl", 'rb'))
    min_distance = float('inf')
    closest_road = None
    for road in roads:
        distance, closest_point, width = point_to_road_distance([x,y], road, args.ref_lat, args.ref_lon)
        if distance < min_distance:
            min_distance = distance
            closest_road = road
    print(closest_road)