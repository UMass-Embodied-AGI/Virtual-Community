import json
import os
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
import time

import carla
from pathlib import Path

def create_opdr(scene_name):
    if os.path.exists(f"ViCo/assets/scenes/{scene_name}/road_data/road_data.osm"):
        with open(f"ViCo/assets/scenes/{scene_name}/road_data/road_data.osm","r", encoding="utf-8") as file:
            osm_data=file.read()
    else:
        assert 0 and "osm data not exist"
    try:
        # Define the desired settings. In this case, default values.
        settings = carla.Osm2OdrSettings()
        settings.proj_string='+proj=merc'
        # Set OSM road types to export to OpenDRIVE
        settings.set_osm_way_types(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"])
        # Convert to .xodr
        xodr_data = carla.Osm2Odr.convert(osm_data, settings)
        with open(f"ViCo/assets/scenes/{scene_name}/road_data/road_data.xodr","w", encoding='utf-8') as file:
            file.write(xodr_data)
    except Exception as e:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    args = parser.parse_args()
    if os.path.exists(f"ViCo/assets/scenes/{args.scene}/road_data"):
        print("Scene exists")
    else:
        print(f"Scene not exist: ViCo/assets/scenes/{args.scene}/road_data")
        exit()
    create_opdr(scene_name=args.scene)

    # Set up the plot area
    plt.figure(figsize=(8, 8))
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()

    # Show the plot
    plt.title(f'Roads with Width - {args.scene}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()