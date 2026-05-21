import requests
import os
import sys
from multiprocessing import Pool
import pymap3d as pm
import re
import io
from collections import defaultdict
import cv2
import math
import json
import random
import difflib
import argparse
import colorsys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

import genesis as gs

current_directory = os.getcwd()
sys.path.append(current_directory)
from tools.constants import google_map_type_to_coarse, google_map_coarse_to_types, coarse_types_priority
from tools.utils import *

def flatten(places):
    # Flatten the list of lists of dictionaries to a list of dictionaries
    flat_list = [item for sublist in places for item in sublist]
    return flat_list

def remove_duplicates(places):
    # Remove duplicates based on 'place_id'
    seen = set()
    new_places = []
    for d in places:
        if d['place_id'] not in seen:
            new_places.append(d)
            seen.add(d['place_id'])
    return new_places

def save_to_json(places, filename='places.json'):
    # Write the list of dictionaries to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(places, json_file, indent=4)

def search_places_single(lat, lng, radius, api_key):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "rankby": "distance",
        # "radius": radius,
        "key": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.json()

def search_places_all(lat, lng, radius, api_key): # deprecated
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": api_key,
    }
    all_results = []
    while True:
        response = requests.get(base_url, params=params)
        response_data = response.json()
        all_results.extend(response_data.get("results", []))
        next_page_token = response_data.get("next_page_token")
        if not next_page_token:
            break
        params["pagetoken"] = next_page_token
    return all_results

def search_places_single_results(lat, lng, radius, api_key):
    return search_places_single(lat, lng, radius, api_key)['results']
    # base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    # params = {
    #     "location": f"{lat},{lng}",
    #     "radius": radius,
    #     "key": api_key,
    # }
    # all_results = []
    # while True:
    #     response = requests.get(base_url, params=params)
    #     response_data = response.json()
    #     all_results.extend(response_data.get("results", []))
    #     next_page_token = response_data.get("next_page_token")
    #     if not next_page_token:
    #         break
    #     params["pagetoken"] = next_page_token
    # return all_results

def search_place_details(place_id, api_key):
    base_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.json()

def get_photo(photo_reference, api_key, maxwidth=400):
    base_url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {
        "maxwidth": maxwidth,
        "photoreference": photo_reference,
        "key": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.content

def search_places_in_area(lat, lng, radius, resolution, api_key, processes = 1):
    # create an empty json object to store the results
    places = []
    to_search = []

    #create a grid of points to search
    #radius is in meters, so we need to convert it to degrees
    #1 degree is approximately 111111 meters
    radius_deg = radius / 111111
    #convert resolution to degrees too
    resolution_deg = resolution / 111111
    #Create a grid of points to search
    #Since the numbers are all floats, we can't use range directly
    #So we should use a while loop instead
    lat_c = lat - radius_deg
    while lat_c < lat + radius_deg:
        lng_c = lng - radius_deg
        while lng_c < lng + radius_deg:
            to_search.append((lat_c, lng_c))
            lng_c += resolution_deg
        lat_c += resolution_deg

    if processes != 1:
        assert processes > 1
        with Pool(processes) as p:
            places = p.starmap(search_places_single_results, [(lat_c, lng_c, resolution, api_key) for lat_c, lng_c in to_search])
        places = flatten(places)
        places = remove_duplicates(places)
        
    else:
        for lat_c, lng_c in to_search:
            #print("Searching", lat_c, lng_c)
            places += search_places_single(lat_c, lng_c, resolution, api_key)['results']
            places = remove_duplicates(places)
    # places = search_places_all(lat, lng, radius, api_key)
    return places

def search_original_places(args, scene_range_meta, api_key):
    print("Start searching places...")
    j = search_places_in_area(scene_range_meta["lat"], scene_range_meta["lng"], scene_range_meta["rad"], args.search_resolution, api_key, 8)
    print("Found", len(j), "places")
    save_to_json(j, f'tools/scene/temp/{args.scene}_places_original.json')
    #convert all lla positions to enu
    with open(f'tools/scene/temp/{args.scene}_places_original.json') as f:
        with open(f'tools/scene/temp/{args.scene}_places_enu_original.json', 'w') as g:
            j = f.readlines()
            # for each line, scan if it is in a format "lat": ... or "lng": ...
            temp_lat, temp_lng = 0, 0
            for i, line_ori in enumerate(j):
                #count the starting spaces of the line
                space_num = line_ori.find(line_ori.strip())
                line = line_ori.strip()
                if line.startswith('"lat":'):
                    temp_lat = float(line.split(':')[1].strip().strip(','))
                    if temp_lng != 0:
                        assert False, "lat and lng are not in pairs"
                elif line.startswith('"lng":'):
                    temp_lng = float(line.split(':')[1].strip().strip(','))
                    if temp_lat == 0:
                        assert False, "lat and lng are not in pairs"
                    #convert to enu
                    x, y, z = pm.geodetic2enu(temp_lat, temp_lng, 0, scene_range_meta["lat"], scene_range_meta["lng"], 0)
                    g.write(' '*space_num + f'"x": {x},\n')
                    g.write(' '*space_num + f'"y": {y},\n')
                    g.write(' '*space_num + f'"z": {z},\n')
                    temp_lat, temp_lng = 0, 0
                else:
                    g.write(line_ori)

def filter_places(args):
    print("Start filtering places...")
    filter_types = ["route", "locality", "political", "neighborhood"]
    filter_names = ["animal"]
    with open(f"tools/scene/temp/{args.scene}_places_enu_original.json", 'r') as file:
        json_text = file.read()
    json_text_cleaned = re.sub(r',\s*([}\]])', r'\1', json_text) # remove trailing commas before closing braces or brackets
    places = json.loads(json_text_cleaned)
    filtered_places = []
    for place in places:
        if not any(type in place["types"] for type in filter_types) and not any(name == place["name"] for name in filter_names):
            if abs(place["geometry"]["location"]["x"]) < args.filter_distance_square and abs(place["geometry"]["location"]["y"]) < args.filter_distance_square:
                filtered_places.append(place)

    print("# filtered places:", len(filtered_places))
    with open(f"tools/scene/temp/{args.scene}_places_enu.json", 'w') as json_file:
        json.dump(filtered_places, json_file, indent=4)
    if args.remove_temp:
        os.remove(f"tools/scene/temp/{args.scene}_places_enu_original.json")

def save_metadata(args):
    print("Start saving to environment metadata...")
    with open(f"tools/scene/temp/{args.scene}_places_enu.json", 'r') as file:
        json_text = file.read()
    json_text_cleaned = re.sub(r',\s*([}\]])', r'\1', json_text) # remove trailing commas before closing braces or brackets
    places = json.loads(json_text_cleaned)
    if os.path.exists("env_places_metadata.json"):
        with open("env_places_metadata.json", 'r') as file:
            metadata = json.load(file)
    else:
        metadata = {}
    with open("tools/scene/place_type_annotations.json", 'r') as file:
        place_type_annotations = json.load(file)
    metadata = {}
    for place in places:
        metadata[place["name"]] = {}
        metadata[place["name"]]["location"] = [place["geometry"]["location"]["x"], 
                                                place["geometry"]["location"]["y"], 
                                                place["geometry"]["location"]["z"]]
        if "types" in place.keys():
            metadata[place["name"]]["types"] = place["types"]
        if args.scene in place_type_annotations["living"]:
            if place["name"] in place_type_annotations["living"][args.scene]:
                if "types" not in place.keys():
                    metadata[place["name"]]["types"] = []
                metadata[place["name"]]["types"].append(place_type_annotations["living"][args.scene][place["name"]])
        if "rating" in place.keys():
            metadata[place["name"]]["rating"] = place["rating"]
        if "user_ratings_total" in place.keys():
            metadata[place["name"]]["user_ratings_total"] = place["user_ratings_total"]
        if "vicinity" in place.keys():
            metadata[place["name"]]["vicinity"] = place["vicinity"]
    with open(f"assets/{args.scene}/raw/places_full.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    if args.remove_temp:
        os.remove(f"tools/scene/temp/{args.scene}_places_enu.json")

def bbox3d_to_bbox2d(bbox3d):
    x_coords = [point[0] for point in bbox3d]
    y_coords = [point[1] for point in bbox3d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return min_x, min_y, max_x, max_y

def find_most_similar_string(input_string, string_list):
    if not string_list:
        return None
    closest_matches = difflib.get_close_matches(input_string, string_list, n=1, cutoff=0.7)
    return closest_matches[0] if closest_matches else None

def generate_diverse_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        color = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    return colors

def generate_legend(mapping):
    num_items = len(mapping)
    fig, ax = plt.subplots(figsize=(4, 0.4 * num_items))
    for i, (label, color) in enumerate(mapping.items()):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=np.array(color)/255))
        ax.text(1.1, i + 0.5, label, va='center', ha='left', fontsize=12)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, num_items)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return Image.open(buf)

def find_closest_bounding_box(query_point):
    closest_building = None
    min_distance = float('inf')
    debug_list = []

    for building_name in building_to_osm_tags:
        assert "custom:bounding_box" in building_to_osm_tags[building_name], f"Building {building_name} has no bounding box!"
        building_to_osm_tags_3d = building_to_osm_tags[building_name]["custom:bounding_box"]
        min_x, min_y, max_x, max_y = bbox3d_to_bbox2d(building_to_osm_tags_3d)

        if min_x < query_point[0] < max_x and min_y < query_point[1] < max_y:
            debug_list.append((building_name, building_to_osm_tags[building_name]["custom:bounding_box"]))

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            distance = math.sqrt((center_x - query_point[0]) ** 2 + (center_y - query_point[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_building = building_name
    # if len(debug_list) > 1: print("debug list:", debug_list)
    return closest_building, debug_list


def get_bbox_center(bbox):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    z_coords = [point[2] for point in bbox]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    return [center_x, center_y, center_z]

def stitch_images_horizontally(image1, image2):
    combined_width = image1.width + image2.width
    combined_height = max(image1.height, image2.height)
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    return combined_image

def overlay_locations_desp_on_image_old():
    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size
    orthographic_scale = int(image_path.split('_')[-1].split('.')[0])
    meter_to_pixel = width / orthographic_scale
    coarse_types = list(google_map_coarse_to_types.keys())
    type_to_color = dict(zip(coarse_types, generate_diverse_colors(len(coarse_types))))
    legend_image = generate_legend(type_to_color)

    for building in building_to_places:
        for i, place in enumerate(building_to_places[building]):
            if building == "open space":
                x, y = places_dict[place["name"]]["location"][:2]
            else:
                if not args.verbose and i > 3: break
                x, y = building_to_osm_tags[building]["custom:bounding_box"][0][:2]
            pixel_x = int(width / 2 + x * meter_to_pixel)
            pixel_y = int(height / 2 - y * meter_to_pixel) + i * 12
            text = place["name"]
            color = type_to_color[place["coarse_type"]]
            draw.text((pixel_x, pixel_y), text, font=ImageFont.truetype("assets/arial.ttf", 12), fill=color)

    # annotated_image.save(image_path.split('.')[0] + "_annotated.png")
    annotated_image = stitch_images_horizontally(annotated_image, legend_image)
    annotated_image.save(image_path.split('.')[0] + "_annotated.png")

def overlay_locations_desp_on_image():
    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    coarse_types = list(google_map_coarse_to_types.keys())
    type_to_color = dict(zip(coarse_types, generate_diverse_colors(len(coarse_types))))
    legend_image = generate_legend(type_to_color)

    for building in building_to_places:
        for i, place in enumerate(building_to_places[building]):
            if building == "open space":
                x, y = places_dict[place["name"]]["location"][:2]
            else:
                if not args.verbose and i > 3: break
                x, y = building_to_osm_tags[building]["custom:bounding_box"][0][:2]
            pixel_x, pixel_y = project_3d_to_2d_from_perspective_camera(np.array([x, y, get_height_at(height_field, x, y)]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"]))
            text = place["name"]
            color = type_to_color[place["coarse_type"]]
            draw.text((pixel_x, pixel_y), text, font=ImageFont.truetype("assets/arial.ttf", 20), fill=color)

    annotated_image = stitch_images_horizontally(annotated_image, legend_image)
    annotated_image.save(f"assets/{args.scene}/global_annotated.png")

def bbox_vis_old(buildings, title):
    bbox_list = []
    for building in buildings:
        if "bounding_box" in buildings[building]:
            bbox_list.append(buildings[building]["bounding_box"])
        elif "custom:bounding_box" in buildings[building]:
            bbox_list.append(buildings[building]["custom:bounding_box"])
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be loaded.")

    height, width, _ = image.shape
    orthographic_scale = int(image_path.split('_')[-1].split('.')[0])
    meter_to_pixel = width / orthographic_scale

    fig, ax = plt.subplots()
    ax.imshow(image)

    for building_to_osm_tags_3d in bbox_list:
        if building_to_osm_tags_3d is not None:
            x_coords = [point[0] for point in building_to_osm_tags_3d]
            y_coords = [point[1] for point in building_to_osm_tags_3d]
            min_x, max_x = width / 2 + min(x_coords) * meter_to_pixel, width / 2 + max(x_coords) * meter_to_pixel
            min_y, max_y = height / 2 - min(y_coords) * meter_to_pixel, height / 2 - max(y_coords) * meter_to_pixel

            rect = patches.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                linewidth=2,
                edgecolor='red',
                facecolor='green',
                alpha=0.5
            )

            ax.add_patch(rect)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    plt.axis('off')
    plt.title(title)
    plt.show()

def bbox_vis(buildings, title):
    bbox_list = []
    for building in buildings:
        if "bounding_box" in buildings[building]:
            bbox_list.append(buildings[building]["bounding_box"])
        elif "custom:bounding_box" in buildings[building]:
            bbox_list.append(buildings[building]["custom:bounding_box"])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be loaded.")

    fig, ax = plt.subplots()
    ax.imshow(image)

    for building_to_osm_tags_3d in bbox_list:
        if building_to_osm_tags_3d is not None:
            projected_points_2d = [project_3d_to_2d_from_perspective_camera(np.array([point[0], point[1], point[2] + 100]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"])) for point in building_to_osm_tags_3d]
            x_coords = [point[0] for point in projected_points_2d]
            y_coords = [point[1] for point in projected_points_2d]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            rect = patches.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                linewidth=2,
                edgecolor='red',
                facecolor='green',
                alpha=0.5
            )

            ax.add_patch(rect)

    # ax.set_xlim(0, width)
    # ax.set_ylim(height, 0)
    plt.axis('off')
    plt.title(title)
    plt.show()

def random_point_on_bbox_edge(min_x, min_y, max_x, max_y, abs_max=385, max_retry_times=100):
    found = False
    retry_times = 0
    while not found:
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            x = random.uniform(min_x, max_x)
            y = min_y
        elif edge == 'bottom':
            x = random.uniform(min_x, max_x)
            y = max_y
        elif edge == 'left':
            x = min_x
            y = random.uniform(min_y, max_y)
        else:
            x = max_x
            y = random.uniform(min_y, max_y)
        if abs(x) < abs_max and abs(y) < abs_max:
            found = True
            xy = (x, y)
        retry_times += 1
        if retry_times > max_retry_times:
            return None
    return xy

def is_point_in_bounding_box(x, y, bounding_box_3d):
    x_coords = [point[0] for point in bounding_box_3d]
    y_coords = [point[1] for point in bounding_box_3d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return min_x <= x <= max_x and min_y <= y <= max_y

def sample_location_on_extended_bounding_box(bounding_box_3d, all_bounding_boxes_3d, extension=1, max_retry_times=100):
    found = False
    retry_times = 0
    while not found:
        x_coords = [point[0] for point in bounding_box_3d]
        y_coords = [point[1] for point in bounding_box_3d]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        extended_min_x, extended_max_x = min_x - extension, max_x + extension
        extended_min_y, extended_max_y = min_y - extension, max_y + extension
        random_point = random_point_on_bbox_edge(extended_min_x, extended_min_y, extended_max_x, extended_max_y)
        if random_point is not None:
            found = True
            for each_bounding_box_3d in all_bounding_boxes_3d:
                if each_bounding_box_3d is not None:
                    if is_point_in_bounding_box(random_point[0], random_point[1], each_bounding_box_3d):
                        found = False
                        break
        retry_times += 1
        if retry_times > max_retry_times:
            return None
    return random_point

def accessibility_filtering_old(buildings, building_to_osm_tags): # deprecated
    accessible_buildings = dict(buildings)
    inaccessible_buildings = {}
    for name1, building1 in buildings.items():
        for name2, building2 in building_to_osm_tags.items():
            if name1 == name2:
                continue
            if building1['bounding_box'] == None or building2['custom:bounding_box'] == None:
                continue
            min_x1, min_y1, max_x1, max_y1 = bbox3d_to_bbox2d(building1['bounding_box'])
            min_x2, min_y2, max_x2, max_y2 = bbox3d_to_bbox2d(building2['custom:bounding_box'])
            if (min_x1 >= min_x2 and max_x1 <= max_x2 and
                min_y1 >= min_y2 and max_y1 <= max_y2):
                inaccessible_buildings[name1] = buildings[name1]
                inaccessible_buildings[name1]["overlapped_building"] = buildings[name2] if name2 in buildings else building_to_osm_tags[name2]
                if name1 in accessible_buildings:
                    del accessible_buildings[name1]
    return accessible_buildings, inaccessible_buildings

def accessibility_filtering(buildings, building_to_osm_tags):
    bbox_list = []
    for building in building_to_osm_tags:
        if "custom:bounding_box" in building_to_osm_tags[building] and building_to_osm_tags[building]["custom:bounding_box"] is not None:
            bbox_list.append(building_to_osm_tags[building]["custom:bounding_box"])
    accessible_buildings = dict(buildings)
    inaccessible_buildings = {}
    for name, building in buildings.items():
        if name != "open space":
            sampled_point = sample_location_on_extended_bounding_box(building['bounding_box'], bbox_list)
            if sampled_point is None:
                inaccessible_buildings[name] = buildings[name]
                del accessible_buildings[name]
            else:
                if any(abs(coords[0]) > 500 or abs(coords[1]) > 500 for coords in building['bounding_box']):
                    inaccessible_buildings[name] = buildings[name]
                    del accessible_buildings[name]
                else:
                    accessible_buildings[name]["outdoor_xy"] = sampled_point
    return accessible_buildings, inaccessible_buildings
            
def remove_none_bbox_entries(buildings): # deprecated
    buildings_copy = dict(buildings)
    for name in buildings:
        if buildings_copy[name]["bounding_box"] == None:
            del buildings_copy[name]
            if args.verbose:
                print(f"Building '{name}' deleted.")
    return buildings_copy

def get_building_to_places():
    building_metadata = {}
    type_stats = defaultdict(int)
    place_metadata = {}
    building_to_places = defaultdict(list)
    mismatched = []
    debug_overlap = []
    building_street_address = {}
    for building_name in building_to_osm_tags:
        # building_metadata[building_name] = {
        #     "bounding_box": building_to_osm_tags[building_name]["custom:bounding_box"],
        #     "places": [],
        # }
        name_street = building_name
        if "addr:housenumber" in building_to_osm_tags[building_name]:
            name_street += ' ' + building_to_osm_tags[building_name]["addr:housenumber"]
        if "addr:street" in building_to_osm_tags[building_name]:
            name_street += ' ' + building_to_osm_tags[building_name]["addr:street"]
        building_street_address[name_street] = building_name

    for place_name in places_dict:
        if "park" in places_dict[place_name]["types"]:
            # open spaces
            building_to_places["open space"].append({"name": place_name, "coarse_type": "open", "fine_types": places_dict[place_name]["types"], "location": places_dict[place_name]["location"]})
            type_stats["open"] += 1
            continue
        # 1. location point is within the bounding box of the building
        building, overlapped_list = find_closest_bounding_box(places_dict[place_name]["location"])
        # if len(overlapped_list) > 1:
        #     debug_overlap.append(set(overlap[0] for overlap in overlapped_list))
        if building is None:
            # alternative method: use street name and building name to do string matching
            if "vicinity" in places_dict[place_name]:
                street_name = places_dict[place_name]["vicinity"]
                query_place_name_street = place_name + ' ' + street_name
                most_similar_in_building_to_osm_tags = find_most_similar_string(query_place_name_street, list(building_street_address.keys()))
                if most_similar_in_building_to_osm_tags:
                    building = building_street_address[most_similar_in_building_to_osm_tags]
                if args.verbose:
                    print(f"Warn: Place {place_name} has no matched bounding box! Query '{query_place_name_street}':", most_similar_in_building_to_osm_tags)
        if building:
            # 2. has a coarse type of interest
            place_types = places_dict[place_name]["types"]
            coarse_types = [google_map_type_to_coarse[place_type] for place_type in place_types if
                            place_type in google_map_type_to_coarse]
            if len(coarse_types) == 0:
                if args.verbose:
                    print(f"Warn: Place {place_name} has no coarse type of interest! Place types: {place_types}")
            if len(coarse_types) > 0:
                coarse_type = max(coarse_types, key=lambda t: coarse_types_priority[t])
                type_stats[coarse_type] += 1
                building_to_places[building].append({"name": place_name, "coarse_type": coarse_type, "fine_types": place_types, "location": places_dict[place_name]["location"]})
                # building_metadata[building]["places"].append({"name": place_name, "coarse_type": coarse_type, "fine_types": place_types, "location": places_dict[place_name]["location"]})
        else:
            mismatched.append(place_name)

    unmatched_buildings = []
    for building_name in building_to_osm_tags:
        osm_types = []
        if "building" in building_to_osm_tags[building_name]:
            osm_types.append(building_to_osm_tags[building_name]["building"])
        if "amenity" in building_to_osm_tags[building_name]:
            osm_types.append(building_to_osm_tags[building_name]["amenity"])
        if "tourism" in building_to_osm_tags[building_name]:
            osm_types.append(building_to_osm_tags[building_name]["tourism"])
        coarse_types = [google_map_type_to_coarse[osm_type] for osm_type in osm_types if osm_type in google_map_type_to_coarse]
        if len(coarse_types) > 0:
            already_exist = False
            coarse_type = max(coarse_types, key=lambda t: coarse_types_priority[t])
            for places in building_to_places[building_name]:
                if places["coarse_type"] == coarse_type:
                    already_exist = True
                    break
            if already_exist:
                continue
            type_stats[coarse_type] += 1
            building_to_places[building_name].append(
                {"name": building_name, "coarse_type": coarse_type, "fine_types": osm_types, "location": get_bbox_center(building_to_osm_tags[building_name]["custom:bounding_box"])})

    for building_name, places in building_to_places.items():
        if len(places) == 0:
            if args.verbose:
                print(f"Warn: Building {building_name} has no matched place!")
            unmatched_buildings.append(building_name)
            continue
        if building_name == "open space":
            building_metadata[building_name] = {
                "bounding_box": None,
                "places": places,
            }
        if building_name.startswith("element"):
            assert places[0]["name"] in places_dict, f"Place {places[0]['name']} not found in places_dict! but the building name is {building_name}"
            real_name = places_dict[places[0]["name"]]["vicinity"].split(',')[0]
        else:
            real_name = building_name
        building_center = get_bbox_center(building_to_osm_tags[building_name]["custom:bounding_box"]) if building_name in building_to_osm_tags else None
        if building_center is not None: 
            building_center[2] = 0
        for i, place in enumerate(places):
            place_metadata[place["name"]] = {
                "building": real_name,
                "coarse_type": place["coarse_type"],
                "fine_types": place["fine_types"],
                "location": [building_center[0], building_center[1], building_center[2] - (i + 1) * 4] if building_center is not None else place["location"],
                "scene": f"assets/{args.scene}/places_scene/{place['name']}.json" if building_center is not None else None,
            }
        if building_center is not None:
            building_metadata[real_name] = {
                "bounding_box": building_to_osm_tags[building_name]["custom:bounding_box"],
                "places": [
                    {
                        "name": place["name"],
                        "coarse_type": place["coarse_type"],
                        "fine_types": place["fine_types"],
                        "location": place_metadata[place["name"]]["location"],
                        "scene": f"assets/{args.scene}/places_scene/{place['name']}.json",
                    } for place in places
                ]
            }
    accessible_buildings, inaccessible_buildings = accessibility_filtering(building_metadata, building_to_osm_tags)
    # print("inaccessible buildings:", inaccessible_buildings)
    json.dump(accessible_buildings, open(f"assets/{args.scene}/building_metadata.json", 'w'), indent=4)
    json.dump(inaccessible_buildings, open(f"assets/{args.scene}/raw/inaccessible_buildings.json", 'w'), indent=4)
    json.dump(update_place_metadata(place_metadata, accessible_buildings), open(f"assets/{args.scene}/place_metadata.json", 'w'), indent=4)
    json.dump(building_to_places, open(f"assets/{args.scene}/raw/building_to_places.json", 'w'), indent=4)
    json.dump(mismatched, open(f"assets/{args.scene}/raw/mismatched.json", 'w'), indent=4)
    # sort type_stats by key
    type_stats = dict(sorted(type_stats.items(), key=lambda item: item[0]))
    json.dump(type_stats, open(f"assets/{args.scene}/raw/type_stats.json", 'w'), indent=4)
    json.dump(generate_new_type_stats(accessible_buildings), open(f"assets/{args.scene}/raw/accessible_type_stats.json", 'w'))
    return accessible_buildings, inaccessible_buildings, building_to_places

def update_place_metadata(place_metadata, accessible_buildings):
    new_place_metadata = {}
    for place_name in place_metadata:
        if place_metadata[place_name]["building"] in accessible_buildings:
            new_place_metadata[place_name] = place_metadata[place_name]
    return new_place_metadata

def generate_new_type_stats(accessible_buildings):
    new_type_stats = {}
    for building_name in accessible_buildings:
        for place in accessible_buildings[building_name]["places"]:
            if place["coarse_type"] in new_type_stats:
                new_type_stats[place["coarse_type"]] += 1
            else:
                new_type_stats[place["coarse_type"]] = 1
    return new_type_stats

def update_building_to_places(building_to_places, accessible_buildings):
    new_building_to_places = {}
    for building_name in building_to_places:
        if building_name in accessible_buildings:
            new_building_to_places[building_name] = building_to_places[building_name]
    return new_building_to_places

def load_city_scene(scene, scene_assets_dir):
    scene.add_entity(
        material=gs.materials.Rigid(
            sdf_min_res=4,
            sdf_max_res=4,
        ),
        morph=gs.morphs.Mesh(
            file=os.path.join(scene_assets_dir, 'terrain.glb'),
            euler=(90.0, 0, 0),
            fixed=True,
            merge_submeshes_for_collision=False,
            group_by_material=True,
        ),
    )
    scene.add_entity(
        material=gs.materials.Rigid(
            sdf_min_res=4,
            sdf_max_res=4,
        ),
        morph=gs.morphs.Mesh(
            file=os.path.join(scene_assets_dir, 'buildings.glb'),
            euler=(90.0, 0, 0),
            fixed=True,
            merge_submeshes_for_collision=False,  # Buildings are constructed separately
            group_by_material=True,
        ),
    )

    scene.add_entity(
        material=gs.materials.Rigid(
            sdf_min_res=4,
            sdf_max_res=4,
        ),
        morph=gs.morphs.Mesh(
            file=os.path.join(scene_assets_dir, 'roof.glb'),
            euler=(90.0, 0, 0),
            fixed=True,
            collision=False,  # No collision needed for roof
            group_by_material=True,
        ),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--search_original_places", action="store_true")
    parser.add_argument("--filter_places", action="store_true")
    parser.add_argument("--filter_distance_square", type=float, required=True)
    parser.add_argument("--search_resolution", type=float, default=135.0)
    parser.add_argument("--save_metadata", action="store_true")
    parser.add_argument("--remove_temp", action="store_true")
    parser.add_argument("--visualize_metadata", action='store_true')
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("args:", args)
    random.seed(args.seed)

    # Check necessary files are existed
    if os.path.exists(f"assets/{args.scene}/raw/building_to_osm_tags.json"):
        print("Necessary file check passed: building_to_osm_tags.json")
    else:
        print(f"Necessary file not exist: assets/{args.scene}/raw/building_to_osm_tags.json")
        exit()

    if os.path.exists(f"assets/{args.scene}/raw/center.txt"):
        print("Necessary file check passed: center.txt")
    else:
        print(f"Necessary file not exist: assets/{args.scene}/raw/center.txt")
        exit()

    # if os.path.exists(f"assets/{args.scene}/orthographic_scale_800.png"):
    #     print("Necessary file check passed: orthographic_scale_800.png")
    # else:
    #     print(f "Necessary file not exist: assets/{args.scene}/orthographic_scale_800.png")
    #     exit()

    # Also check height field, despite not used for annotating the scene (used in character generation)
    final_folder_mapping = {"newyork": "NY_ok", "elpaso": "EL_PASO_ok"}
    if args.scene not in final_folder_mapping:
        final_folder_mapping[args.scene] = args.scene.upper() + "_ok"
    terrain_height_path=os.path.join(get_assets_dir(), f"ViCo/scene/final/{final_folder_mapping[args.scene]}/height_field.npz")
    
    if os.path.exists(terrain_height_path):
        print("Necessary file check passed: height_field.npz")
        height_field = np.load(terrain_height_path)
        if np.all(height_field["terrain_alt"] > 0):
            print("Height field all greater than 0. Passed.")
        else:
            print(f"Height field has values smaller or equal to 0. Please double check {terrain_height_path}.")
            exit()
    else:
        print(f"Necessary file not exist: {terrain_height_path}")
        exit()

    # Load height field as LinearNDInterpolatorExt
    height_field = load_height_field(terrain_height_path)

    if not os.path.exists(os.path.join(f"assets/{args.scene}/global.png")):
        print(f"assets/{args.scene}/global.png not exists, start loading scenes and take a global image of the scene from the perspective camera in Genesis")
        if not gs._initialized:
            gs.init(seed=0, precision="32", logging_level="info", backend=gs.cpu)

        gs_scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1000, 1000),
                camera_pos=np.array([0.0, 0.0, 1000]),
                camera_lookat=np.array([0, 0.0, 0.0]),
                camera_fov=60,
            ),
            sim_options=gs.options.SimOptions(),
            rigid_options=gs.options.RigidOptions(
                gravity=(0.0, 0.0, -9.8),
                enable_collision=False,
            ),
            avatar_options=gs.options.AvatarOptions(
                enable_collision=False,
            ),
            renderer=gs.renderers.Rasterizer(),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                segmentation_level="entity",
                lights=[{'type': 'directional',
                    'dir': (0, 0, -1),
                    'color': (1.0, 1.0, 1.0),
                    'intensity': 10.0}, ]
            ),
            show_viewer=False,
            show_FPS=False,
        )

        load_city_scene(gs_scene, os.path.join(get_assets_dir(), f"ViCo/scene/final/{final_folder_mapping[args.scene]}"))

        global_cam = gs_scene.add_camera(
            res=(2000, 2000),
            pos=(0.0, 0.0, 1000.0),
            lookat=(0, 0.0, 0.0),
            fov=60,
            GUI=False,
            far=16000.0
        )

        gs_scene.build()
        gs_scene.reset()

        global_rgb, _, _ = global_cam.render(depth=False)
        Image.fromarray(global_rgb).save(os.path.join(f"assets/{args.scene}/global.png"))
        print("Saved global image to asset folder.")

        global_cam_parameters = {}
        global_cam_parameters["camera_res"] = global_cam.res
        global_cam_parameters["camera_fov"] = global_cam.fov
        global_cam_parameters["camera_extrinsics"] = global_cam.extrinsics.tolist()
        with open(f"assets/{args.scene}/global_cam_parameters.json", "w") as f: 
            json.dump(global_cam_parameters, f)
        print("Saved global camera parameters to asset folder")
    else:
        global_cam_parameters = json.load(open(f"assets/{args.scene}/global_cam_parameters.json", 'r'))

    # Search places
    if args.search_original_places:
        with open('tools/scene/google_map_api.txt') as f:
            api_key = f.readline().strip()
    scene_range_meta = {}
    with open(f'assets/{args.scene}/raw/center.txt') as f:
        scene_range_meta["lat"], scene_range_meta["lng"] = map(float, f.read().strip().split(' '))
        scene_range_meta["rad"] = 400.0
    scene_range_meta["rad"] = scene_range_meta["rad"] * math.sqrt(2)
    if not os.path.exists(f"assets/{args.scene}/raw/places_full.json"):
        if args.search_original_places:
            search_original_places(args, scene_range_meta, api_key)
        if args.filter_places:
            filter_places(args)
        if args.save_metadata:
            save_metadata(args)
        print("Finish searching places.")
    else:
        print("Exists: places_full.json, skipping...")

    # Generate metadata
    # image_path = f"assets/{args.scene}/orthographic_scale_800.png"
    image_path = f"assets/{args.scene}/global.png"
    # Check if orthographic view exists, if not, create a white image
    if not os.path.exists(image_path):
        white_orthographic_image = Image.new("RGB", (1000, 1000), "white")
        white_orthographic_image.save(image_path)
    with open(f"assets/{args.scene}/raw/places_full.json", 'r') as file:
        places_dict = json.load(file)
    with open(f"assets/{args.scene}/raw/building_to_osm_tags.json", 'r') as file:
        building_to_osm_tags = json.load(file)
    accessible_buildings, inaccessible_buildings, building_to_places = get_building_to_places()
    if args.visualize_metadata:
        overlay_locations_desp_on_image()
        bbox_vis(building_to_osm_tags, "building_to_osm_tags buildings")
        bbox_vis(accessible_buildings, "accessible buildings")
        bbox_vis(inaccessible_buildings, "inaccessible buildings")
    print("Finished generating scene metadata.")