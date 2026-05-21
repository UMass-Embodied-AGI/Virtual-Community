import requests
import os
import sys
from multiprocessing import Pool
import re
import io
from collections import defaultdict
import cv2
import math
import json
import random
import difflib
import shutil
import argparse
import colorsys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import copy
import trimesh
import pickle

current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from tools.constants import google_map_type_to_coarse, google_map_coarse_to_types, coarse_types_priority, ASSETS_PATH
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
	# print("search_places_single response:", response.json())
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
	import pymap3d as pm
	j = search_places_in_area(scene_range_meta["lat"], scene_range_meta["lng"], scene_range_meta["rad"], args.search_resolution, api_key, 8)
	print("Found", len(j), "places.")
	if len(j) == 0:
		print("No places found. It's likely your API key is invalid or your API hasen't enabled Places API and Places (New) API.")
		exit()
	save_to_json(j, f'assets/scenes/{args.scene}/raw/places_original.json')
	#convert all lla positions to enu
	with open(f'assets/scenes/{args.scene}/raw/places_original.json') as f:
		with open(f'assets/scenes/{args.scene}/raw/places_enu_original.json', 'w') as g:
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
	filter_types = ["route", "locality", "political", "neighborhood"]
	filter_names = ["animal"]
	with open(f"assets/scenes/{args.scene}/raw/places_enu_original.json", 'r') as file:
		json_text = file.read()
	json_text_cleaned = re.sub(r',\s*([}\]])', r'\1', json_text) # remove trailing commas before closing braces or brackets
	places = json.loads(json_text_cleaned)
	filtered_places = []
	for place in places:
		if not any(type in place["types"] for type in filter_types) and not any(name == place["name"] for name in filter_names):
			if abs(place["geometry"]["location"]["x"]) < args.filter_distance_square and abs(place["geometry"]["location"]["y"]) < args.filter_distance_square:
				filtered_places.append(place)

	print("# filtered places:", len(filtered_places))
	with open(f"assets/scenes/{args.scene}/raw/places_enu.json", 'w') as json_file:
		json.dump(filtered_places, json_file, indent=4)
	# if args.remove_temp:
	#     os.remove(f"assets/scenes/{args.scene}/raw/places_enu_original.json")

def save_metadata(args):
	with open(f"assets/scenes/{args.scene}/raw/places_enu.json", 'r') as file:
		json_text = file.read()
	json_text_cleaned = re.sub(r',\s*([}\]])', r'\1', json_text) # remove trailing commas before closing braces or brackets
	places = json.loads(json_text_cleaned)
	if os.path.exists("env_places_metadata.json"):
		with open("env_places_metadata.json", 'r') as file:
			metadata = json.load(file)
	else:
		metadata = {}
	# with open("tools/scene/place_type_annotations.json", 'r') as file:
	#     place_type_annotations = json.load(file)
	# metadata = {}
	for place in places:
		metadata[place["name"]] = {}
		metadata[place["name"]]["location"] = [place["geometry"]["location"]["x"],
											   place["geometry"]["location"]["y"],
											   place["geometry"]["location"]["z"]]
		if "types" in place.keys():
			metadata[place["name"]]["types"] = place["types"]
		# if args.scene in place_type_annotations["living"]:
		#     if place["name"] in place_type_annotations["living"][args.scene]:
		#         if "types" not in place.keys():
		#             metadata[place["name"]]["types"] = []
		#         metadata[place["name"]]["types"].append(place_type_annotations["living"][args.scene][place["name"]])
		if "rating" in place.keys():
			metadata[place["name"]]["rating"] = place["rating"]
		if "user_ratings_total" in place.keys():
			metadata[place["name"]]["user_ratings_total"] = place["user_ratings_total"]
		if "vicinity" in place.keys():
			metadata[place["name"]]["vicinity"] = place["vicinity"]
	with open(f"assets/scenes/{args.scene}/raw/places_full.json", 'w') as json_file:
		json.dump(metadata, json_file, indent=4)
	if args.remove_temp:
		os.remove(f"assets/scenes/{args.scene}/raw/places_enu.json")

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

def overlay_locations_desp_on_image():
	annotated_image = Image.open(image_path).convert("RGB")
	draw = ImageDraw.Draw(annotated_image)
	coarse_types = list(google_map_coarse_to_types.keys())
	type_to_color = dict(zip(coarse_types, generate_diverse_colors(len(coarse_types))))
	coarse_types.append("transit")
	type_to_color["transit"] = (0, 0, 0)
	coarse_type_to_letter = {}
	for i, ctype in enumerate(sorted(coarse_types)):
		coarse_type_to_letter[ctype] = chr(ord('A') + i)
	places_by_coarse_type = defaultdict(list)
	place_indicators = {}
	for place_name in place_metadata:
		place = place_metadata[place_name]
		ctype = place["coarse_type"]
		places_by_coarse_type[ctype].append(place_name)
	for ctype in sorted(places_by_coarse_type.keys()):
		letter = coarse_type_to_letter[ctype]
		for i, place in enumerate(places_by_coarse_type[ctype]):
			indicator = f"{letter}{i+1}"
			for building in building_metadata:
				place_indicators[(building, place)] = indicator
	for building in building_metadata:
		for i, place in enumerate(building_metadata[building]["places"]):
			place_name = place["name"]
			x, y = place["location"][:2]
			# if building == "open space":
			#     x, y = place["location"][:2]
			# else:
			#     # if not args.verbose and i > 3: break
			#     x, y = building_metadata[building]["bounding_box"][0][:2]
			pixel_x, pixel_y = project_3d_to_2d_from_perspective_camera(np.array([x, y, get_height_at(height_field, x, y)]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"]))
			indicator_text = place_indicators[(building, place_name)]
			ctype = place["coarse_type"]
			color = type_to_color[ctype]
			draw.text((pixel_x, pixel_y + i * 14), indicator_text, font=ImageFont.truetype("assets/arial.ttf", 20), fill=color)
	world_filter_distance = args.filter_distance_square
	corners_2d_pixel = []
	for corners_2d_world in [[-world_filter_distance, world_filter_distance], [world_filter_distance, world_filter_distance], [world_filter_distance, -world_filter_distance], [-world_filter_distance, -world_filter_distance], [-world_filter_distance, world_filter_distance]]:
		x, y = corners_2d_world
		corners_2d_pixel.append(project_3d_to_2d_from_perspective_camera(np.array([x, y, get_height_at(height_field, x, y)]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"])))

	draw.line(corners_2d_pixel, fill="red", width=3)
	legend_image = generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter)
	annotated_image = stitch_images_horizontally(annotated_image, legend_image)
	annotated_image.save(f"assets/scenes/{args.scene}/global_annotated.png")

def generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter):
	sorted_ctypes = sorted(places_by_coarse_type.keys())
	col_height = 26
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, col_height))
	ax_left, ax_right = axes[0], axes[1]
	current_y_left = 0
	current_y_right = 0

	def draw_entry(ax, start_y, pname, i):
		indicator = f"{letter}{i+1}"
		color = np.array(type_to_color[ctype]) / 255.0
		ax.add_patch(plt.Rectangle((0, start_y), 0.15, 0.15, color=color))
		ax.text(0.35, start_y + 0.1, f"{pname} ({indicator})", va='center', ha='left', fontsize=12)
		return start_y + 0.3

	draw_on_left = True
	for ctype in sorted_ctypes:
		increase_y_for_ctype = True
		letter = coarse_type_to_letter[ctype]
		if draw_on_left:
			ax_left.text(0.0, current_y_left, f"{letter} ({ctype})", fontsize=14, fontweight='bold')
			current_y_left += 0.3
		else:
			ax_right.text(0.0, current_y_right, f"{letter} ({ctype})", fontsize=14, fontweight='bold')
			current_y_right += 0.3
		for i, pname in enumerate(places_by_coarse_type[ctype]):
			if draw_on_left:
				current_y_left = draw_entry(ax_left, current_y_left, pname, i)
			else:
				current_y_right = draw_entry(ax_right, current_y_right, pname, i)
				increase_y_for_ctype = True
			if draw_on_left and ((col_height - current_y_left) < 1.5):
				draw_on_left = False
				increase_y_for_ctype = False
		if draw_on_left:
			current_y_left += 0.6
		else:
			if increase_y_for_ctype:
				current_y_right += 0.6

	ax_left.set_ylim(0, col_height)
	ax_right.set_ylim(0, col_height)
	ax_left.set_xlim(0, 6)
	ax_right.set_xlim(0, 6)
	ax_left.invert_yaxis()
	ax_right.invert_yaxis()
	ax_left.axis('off')
	ax_right.axis('off')

	buf = io.BytesIO()
	plt.savefig(buf, format='png', bbox_inches='tight')
	plt.close(fig)
	buf.seek(0)
	return Image.open(buf)

# def bbox_vis_old(buildings, title):
#     bbox_list = []
#     for building in buildings:
#         if "bounding_box" in buildings[building]:
#             bbox_list.append(buildings[building]["bounding_box"])
#         elif "custom:bounding_box" in buildings[building]:
#             bbox_list.append(buildings[building]["custom:bounding_box"])
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"The image at path {image_path} could not be loaded.")

#     height, width, _ = image.shape
#     orthographic_scale = int(image_path.split('_')[-1].split('.')[0])
#     meter_to_pixel = width / orthographic_scale

#     fig, ax = plt.subplots()
#     ax.imshow(image)

#     for building_to_osm_tags_3d in bbox_list:
#         if building_to_osm_tags_3d is not None:
#             x_coords = [point[0] for point in building_to_osm_tags_3d]
#             y_coords = [point[1] for point in building_to_osm_tags_3d]
#             min_x, max_x = width / 2 + min(x_coords) * meter_to_pixel, width / 2 + max(x_coords) * meter_to_pixel
#             min_y, max_y = height / 2 - min(y_coords) * meter_to_pixel, height / 2 - max(y_coords) * meter_to_pixel

#             rect = patches.Rectangle(
#                 (min_x, min_y),
#                 max_x - min_x,
#                 max_y - min_y,
#                 linewidth=2,
#                 edgecolor='red',
#                 facecolor='green',
#                 alpha=0.5
#             )

#             ax.add_patch(rect)

#     ax.set_xlim(0, width)
#     ax.set_ylim(height, 0)
#     plt.axis('off')
#     plt.title(title)
#     plt.show()

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

def is_point_in_bounding_box(x, y, bounding_box_3d):
	x_coords = [point[0] for point in bounding_box_3d]
	y_coords = [point[1] for point in bounding_box_3d]
	min_x, max_x = min(x_coords), max(x_coords)
	min_y, max_y = min(y_coords), max(y_coords)
	return min_x <= x <= max_x and min_y <= y <= max_y

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

def accessibility_filtering_old2(buildings, building_to_osm_tags):
	bbox_list = []
	for building in building_to_osm_tags:
		if "custom:bounding_box" in building_to_osm_tags[building] and building_to_osm_tags[building]["custom:bounding_box"] is not None:
			bbox_list.append(building_to_osm_tags[building]["custom:bounding_box"])
	accessible_buildings = dict(buildings)
	inaccessible_buildings = {}
	for name, building in tqdm.tqdm(buildings.items()):
		if name != "open space":
			sampled_point = sample_location_on_extended_bounding_box(building['bounding_box'], bbox_list)
			if sampled_point is None:
				inaccessible_buildings[name] = buildings[name]
				del accessible_buildings[name]
			else:
				if any(abs(coords[0]) > args.filter_distance_square or abs(coords[1]) > args.filter_distance_square for coords in building['bounding_box']):
					inaccessible_buildings[name] = buildings[name]
					del accessible_buildings[name]
				else:
					accessible_buildings[name]["outdoor_xy"] = sampled_point
	return accessible_buildings, inaccessible_buildings

def accessibility_filtering(buildings):
	accessible_buildings = dict(buildings)
	inaccessible_buildings = {}
	for name, building in tqdm.tqdm(buildings.items()):
		if name != "open space":
			sampled_point = sample_location_on_extended_bounding_box_flood_fill(obstacle_grid, building['bounding_box'], obstacle_grid_parameters["resolution"], obstacle_grid_parameters["min_x"], obstacle_grid_parameters["min_y"], obstacle_grid_parameters["nx"], obstacle_grid_parameters["ny"], max_retry_times=25)
			if sampled_point is None:
				inaccessible_buildings[name] = buildings[name]
				del accessible_buildings[name]
			else:
				if any(abs(coords[0]) > args.filter_distance_square or abs(coords[1]) > args.filter_distance_square for coords in bbox_center_to_corners_repr(building['bounding_box'])):
					inaccessible_buildings[name] = buildings[name]
					del accessible_buildings[name]
				else:
					accessible_buildings[name]["outdoor_xy"] = sampled_point
	return accessible_buildings, inaccessible_buildings

def probabilistic_place_filtering(places_dict_temp, max_places=128):
	coarse_type_counts = defaultdict(list)
	for place_name, place_info in places_dict_temp.items():
		coarse_type = place_info["coarse_type"]
		coarse_type_counts[coarse_type].append(place_name)
	sorted_coarse_types = sorted(coarse_type_counts.items(), key=lambda item: len(item[1]), reverse=True)
	total_places = len(places_dict_temp.keys())
	places_to_drop = total_places - max_places
	if places_to_drop > 0:
		drop_probabilities = {}
		for coarse_type, places in sorted_coarse_types:
			drop_probabilities[coarse_type] = len(places) / total_places
		drop_probabilities = {k: v ** 1.2 for k, v in drop_probabilities.items()} # exponential scaling
		total_probability = sum(drop_probabilities.values())
		drop_probabilities = {k: v / total_probability for k, v in drop_probabilities.items()} # normalize the probabilities
		places_dropped = 0
		while places_dropped < places_to_drop:
			for coarse_type, places in sorted_coarse_types:
				if len(coarse_type_counts[coarse_type]) <= 10:
					continue
				if places_dropped >= places_to_drop:
					break
				if random.random() < drop_probabilities[coarse_type] and places:
					# print("Dropped", places[-1], "of type", coarse_type)
					place_to_drop = places.pop()
					del places_dict_temp[place_to_drop]
					places_dropped += 1
	# print(f"Probabilistically dropped places to limit to {max_places}, total number of places now:", len(places_dict_temp.keys()))
	return places_dict_temp

def remove_extra_places_from_buildings(buildings_temp, max_places=5):
	# Step 1: remove duplicate-type places in each building
	for building in buildings_temp:
		place_types = set()
		unique_places = []
		for place in buildings_temp[building]["places"]:
			if place["coarse_type"] not in place_types:
				place_types.add(place["coarse_type"])
				unique_places.append(place)
		buildings_temp[building]["places"] = unique_places

	# Step 2: remove places that has the same name with the scene name
	for building in buildings_temp:
		valid_places = []
		for place in buildings_temp[building]["places"]:
			if place["name"].lower() != args.scene.lower():
				valid_places.append(place)
		buildings_temp[building]["places"] = valid_places

	# Step 3: remove extra places in each building
	for building in buildings_temp:
		if len(buildings_temp[building]["places"]) <= max_places:
			continue
		while len(buildings_temp[building]["places"]) > max_places:
			buildings_temp[building]["places"].pop(random.randint(0, len(buildings_temp[building]["places"]) - 1))
	return buildings_temp

def remove_none_bbox_entries(buildings): # deprecated
	buildings_copy = dict(buildings)
	for name in buildings:
		if buildings_copy[name]["bounding_box"] == None:
			del buildings_copy[name]
			if args.verbose:
				print(f"Building '{name}' deleted.")
	return buildings_copy

# def choose_6_stores(buildings_temp):
# 	# only choose 6 stores that correspond to different indoor scenes and are evenly distributed (currently random choice)
# 	store_places = []
# 	for building in buildings_temp:
# 		for place in buildings_temp[building]["places"]:
# 			if place["coarse_type"] == "stores":
# 				store_places.append((building, place))
# 	selected_stores = []
# 	selected_buildings = set()
# 	while len(selected_stores) < 6 and store_places:
# 		building, place = random.choice(store_places)
# 		if building not in selected_buildings:
# 			selected_stores.append((building, place))
# 			selected_buildings.add(building)
# 		store_places.remove((building, place))
# 	scene_index = 0
# 	for building in buildings_temp:
# 		chosen_places = []
# 		for place in buildings_temp[building]["places"]:
# 			if place["coarse_type"] != "stores":
# 				chosen_places.append(place)
# 			else:
# 				if place["name"] in [store[1]["name"] for store in selected_stores]:
# 					updated_place = copy.deepcopy(place)
# 					updated_place["scene"] = coarse_indoor_scene["stores"][scene_index]
# 					chosen_places.append(updated_place)
# 					scene_index += 1
# 		buildings_temp[building]["places"] = chosen_places
# 	return buildings_temp

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
		if not any(type_ in [t for coarse in google_map_coarse_to_types if coarse != "open" for t in google_map_coarse_to_types[coarse]] for type_ in places_dict[place_name]["types"]):
			if "park" in places_dict[place_name]["types"]:
				# open spaces
				loc = places_dict[place_name]["location"]
				loc[2] = 0.0
				building_to_places["open space"].append({"name": place_name, "coarse_type": "open", "fine_types": places_dict[place_name]["types"], "location": loc})
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
			if "real_estate_agency" in place_types and "llc" in place_name.lower():
				# print(f"Skipped {place_name}.")
				continue
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
		if '/' in building_name:
			print(f"Warning: Building name '{building_name}' contains '/', which may cause issues with file paths. Replacing with '_'")
			building_name = building_name.replace('/', '_')
		for i, place in enumerate(places):
			place_metadata[place["name"]] = {
				"building": real_name,
				"coarse_type": place["coarse_type"],
				"fine_types": place["fine_types"],
				"location": [building_center[0], building_center[1], building_center[2] - (i + 1) * 4] if building_center is not None else place["location"],
				# "scene": random.sample(coarse_indoor_scene[place['coarse_type']], 1)[0] if building_center is not None else None,
			}
		import genesis as gs
		if building_name != "open space":
			mesh = trimesh.load(os.path.join(get_assets_dir(), scene_assets_dir, 'buildings', "buildings_" + building_name + ".glb"))
			rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1, 0, 0])
			mesh.apply_transform(rotation_matrix)
			obb = mesh.bounding_box_oriented
			this_bounding_box = obb.vertices
			points_2d = np.unique(this_bounding_box[:, :2], axis=0)
			if points_2d.shape[0] != 4:
				this_bounding_box = irregular_to_regular_bbox(this_bounding_box)
			this_bounding_box = bbox_corners_to_center_repr(this_bounding_box)
		else:
			this_bounding_box = None
		
		if building_center is not None:
			# There's possibility that two buildings have the same name, the later one will overwrite the former one
			building_metadata[real_name] = {
				"building_glb": "buildings_" + building_name + ".glb",
				"bounding_box": this_bounding_box,
				"places": [
					{
						"name": place["name"],
						"coarse_type": place["coarse_type"],
						"fine_types": place["fine_types"],
						"location": place_metadata[place["name"]]["location"],
						# "scene": random.sample(coarse_indoor_scene[place['coarse_type']], 1)[0],
					} for place in places
				]
			}
	print("Starts filtering buildings...")
	accessible_buildings, inaccessible_buildings = accessibility_filtering(building_metadata)
	print("Finished filtering buildings.")
	json.dump(inaccessible_buildings, open(f"assets/scenes/{args.scene}/raw/inaccessible_buildings.json", 'w'), indent=4)
	# print("Starts choosing stores...")
	# updated_buildings = choose_6_stores(copy.deepcopy(accessible_buildings))
	# print("Finished choosing stores.")
	print("Starts removing extra places from buildings...")
	buildings_with_capped_places = remove_extra_places_from_buildings(accessible_buildings)
	print("Finished removing extra places from buildings.")
	original_place_metadata = copy.deepcopy(place_metadata)
	place_metadata = update_place_metadata(place_metadata, buildings_with_capped_places)
	print("Starts filtering places...")
	place_metadata = probabilistic_place_filtering(place_metadata)
	print("Finished filtering places.")
	json.dump(place_metadata, open(f"assets/scenes/{args.scene}/place_metadata.json", 'w'), indent=4)
	building_metadata = update_buildings(buildings_with_capped_places, place_metadata)
	json.dump(building_metadata, open(f"assets/scenes/{args.scene}/building_metadata.json", 'w'), indent=4)
	json.dump(building_to_places, open(f"assets/scenes/{args.scene}/raw/building_to_places.json", 'w'), indent=4)
	json.dump(mismatched, open(f"assets/scenes/{args.scene}/raw/mismatched.json", 'w'), indent=4)
	# sort type_stats by key
	type_stats = dict(sorted(type_stats.items(), key=lambda item: item[0]))
	json.dump(type_stats, open(f"assets/scenes/{args.scene}/raw/type_stats.json", 'w'), indent=4)
	json.dump(generate_new_type_stats(building_metadata), open(f"assets/scenes/{args.scene}/raw/type_stats_accessible.json", 'w'))

	# if args.align_legacy:
	# 	legacy_place_path = f"assets/scenes/{args.scene}/legacy/place_metadata.json"
	# 	if os.path.exists(legacy_place_path):
	# 		legacy_place_metadata = json.load(open(legacy_place_path))
	# 	else:
	# 		print("Legacy metadata not found!")
	# 		exit()
	# 	place_metadata_copy = copy.deepcopy(place_metadata)
	# 	for place in place_metadata_copy:
	# 		if "scene" in place_metadata_copy[place] and place_metadata_copy[place]["scene"] is not None and "/store-" in place_metadata_copy[place]["scene"]:
	# 			del place_metadata[place]
	# 	for place in legacy_place_metadata:
	# 		if legacy_place_metadata[place]["coarse_type"] != "transit" and place not in place_metadata:
	# 			place_metadata[place] = legacy_place_metadata[place]
	# 			place_metadata[place]["building"] = original_place_metadata[place]["building"]
	# 			place_metadata[place] = {key: place_metadata[place][key] for key in ["building", "coarse_type", "fine_types", "location", "scene"]}
	# 	building_metadata = update_buildings(accessible_buildings, place_metadata)
	# 	place_metadata = update_place_metadata(place_metadata, building_metadata)
	# 	json.dump(place_metadata, open(f"assets/scenes/{args.scene}/place_metadata.json", 'w'), indent=4)
	# 	json.dump(building_metadata, open(f"assets/scenes/{args.scene}/building_metadata.json", 'w'), indent=4)
	# 	print("Aligned to legacy metadata.")

	return place_metadata, building_metadata, inaccessible_buildings, building_to_places

def update_place_metadata(place_metadata, accessible_buildings):
	new_place_metadata = {}
	for place_name in place_metadata:
		if place_metadata[place_name]["building"] in accessible_buildings and place_name in [place["name"] for place in accessible_buildings[place_metadata[place_name]["building"]]["places"]]:
			new_place_metadata[place_name] = copy.deepcopy(place_metadata[place_name])
			# if place_metadata[place_name]["building"] != "open space":
			# 	for place in accessible_buildings[place_metadata[place_name]["building"]]["places"]:
			# 		if place["name"] == place_name:
			# 			new_place_metadata[place_name]["scene"] = place["scene"]
	return new_place_metadata

def update_buildings(accessible_buildings, place_metadata):
	new_buildings = {}
	for building_name in accessible_buildings:
		new_buildings[building_name] = accessible_buildings[building_name].copy()
		new_buildings[building_name]["places"] = []
		for place in accessible_buildings[building_name]["places"]:
			if place["name"] in place_metadata:
				new_buildings[building_name]["places"].append(place)
				# new_buildings[building_name]["places"][-1]["scene"] = place_metadata[place["name"]]["scene"]
		if len(new_buildings[building_name]["places"]) == 0:
			del new_buildings[building_name]
	all_buildings_in_place_metadata = set()
	for place_name in place_metadata:
		all_buildings_in_place_metadata.add(place_metadata[place_name]["building"])
	temp_new_buildings = copy.deepcopy(new_buildings)
	for building_name in temp_new_buildings:
		if building_name not in all_buildings_in_place_metadata:
			del new_buildings[building_name]
	return new_buildings

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
	import genesis as gs
	from genesis.options import CoacdOptions
	scene.add_entity(
		material=gs.materials.Rigid(
			sdf_min_res=4,
			sdf_max_res=4,
		),
		morph=gs.morphs.Mesh(
			file=os.path.join(scene_assets_dir, 'terrain.glb'),
			euler=(90.0, 0, 0),
			fixed=True,
			collision=False,
			merge_submeshes_for_collision=False,
			group_by_material=True,
		),
	)
	buildings_dir = str(os.path.join(get_assets_dir(), scene_assets_dir, 'buildings'))
	print(buildings_dir)
	if os.path.exists(buildings_dir):
		gs.logger.info(f"Loading buildings separately")
		for building in os.listdir(buildings_dir):
			if building.endswith('.glb'):
				scene.add_entity(
					material=gs.materials.Rigid(
						sdf_min_res=4,
						sdf_max_res=4,
					),
					morph=gs.morphs.Mesh(
						file=os.path.join(scene_assets_dir, 'buildings', building),
						euler=(90.0, 0, 0),
						fixed=True,
						collision=False,
						merge_submeshes_for_collision=False,
						group_by_material=True,
						decompose_nonconvex=False,
						coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
					),
				)
	else:
		scene.add_entity(
			type= "structure",
			name= "buildings",
			material=gs.materials.Rigid(
				sdf_min_res=4,
				sdf_max_res=4,
			),
			morph=gs.morphs.Mesh(
				file=os.path.join(scene_assets_dir, 'buildings.glb'),
				euler=(90.0, 0, 0),
				fixed=True,
				collision=False,
				merge_submeshes_for_collision=False,
				group_by_material=True,
				decompose_nonconvex=False,
				coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
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
	parser.add_argument("--overwrite", "-o", action='store_true')
	parser.add_argument("--visualize_metadata", "-vis", action='store_true')
	parser.add_argument("--verbose", "-v", action='store_true')
	# parser.add_argument("--seed", type=int, default=0)
	# search original places with Google Maps API
	parser.add_argument("--search_original_places", action="store_true")
	parser.add_argument("--filter_places", action="store_true")
	parser.add_argument("--filter_distance_square", type=float)
	parser.add_argument("--search_resolution", type=float, default=135.0)
	parser.add_argument("--save_metadata", action="store_true")
	parser.add_argument("--remove_temp", action="store_true")
	# generate metadata
	parser.add_argument("--generate_metadata", action="store_true")
	# align legacy metadata
	# parser.add_argument("--align_legacy", action="store_true")
	parser.add_argument("--only_update_bbox", action="store_true")
	args = parser.parse_args()
	print("args:", args)
	# random.seed(args.seed)
	import genesis as gs
	scene_assets_dir = f"ViCo/scene/v1/{args.scene}"

	if not args.only_update_bbox:
		coarse_indoor_scene = json.load(open(os.path.join(ASSETS_PATH, "coarse_type_to_indoor_scene.json"), 'r'))
		# Check necessary files are existed
		if os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "building_to_osm_tags.json")):
			print("Necessary file check passed: building_to_osm_tags.json")
		else:
			print(f"Necessary file not exist: {os.path.join(ASSETS_PATH, 'scenes', args.scene, 'raw', 'building_to_osm_tags.json')}")
			exit()

		if os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "center.txt")):
			print("Necessary file check passed: center.txt")
		else:
			print(f"Necessary file not exist: {os.path.join(ASSETS_PATH, 'scenes', args.scene, 'raw', 'center.txt')}")
			exit()

		# if os.path.exists(f"assets/scenes/{args.scene}/orthographic_scale_800.png"):
		#     print("Necessary file check passed: orthographic_scale_800.png")
		# else:
		#     print(f "Necessary file not exist: assets/scenes/{args.scene}/orthographic_scale_800.png")
		#     exit()

		# Also check height field, despite not used for annotating the scene (used in character generation)
		height_field_path=os.path.join(get_assets_dir(), f"ViCo/scene/v1/{args.scene}/height_field.npz")

		if os.path.exists(height_field_path):
			print("Necessary file check passed: height_field.npz")
			height_field = np.load(height_field_path)
			if np.all(height_field["terrain_alt"] > 0):
				print("Height field all greater than 0. Passed.")
			else:
				print(f"Height field has values smaller or equal to 0. Please double check {height_field_path}.")
				exit()
		else:
			print(f"Necessary file not exist: {height_field_path}")
			exit()

		# Load height field as LinearNDInterpolatorExt
		height_field = load_height_field(height_field_path)

		if not os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "global.png")):
			print(f"start loading scenes and take a global image of the scene from the perspective camera in Genesis")
			if not gs._initialized:
				gs.init(seed=0, precision="32", logging_level="info", backend=gs.gpu)

			gs_scene = gs.Scene(
				viewer_options=gs.options.ViewerOptions(
					res=(2000, 2000),
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
					lights=[
						{
							'type': 'directional',
							'dir': (0, -1, -1),
							'color': (1.0, 1.0, 1.0),
							'intensity': 7.0,
						},
						{
							'type': 'directional',
							'dir': (0, 1, -1),
							'color': (1.0, 1.0, 1.0),
							'intensity': 7.0,
						},

					]  # DAY
					#   'intensity' : 2.5},] # NIGHT
				),
				show_viewer=False,
				show_FPS=False,
			)

			load_city_scene(gs_scene, scene_assets_dir)

			global_cam = gs_scene.add_camera(
				res=(2000, 2000),
				pos=(0.0, 0.0, 1000.0),
				lookat=(0, 0.0, 0.0),
				fov=60,
				GUI=False
			)

			gs_scene.build()
			gs_scene.reset()

			global_rgb, _, _, _ = global_cam.render()
			Image.fromarray(global_rgb).save(os.path.join(ASSETS_PATH, "scenes", args.scene, "global.png"))
			# Image.fromarray(global_depth).save(os.path.join(f"assets/scenes/{args.scene}/global_depth.png"))
			print("Saved global image to asset folder.")

			global_cam_parameters = {}
			global_cam_parameters["camera_res"] = global_cam.res
			global_cam_parameters["camera_fov"] = global_cam.fov
			global_cam_parameters["camera_extrinsics"] = global_cam.extrinsics.tolist()
			with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "global_cam_parameters.json"), "w") as f:
				json.dump(global_cam_parameters, f)
			print("Saved global camera parameters to asset folder")
		else:
			print("Exists: global.png, skipping...")
			global_cam_parameters = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "global_cam_parameters.json"), 'r'))

		# Search places
		if not args.search_original_places and os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "places_full.json")):
			print("Exists: places_full.json, skipping...")
			# # Back up first and only filtering
			# shutil.copy2(f"assets/scenes/{args.scene}/raw/places_full.json", f"assets/scenes/{args.scene}/raw/places_full_old.json")
			# with open(f"assets/scenes/{args.scene}/raw/places_full.json", 'r') as file:
			#     places_dict = json.load(file)
			#     filtered_places_dict = {}
			#     for place in places_dict:
			#         if abs(places_dict[place]["location"][0]) <= args.filter_distance_square and abs(places_dict[place]["location"][1]) <= args.filter_distance_square:
			#             filtered_places_dict[place] = places_dict[place]
			# with open(f"assets/scenes/{args.scene}/raw/places_full.json", 'w') as file:
			#     json.dump(filtered_places_dict, file, indent=4)
		else:
			if not os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "places_enu_original.json")) or args.search_original_places:
				print("Start searching places...")
				scene_range_meta = {}
				with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "center.txt")) as f:
					scene_range_meta["lat"], scene_range_meta["lng"] = map(float, f.read().strip().split(' '))
					scene_range_meta["rad"] = 400.0
				scene_range_meta["rad"] = scene_range_meta["rad"] * math.sqrt(2)
				with open('google_map_api.txt') as f:
					api_key = f.readline().strip()
				search_original_places(args, scene_range_meta, api_key)
				print("Finish searching places.")
			if args.filter_places:
				print("Start filtering places...")
				filter_places(args)
				print("Finish filtering places.")
			if args.save_metadata:
				print("Start saving metadata...")
				save_metadata(args)
				print("Finish saving metadata.")

		# Generate metadata
		if args.generate_metadata:
			# Generate a new json file containing bounding boxes of all loaded buildings
			all_loaded_building_bboxes = []
			buildings_dir = os.path.join(get_assets_dir(), scene_assets_dir, 'buildings')
			if os.path.exists(buildings_dir):
				for building in os.listdir(buildings_dir):
					if building.endswith('.glb'):
						mesh = trimesh.load(os.path.join(buildings_dir, building))
						rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1, 0, 0])
						mesh.apply_transform(rotation_matrix)
						obb = mesh.bounding_box_oriented
						this_bounding_box = obb.vertices
						points_2d = np.unique(this_bounding_box[:, :2], axis=0)
						if points_2d.shape[0] != 4:
							this_bounding_box = irregular_to_regular_bbox(this_bounding_box)
						this_bounding_box = bbox_corners_to_center_repr(this_bounding_box)
						all_loaded_building_bboxes.append(this_bounding_box)
			json.dump(all_loaded_building_bboxes, open(os.path.join(ASSETS_PATH, "scenes", args.scene, "all_loaded_building_bboxes.json"), 'w'), separators=(",", ":"))
			print("Generated bounding boxes of all loaded buildings.")

			# Compute the obstacle grid and its parameters
			if not os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "obstacle_grid.pkl")):
				obstacle_grid_parameters = {
					"bbox_extension": 1.0,
					"resolution": 0.5,
					"padding": 1.0
				}
				all_processed_bboxes = [get_bbox(bbox, None) for bbox in all_loaded_building_bboxes]
				all_processed_bboxes = [compute_extended_polygon(bbox, extension=obstacle_grid_parameters["bbox_extension"]) for bbox in all_processed_bboxes if bbox is not None]
				print("Totol number of polygons for generating the obstacle grid:", len(all_processed_bboxes))
				xs = [x for poly in all_processed_bboxes for x, _ in poly]
				ys = [y for poly in all_processed_bboxes for _, y in poly]
				min_x, max_x = min(xs) - obstacle_grid_parameters["padding"], max(xs) + obstacle_grid_parameters["padding"]
				min_y, max_y = min(ys) - obstacle_grid_parameters["padding"], max(ys) + obstacle_grid_parameters["padding"]
				nx = int((max_x - min_x) / obstacle_grid_parameters["resolution"]) + 1
				ny = int((max_y - min_y) / obstacle_grid_parameters["resolution"]) + 1
				obstacle_grid_parameters["min_x"] = min_x
				obstacle_grid_parameters["max_x"] = max_x
				obstacle_grid_parameters["min_y"] = min_y
				obstacle_grid_parameters["max_y"] = max_y
				obstacle_grid_parameters["nx"] = nx
				obstacle_grid_parameters["ny"] = ny
				obstacle_grid = generate_obstacle_grid(all_processed_bboxes, obstacle_grid_parameters["resolution"], min_x, min_y, nx, ny)
				# save the obstacle grid (numpy array)
				obstacle_grid_save = {
					"grid": obstacle_grid,
					"parameters": obstacle_grid_parameters
				}
				pickle.dump(obstacle_grid_save, open(os.path.join(ASSETS_PATH, "scenes", args.scene, "obstacle_grid.pkl"), 'wb'))
			else:
				print("Obstacle grid already exists, skipping generation.")
				obstacle_grid_save = pickle.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "obstacle_grid.pkl"), 'rb'))
				obstacle_grid = obstacle_grid_save["grid"]
				obstacle_grid_parameters = obstacle_grid_save["parameters"]
				print("Loaded obstacle grid.")
		
			image_path = os.path.join(ASSETS_PATH, "scenes", args.scene, "global.png")
			if not os.path.exists(image_path):
				white_global_image = Image.new("RGB", (2000, 2000), "white")
				white_global_image.save(image_path)
			with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "places_full.json"), 'r') as file:
				places_dict = json.load(file)
			with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "building_to_osm_tags.json"), 'r') as file:
				building_to_osm_tags = json.load(file)
			place_metadata, building_metadata, inaccessible_buildings, building_to_places = get_building_to_places()
			# overlay_locations_desp_on_image()
			if args.visualize_metadata:
				bbox_vis(building_to_osm_tags, "building_to_osm_tags buildings")
				bbox_vis(building_metadata, "accessible buildings")
				bbox_vis(inaccessible_buildings, "inaccessible buildings")
			print("Finished generating scene metadata.")

		# Check alignment between place metadata and building metadata
		metadata_alignment_error_flag = False
		place_metadata = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "place_metadata.json"), 'r'))
		building_metadata = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "building_metadata.json"), 'r'))
		# first check if every place in place_metadata is in building_metadata
		for place in place_metadata:
			if place_metadata[place]["building"] not in building_metadata:
				print(f"Place {place} has no matched building in building metadata!")
				metadata_alignment_error_flag = True
				continue
			places_in_building_metadata = [place["name"] for place in building_metadata[place_metadata[place]["building"]]["places"]]
			if place not in places_in_building_metadata:
				print(f"Place {place} does not exist in building metadata!")
				metadata_alignment_error_flag = True
		# then check if every building in building_metadata exists in place_metadata
		all_buildings_in_place_metadata = set()
		for place in place_metadata:
			all_buildings_in_place_metadata.add(place_metadata[place]["building"])
		for building in building_metadata:
			if building not in all_buildings_in_place_metadata:
				print(f"Building {building} has no matched place in place metadata!")
				metadata_alignment_error_flag = True
		if metadata_alignment_error_flag:
			print("Scene metadata alignment error! That means the previous code has some bugs, need to investigate!")
			exit()
		else:
			print("Scene metadata alignment success!")
	
	# if not os.path.exists(f"assets/scenes/{args.scene}/building_metadata.json"):
	# 	print(f"Building metadata not found for scene {args.scene}.")
	# 	exit()
	# building_metadata = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "building_metadata.json"), 'r'))
	# scene_assets_dir = f"ViCo/scene/v1/{args.scene}"
	# for building in building_metadata:
	# 	# If encounter 'ValueError: string is not a file' error, change the '/' in the glb name after 'buildings_' to '_' should generally fix the problem.
	# 	if "building_glb" in building_metadata[building]:
	# 		building_glb_after_buildings_ = building_metadata[building]["building_glb"].split('buildings_')[1]
	# 		if '/' in building_glb_after_buildings_:
	# 			building_metadata[building]["building_glb"] = building_metadata[building]["building_glb"].split('buildings_')[0] + "buildings_" + building_glb_after_buildings_.replace('/', '_')
	# 	if "bounding_box" in building_metadata[building] and building_metadata[building]["bounding_box"] is not None:
	# 		mesh = trimesh.load(os.path.join(get_assets_dir(), scene_assets_dir, 'buildings', building_metadata[building]["building_glb"]))
	# 		rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1, 0, 0])
	# 		mesh.apply_transform(rotation_matrix)
	# 		obb = mesh.bounding_box_oriented
	# 		building_metadata[building]["bounding_box"] = obb.vertices.tolist()
	# json.dump(building_metadata, open(os.path.join(ASSETS_PATH, "scenes", args.scene, "building_metadata.json"), 'w'), indent=4)
	# print("Updated bounding boxes in building metadata.")