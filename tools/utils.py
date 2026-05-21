from typing import Union
import os
import re
import cv2
import random
import colorsys
import numpy as np
import torch
import matplotlib.colors as mcolors
from rtree import index as rtree_index
from datetime import datetime, timedelta
import scipy
from PIL import Image
import json
import tqdm
import traceback
from collections import defaultdict
import math
from multiprocessing import Lock

def get_assets_dir():
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

_saving_lock = Lock()
_saving_map_lock = {}
def atomic_save(file_path: str, data: Union[str, bytes, Image.Image]):
	assert type(file_path) is str, "file_path should be a string"
	_saving_lock.acquire()
	if file_path not in _saving_map_lock:
		_saving_map_lock[file_path] = Lock()
	lock = _saving_map_lock[file_path]
	_saving_lock.release()
	lock.acquire()
	# atomic save
	try:
		if not os.path.exists(file_path):
			if isinstance(data, Image.Image):
				data.save(file_path)
			else:
				mode = "w" if isinstance(data, str) else "wb"
				with open(file_path, mode) as f:
					f.write(data)
		else:
			base, ext = os.path.splitext(file_path)
			tmp_file_path = base + ".tmp" + ext
			if isinstance(data, Image.Image):
				data.save(tmp_file_path)
			else:
				mode = "w" if isinstance(data, str) else "wb"
				with open(tmp_file_path, mode) as f:
					f.write(data)
			os.replace(tmp_file_path, file_path)
	except Exception as e:
		print(f"Warning: Failed to save {file_path} with error: {e}, traceback: {traceback.format_exc()}")
	finally:
		lock.release()

def gs_quat2euler(quat):  # xyz
	# Extract quaternion components
	qw, qx, qy, qz = quat.unbind(-1)

	# Roll (x-axis rotation)
	sinr_cosp = 2 * (qw * qx + qy * qz)
	cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
	roll = torch.atan2(sinr_cosp, cosr_cosp)

	# Pitch (y-axis rotation)
	sinp = 2 * (qw * qy - qz * qx)
	pitch = torch.where(
		torch.abs(sinp) >= 1,
		torch.sign(sinp) * torch.tensor(torch.pi / 2),
		torch.asin(sinp),
	)

	# Yaw (z-axis rotation)
	siny_cosp = 2 * (qw * qz + qx * qy)
	cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
	yaw = torch.atan2(siny_cosp, cosy_cosp)

	return torch.stack([roll, pitch, yaw], dim=-1)

def merge_step_files(steps_folder_path, agent_names, num_steps=43200, overwrite=False):
	if os.path.exists(os.path.join(steps_folder_path, "all_steps.json")) and not overwrite:
		all_steps_json = json.load(open(os.path.join(steps_folder_path, "all_steps.json"), 'r'))
		return all_steps_json
	all_steps_json = {}
	all_chat_history = {}
	conversation_cumulative_lookup = defaultdict(list)
	step_storage = defaultdict(list)
	for name in tqdm.tqdm(agent_names):
		agent_folder_path = os.path.join(steps_folder_path, name)
		agent_steps_json = []
		agent_chat_history = [[]]
		for i in tqdm.tqdm(range(num_steps - 1)): # some agent's steps file may not have the last step.
			step_json = json.load(open(os.path.join(agent_folder_path, f'{i:06d}.json')))
			step_storage[name].append([step_json['curr_time'], step_json["action"]])
			obs = {}
			if "current_place" in step_json["obs"]:
				obs["current_place"] = step_json["obs"]["current_place"]
			if "pose" in step_json["obs"]:
				obs["pose"] = step_json["obs"]["pose"]
			if "action_status" in step_json["obs"] and step_json["obs"]["action_status"] == "FAIL":
				try:
					agent_steps_json[-1]["action"] = "failed to " + agent_steps_json[-1]["action"]["type"]
				except Exception as e:
					print(f"Error: {e} with traceback: {traceback.format_exc()}")
			agent_steps_json.append({
				"obs": obs,
				"curr_time": step_json["curr_time"],
				"action": step_json["action"],
				"action_desp": step_json["action_desp"] if "action_desp" in step_json else None,
			})
			if "chatting_buffer" in step_json and step_json["chatting_buffer"]:
				# if type(step_json["chatting_with"][0]) == list:
				# 	# a fix for the bug in logging steps
				# 	step_json["chatting_buffer"] = step_json["chatting_buffer"][0]
				# 	step_json["chatting_with"] = step_json["chatting_with"][0]
				chatting_buffer = step_json["chatting_buffer"]
				if len(chatting_buffer) > len(agent_chat_history[-1]):
					agent_chat_history[-1] = chatting_buffer
				elif len(chatting_buffer) == len(agent_chat_history[-1]):
					continue
				else:
					agent_chat_history.append(chatting_buffer)
		all_steps_json[name] = agent_steps_json
		if agent_chat_history[0]:
			all_chat_history[name] = agent_chat_history
	for i in tqdm.tqdm(range(num_steps - 1)):
		has_convo = False
		for name in agent_names:
			if step_storage[name][i][1] is not None and step_storage[name][i][1]["type"] == "converse":
				has_convo = True
				break
		if has_convo:
			for j in range(i):
				for name in agent_names:
					step_time, step_action = step_storage[name][j]
					if step_action is not None and step_action["type"] == "converse":
						conversation_cumulative_lookup[i].append(f"[{step_time}] {name}: {step_action['arg1']}")
	json.dump(all_steps_json, open(os.path.join(steps_folder_path, "all_steps.json"), 'w'), separators=(',', ':'))
	json.dump(all_chat_history, open(os.path.join(steps_folder_path, "all_chat_history.json"), 'w'), separators=(',', ':'), indent=2)
	json.dump(conversation_cumulative_lookup, open(os.path.join(steps_folder_path, "conversation_cumulative_lookup.json"), 'w'), separators=(',', ':'))
	return all_steps_json

def top_highest_x_values(d, x):
	top_v = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:x])
	return top_v

def min_max_normalize_dict(d:dict, target_min:float=0, target_max:float=1):
	keys = list(d.keys())
	values = np.array(list(d.values()))

	# Perform min-max normalization
	min_val, max_val = np.min(values), np.max(values)
	range_val = max_val - min_val

	if range_val == 0:
		normalized_values = np.full_like(values, (target_max + target_min) / 2)
	else:
		normalized_values = (values - min_val) * (target_max - target_min) / range_val + target_min

	return dict(zip(keys, normalized_values.tolist()))

def json_converter(obj):
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, datetime):
		return obj.strftime("%B %d, %Y, %H:%M:%S")
	return obj.__dict__

def is_near_axis_aligned_goal(curr_x, curr_y, goal_bbox, goal_pos, threshold=2):
	# threshold is for bbox rather than Euclidean distance
	# import pdb; pdb.set_trace()
	bbox = get_axis_aligned_bbox(goal_bbox, goal_pos)
	x_coords = [point[0] for point in bbox]
	y_coords = [point[1] for point in bbox]
	min_x, max_x = min(x_coords) - threshold, max(x_coords) + threshold
	min_y, max_y = min(y_coords) - threshold, max(y_coords) + threshold
	if min_x <= curr_x <= max_x and min_y <= curr_y <= max_y:
		return True
	return False

def is_near_goal(curr_x, curr_y, goal_bbox, goal_pos, threshold=2):
	if goal_bbox is None:
		if not isinstance(goal_pos, np.ndarray):
			goal_pos = np.array(goal_pos)
		axis_aligned_bbox = np.array([goal_pos - 2, goal_pos + 2])
		x_coords = [point[0] for point in axis_aligned_bbox]
		y_coords = [point[1] for point in axis_aligned_bbox]
		min_x, max_x = min(x_coords) - threshold, max(x_coords) + threshold
		min_y, max_y = min(y_coords) - threshold, max(y_coords) + threshold
		if min_x <= curr_x <= max_x and min_y <= curr_y <= max_y:
			return True
		return False
	else:
		flag = is_point_in_extended_bounding_box(curr_x, curr_y, get_bbox(goal_bbox, None), extension=threshold)
		# print("is_point_in_extended_bounding_box:", flag)
		return flag

def get_axis_aligned_bbox(goal_bbox, goal_pos) -> np.ndarray:
	if isinstance(goal_bbox, np.ndarray):
		goal_bbox = goal_bbox.tolist()
	if goal_bbox is None:
		if not isinstance(goal_pos, np.ndarray):
			goal_pos = np.array(goal_pos)
		return np.array([goal_pos - 2, goal_pos + 2])
	x_coords = [point[0] for point in goal_bbox]
	y_coords = [point[1] for point in goal_bbox]
	min_x, max_x = min(x_coords), max(x_coords)
	min_y, max_y = min(y_coords), max(y_coords)
	return np.array([[min_x, min_y], [max_x, max_y]])

def irregular_to_regular_bbox(points):
	xy_points = points[:, :2].astype(np.float32)
	rect = cv2.minAreaRect(xy_points)
	box = cv2.boxPoints(rect)
	box = np.array(box, dtype=np.float32)
	z_min = np.min(points[:, 2])
	z_max = np.max(points[:, 2])
	bottom_face = np.hstack([box, np.full((4, 1), z_min, dtype=np.float32)])
	top_face = np.hstack([box, np.full((4, 1), z_max, dtype=np.float32)])
	bbox = np.vstack([top_face, bottom_face])
	return bbox

def bbox_corners_to_center_repr(corners: np.ndarray):
	assert corners.shape == (8, 3), "Corners representation should be of shape (8, 3)"
	center = np.mean(corners, axis=0)
	z_min = np.min(corners[:, 2])
	z_max = np.max(corners[:, 2])
	dz = z_max - z_min
	points_2d = np.unique(corners[:, :2], axis=0)
	if points_2d.shape[0] != 4:
		print(corners)
		print(points_2d.shape)
		raise ValueError("Irregular bounding box detected.")
	centroid_2d = np.mean(points_2d, axis=0)
	angles = np.arctan2(points_2d[:, 1] - centroid_2d[1], points_2d[:, 0] - centroid_2d[0])
	sort_order = np.argsort(angles)
	ordered_points = points_2d[sort_order]
	p0 = ordered_points[0]
	p1 = ordered_points[1]
	dx = np.linalg.norm(p1 - p0)
	p2 = ordered_points[2]
	dy = np.linalg.norm(p2 - p1)
	euler = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
	return [float(center[0]), float(center[1]), float(center[2]), float(dx), float(dy), float(dz), float(euler)]

def bbox_center_to_corners_repr(center_repr: list):
	assert len(center_repr) == 7, "Center representation should be of length 7"
	cx, cy, cz, dx, dy, dz, euler = center_repr
	hx, hy, hz = dx / 2, dy / 2, dz / 2
	local_corners = np.array([
		[ hx,  hy,  hz],
		[ hx, -hy,  hz],
		[-hx, -hy,  hz],
		[-hx,  hy,  hz],
		[ hx,  hy, -hz],
		[ hx, -hy, -hz],
		[-hx, -hy, -hz],
		[-hx,  hy, -hz]
	])
	cos_angle = np.cos(euler)
	sin_angle = np.sin(euler)
	R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
	rotated_xy = (R @ local_corners[:, :2].T).T
	rotated_corners = local_corners.copy()
	rotated_corners[:, :2] = rotated_xy
	corners = rotated_corners + np.array([cx, cy, cz])
	return corners

def get_bbox(goal_bbox, goal_pos) -> np.ndarray:
	if isinstance(goal_bbox, np.ndarray):
		goal_bbox = goal_bbox.tolist()
	if goal_bbox is None:
		if not isinstance(goal_pos, np.ndarray):
			goal_pos = np.array(goal_pos)
		x_min, y_min = goal_pos - 2
		x_max, y_max = goal_pos + 2
		return np.array([
			[x_min, y_min],
			[x_min, y_max],
			[x_max, y_min],
			[x_max, y_max]
		])
	assert len(goal_bbox) == 7, "The goal_bbox must be in center representation [cx,cy,cz,dx,dy,dz,euler]"
	goal_bbox = bbox_center_to_corners_repr(goal_bbox)
	bottom_surface = [[bbox[0], bbox[1]] for bbox in goal_bbox[-4:]]
	return np.array(bottom_surface)

def distance_to_bbox(pos, bbox):
	x_coords = [point[0] for point in bbox]
	y_coords = [point[1] for point in bbox]
	min_x, max_x = min(x_coords), max(x_coords)
	min_y, max_y = min(y_coords), max(y_coords)
	if pos[0] < min_x:
		dx = min_x - pos[0]
	elif pos[0] > max_x:
		dx = pos[0] - max_x
	else:
		dx = 0
	if pos[1] < min_y:
		dy = min_y - pos[1]
	elif pos[1] > max_y:
		dy = pos[1] - max_y
	else:
		dy = 0
	return dx + dy

def round_numericals(obj, precision=2):
	if isinstance(obj, dict):
		for key in obj:
			obj[key] = round_numericals(obj[key], precision)
	elif isinstance(obj, list):
		for i in range(len(obj)):
			obj[i] = round_numericals(obj[i], precision)
	elif isinstance(obj, float):
		obj = round(obj, precision)
	return obj


def motion_schedule_processor(json_data):
	
	valid_motion_types = {"navigate_to", "sleep", "wake", "pick", "put", "stand", 
						  "sit", "drink", "eat", "wait", "look_at"}
	for entry in json_data:
		if not all(key in entry for key in ["description", "time", "motion_list"]):
			return "Missing required keys in an activity entry."

		try:
			datetime.strptime(entry["time"], "%H:%M:%S")
		except ValueError:
			return f"Invalid time format: {entry['time']}"
		
		for motion in entry["motion_list"]:
			if "type" not in motion or motion["type"] not in valid_motion_types:
				return f"Invalid motion type: {motion.get('type', 'UNKNOWN')}"
			
			if motion["type"] == "navigate_to":
				if "arg1" not in motion or not isinstance(motion["arg1"], str):
					return "Invalid 'navigate_to' arguments. 'arg1' should be a location name (string)."
			
			elif motion["type"] == "pick":
				if "arg1" not in motion or not isinstance(motion["arg1"], int):
					return "Invalid hand ID in 'pick' action. 'arg1' should be an integer."
				if "arg2" not in motion or not isinstance(motion["arg2"], str):
					return "Invalid 'pick' arguments. 'arg2' should be an object name (string)."
			
			elif motion["type"] == "put":
				if "arg1" not in motion or not isinstance(motion["arg1"], int):
					return "Invalid hand ID in 'put' action. 'arg1' should be an integer."
				if "arg2" in motion and not isinstance(motion["arg2"], str):
					continue

			elif motion["type"] in ["sit", "look_at"]:
				if "arg1" not in motion or not isinstance(motion["arg1"], str):
					return "Invalid 'sit' arguments. 'arg1' should be a location name (string)."
			
			elif motion["type"] in {"drink", "eat"}:
				if "arg1" not in motion or not isinstance(motion["arg1"], int):
					return f"Invalid hand ID in '{motion['type']}' action. 'arg1' should be an integer."
				
	return ""

def schedule_validator(hourly_schedule, s_mem, curr_time, place_metadata=None, logger=None):
	error_messages = []
	all_places = s_mem.get_places()
	for schedule_step in hourly_schedule:
		place = schedule_step["place"]
		if place is not None and place not in all_places:
			for correct_place in all_places:
				if place.lower() == correct_place.lower():
					schedule_step["place"] = correct_place
					break
	first_schedule = True
	for i, schedule_step in enumerate(hourly_schedule):
		if datetime.strptime(schedule_step["end_time"], "%H:%M:%S").time() < curr_time.time():
			continue
		if i == 0:
			if schedule_step['start_time'] != "00:00:00":
				logger.warning(f"Warning: first activity start time is not 00:00:00. Fixed it.")
				schedule_step['start_time'] = "00:00:00"
				# error_messages.append(f"Error: first activity start time is not 00:00:00.")
		else:
			if schedule_step['start_time'] != hourly_schedule[i-1]['end_time']:
				if schedule_step['start_time'] < hourly_schedule[i-1]['end_time']:
					error_messages.append(f"Error: activity {i} start time is less than previous activity end time.")
				else:
					# try to fix it
					if schedule_step['type'] == "commute":
						schedule_step['start_time'] = hourly_schedule[i-1]['end_time']
					else:
						if hourly_schedule[i-1]['type'] == "commute":
							lst_place = hourly_schedule[i-1]['end_place']
						else:
							lst_place = hourly_schedule[i-1]['place']
						if schedule_step['place'] == lst_place:
							schedule_step['start_time'] = hourly_schedule[i-1]['end_time']
						else:
							new_schedule_step = {
								"type": "commute",
								"activity": "Commute to " + schedule_step['place'],
								"place": None,
								"building": None,
								"start_time": hourly_schedule[i-1]['end_time'],
								"end_time": schedule_step['start_time'],
								"start_place": hourly_schedule[i-1]['place'],
								"end_place": schedule_step['place']
							}
							hourly_schedule.insert(i, new_schedule_step)
					if logger is not None:
						logger.warning(f"Warning: activity {i} start time is not equal to previous activity end time. Fixed it.")


		if i == len(hourly_schedule) - 1:
			if schedule_step['end_time'] != "23:59:59":
				if logger is not None:
					logger.warning(f"Warning: last activity end time is not 23:59:59. Fixed it.")
				schedule_step['end_time'] = "23:59:59"
				# error_messages.append(f"Error: last activity end time is not 23:59:59.")

		if schedule_step["type"] == "commute":
			if schedule_step["place"] is not None:
				error_messages.append(f"Error: place is not None for commute activity.")
			elif i == 0:
				hourly_schedule[i+1]["start_time"] = schedule_step["start_time"]
				del hourly_schedule[0]
				logger.warning(f"Warning: Generated commute activity at the beginning of schedule. Fixed it.")
			elif i == len(hourly_schedule) - 1:
				hourly_schedule[i-1]["end_time"] = schedule_step["end_time"]
				del hourly_schedule[-1]
				logger.warning(f"Warning: Generated commute activity at the end of schedule. Fixed it.")
			elif hourly_schedule[i + 1]["type"] == "commute":
				error_messages.append(f"Error: commute activity is followed by another commute activity.")
			else:
				schedule_step["start_place"] = hourly_schedule[i-1]["place"]
				schedule_step["end_place"] = hourly_schedule[i+1]["place"]
		else:
			if schedule_step["place"] not in all_places:
				error_messages.append(f"Error: place '{schedule_step['place']}' not exists in spatial memory {repr(s_mem.get_places())}.")
				continue
			should_be_building = s_mem.get_building_from_place(schedule_step["place"])
			if should_be_building is None or schedule_step["building"].lower() != should_be_building.lower():
				schedule_step["building"] = should_be_building
				if logger is not None:
					logger.warning(f"Warning: building of place '{schedule_step['place']}' is not equal to building in spatial memory. Fixed it.")
				# error_messages.append(f"Error: building of place '{schedule_step['place']}' is not equal to building in spatial memory.")
				continue
			lst_building = hourly_schedule[i-1]["building"] if i > 0 else None
			if lst_building is not None and lst_building.lower() != schedule_step["building"].lower():
				# error_messages.append(f"Error: last place '{lst_building}' is not equal to activity {i}'s place '{schedule_step['building']}' and need to commute first.")
				# try to fix it
				remaining_time = (datetime.strptime(hourly_schedule[i]["end_time"], "%H:%M:%S") - datetime.strptime(hourly_schedule[i]["start_time"], "%H:%M:%S")).seconds // 60
				end_time = datetime.strptime(hourly_schedule[i-1]["end_time"], "%H:%M:%S") + timedelta(minutes=min(10, remaining_time))
				new_schedule_item = {
					"type": "commute",
					"activity": "Commute to " + schedule_step['place'],
					"place": None,
					"building": None,
					"start_time": hourly_schedule[i-1]['end_time'],
					"end_time": end_time.strftime("%H:%M:%S"),
					"start_place": hourly_schedule[i-1]['place'],
					"end_place": schedule_step['place']
				}
				hourly_schedule.insert(i, new_schedule_item)
				hourly_schedule[i + 1]["start_time"] = new_schedule_item["end_time"]

		first_schedule = False

	return error_messages

class EventSystem:
	# Implementation based on R-Tree
	def __init__(self):
		self.idx = rtree_index.Index(properties=rtree_index.Property(dimension=3))
		self.events = {}
		self.event_id_counter = 0
		self.eps = 1e-2

	def add(self, type, pos, r, content, priority=0, subject=None, predicate=None, object=None) -> list:
		# r is radius
		# content supposed to be in text form
		x, y, z = pos
		if type == "speech":
			pos[2] += 1 # to make sure speech events are above the ground
		bbox = (x - r, y - r, z - r, x + r, y + r, z + r)
		self.idx.insert(self.event_id_counter, bbox)
		self.events[self.event_id_counter] = {
			"type": type,
			"position": pos,
			"r": r,
			"content": content,
			"priority": priority,
			"subject": subject,
			"predicate": predicate,
			"object": object,
			"bbox": bbox
		}
		new_event_id = self.event_id_counter
		self.event_id_counter += 1

		if type == "speech":
			intersecting_ids = list(self.idx.intersection(bbox))
			intersecting_speech_ids = [event_id for event_id in intersecting_ids if
									   event_id != new_event_id and self.events[event_id]["type"] == "speech"]
			if intersecting_speech_ids:
				all_speech_ids = intersecting_speech_ids + [new_event_id]
				# only keep the highest priority one
				event_ids_to_delete = sorted(all_speech_ids,
										   key=lambda event_id: self.events[event_id]["priority"], reverse=True)[1:]
				return self.delete(event_ids_to_delete)
		return []

	def delete(self, event_ids:list[int]) -> list:
		deleted_subjects = []
		for event_id in event_ids:
			deleted_subjects.append(self.events[event_id]["subject"])
			# print("events:", self.events)
			# print("event_id:", event_id)
			# print("all entries before:", list(self.idx.intersection((-float('inf'), -float('inf'), -float('inf'), float('inf'), float('inf'), float('inf')))))
			self.idx.delete(event_id, self.events[event_id]["bbox"])
			# print("all entries after:", list(self.idx.intersection((-float('inf'), -float('inf'), -float('inf'), float('inf'), float('inf'), float('inf')))))
			del self.events[event_id]

		return deleted_subjects

	def get(self, ref_pos: list) -> list:
		x, y, z = ref_pos
		# print("intersection of:", (x-self.eps, y-self.eps, z-self.eps, x+self.eps, y+self.eps, z+self.eps))
		results_in_range = list(self.idx.intersection((x-self.eps, y-self.eps, z-self.eps, x+self.eps, y+self.eps, z+self.eps)))
		# print("results in range:", results_in_range)
		# print("events:", self.events)
		events_in_range = [self.events[event_id] for event_id in results_in_range]
		return events_in_range

	def update(self, event_id, content):
		self.events[event_id]["content"] = content

	def clear(self):
		# later add clear based on time of duration
		self.idx = rtree_index.Index(properties=rtree_index.Property(dimension=3))
		self.events = {}
		self.event_id_counter = 0

def generate_diverse_colors(n):
	colors = []
	for i in range(n):
		hue = i / n
		color = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
		colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))
	return colors

def map_lang_color_to_rgba(lang_color):
	rgb = mcolors.to_rgb(lang_color)
	return (*rgb, 1.0)

def map_lang_colors_to_rgb(lang_colors):
	rgb_list = []
	for lang_color in lang_colors:
		rgb_list.append(mcolors.to_rgb(lang_color))
	return rgb_list

def stitch_images_horizontally(image1, image2):
	combined_width = image1.width + image2.width
	combined_height = image1.height
	combined_image = Image.new('RGB', (combined_width, combined_height))
	combined_image.paste(image1, (0, 0))
	combined_image.paste(image2, (image1.width, 0))
	return combined_image

def opengl_projection_matrix_to_intrinsics(P: np.ndarray, width: int, height: int):
	"""Convert OpenGL projection matrix to camera intrinsics.
	Args:
		P (np.ndarray): OpenGL projection matrix.
		width (int): Image width.
		height (int): Image height
	Returns:
		np.ndarray: Camera intrinsics. [3, 3]
	"""
	fx = P[0, 0] * width / 2
	fy = P[1, 1] * height / 2
	cx = (1.0 - P[0, 2]) * width / 2
	cy = (1.0 + P[1, 2]) * height / 2
	K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
	return K

def render_topdown_locators(image_path, locator_positions, colors, circle_radii, camera=None):
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"The image at path {image_path} could not be loaded.")

	if camera is None: # use orthographic mode which doesn't use camera intrinsics and extrinsics
		height, width, _ = image.shape
		orthographic_scale = image_path.split('_')[-1].split('.')[0]
		if not orthographic_scale.isdigit():
			raise ValueError("Image file name for the render_topdown_locators() must be SCENE_orthographic_scale_POSITIVEINTEGER.png")
		meter_to_pixel = width / int(orthographic_scale)

		for pos, color, radius in zip(locator_positions, colors, circle_radii):
			pixel_x = int(width / 2 + pos[0] * meter_to_pixel)
			pixel_y = int(height / 2 - pos[1] * meter_to_pixel)
			cv2.circle(image, (pixel_x, pixel_y), radius, (color[2]*255, color[1]*255, color[0]*255), -1)
	else:
		# print("res:", camera.res[0], camera.res[1])
		f_x = camera.res[0] / (2.0 * np.tan(np.radians(camera.fov / 2.0)))
		f_y = camera.res[1] / (2.0 * np.tan(np.radians(camera.fov / 2.0)))
		intrinsic_K = np.array([[f_x, 0.0, camera.res[0]/2.0],
								[0.0, f_y, camera.res[1]/2.0],
								[0.0, 0.0, 1.0]])
		# print(intrinsic_K)
		# print(camera._rasterizer)
		# print(camera._rasterizer._camera_nodes)
		# intrinsic_K = opengl_projection_matrix_to_intrinsics(
		#     camera._rasterizer._camera_nodes.get_projection_matrix(), width=camera.res[0], height=camera.res[1]
		# )
		extrinsic = camera.extrinsics
		extrinsic = extrinsic[:3, :4]

		for pos, color, radius in zip(locator_positions, colors, circle_radii):
			P_world = np.append(pos, 1.0)
			P_camera = extrinsic @ P_world
			P_image = intrinsic_K @ P_camera
			pixel_x = int(P_image[0] / P_image[2])
			pixel_y = int(P_image[1] / P_image[2])
			# print(pixel_x, pixel_y)
			cv2.circle(image, (pixel_x, pixel_y), radius, (color[2]*255, color[1]*255, color[0]*255), -1)

	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_png_images(directory):
	images = []
	pattern = re.compile(r'rgb_(\d{4})\.png')
	sorted_files = sorted(
		[f for f in os.listdir(directory) if pattern.match(f)],
		key=lambda x: int(pattern.match(x).group(1))
	)
	for filename in sorted_files:
		img_path = os.path.join(directory, filename)
		img = Image.open(img_path)
		img_array = np.array(img)
		images.append(img_array)

	return images

def get_accessible_places(building_metadata, place_metadata, pose, current_building, current_place):
	accessible_places = []
	if current_place is not None:
		assert current_building != "open space", "The current building must be provided if the agent is not in the open space."
		accessible_places.extend([place["name"] for place in building_metadata[current_building]["places"] if place["name"] != current_place])
		accessible_places.append("open space")
		return accessible_places
	for building in building_metadata:
		if building == "open space":
			for place in building_metadata[building]["places"]:
				if place["name"] in place_metadata and is_near_goal(pose[0], pose[1], None, place["location"][:2], threshold=2):
					accessible_places.append(place["name"])
		if building_metadata[building]['bounding_box'] is None:
			continue
		if is_near_goal(pose[0], pose[1], building_metadata[building]['bounding_box'], None, threshold=2):
			accessible_places.extend([place["name"] for place in building_metadata[building]["places"] if place["name"] in place_metadata])
	return accessible_places

def autocrop(np_img):
	"""Return the numpy image without empty margins."""
	if len(np_img.shape) == 3:
		if np_img.shape[2] == 4:
			thresholded_img = np_img[:,:,3] # use the mask
		else:
			thresholded_img = np_img.max(axis=2) # black margins
	zone_x = thresholded_img.max(axis=0).nonzero()[0]
	xmin, xmax = zone_x[0], zone_x[-1]
	zone_y = thresholded_img.max(axis=1).nonzero()[0]
	ymin, ymax = zone_y[0], zone_y[-1]
	return np_img[ymin:ymax+1, xmin:xmax+1]

def project_3d_to_2d_from_perspective_camera(pos_3d, camera_res, camera_fov, camera_extrinsics):
	f_x = camera_res[0] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	f_y = camera_res[1] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	intrinsic_K = np.array([[f_x, 0.0, camera_res[0] / 2.0],
							[0.0, f_y, camera_res[1] / 2.0],
							[0.0, 0.0, 1.0]])
	extrinsic = camera_extrinsics[:3, :4]
	P_world = np.append(pos_3d, 1.0)
	P_camera = extrinsic @ P_world
	P_image = intrinsic_K @ P_camera
	pixel_x = int(P_image[0] / P_image[2])
	pixel_y = int(P_image[1] / P_image[2])
	return pixel_x, pixel_y

def compute_extended_polygon(bbox, extension):
	def order_points(points):
		center = np.mean(points, axis=0)
		angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
		return points[np.argsort(angles)]

	def line_intersection(p1, p2, p3, p4):
		x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
		denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
		if np.isclose(denom, 0):
			raise ValueError("Extended polygon cannot be computed because lines are parallel and do not intersect.")
		ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
		iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
		return np.array([ix, iy])

	def offset_polygon(polygon, offset):
		N = polygon.shape[0]
		area = 0.0
		for i in range(N):
			x1, y1 = polygon[i]
			x2, y2 = polygon[(i + 1) % N]
			area += x1 * y2 - x2 * y1
		ccw = area > 0
		offset_lines = []
		for i in range(N):
			p1 = polygon[i]
			p2 = polygon[(i + 1) % N]
			edge = p2 - p1
			if ccw:
				n = np.array([edge[1], -edge[0]])
			else:
				n = np.array([-edge[1], edge[0]])
			n = n / np.linalg.norm(n)
			offset_lines.append((p1 + offset * n, p2 + offset * n))
		new_polygon = []
		for i in range(N):
			pt = line_intersection(offset_lines[i][0], offset_lines[i][1], offset_lines[i - 1][0], offset_lines[i - 1][1])
			new_polygon.append(pt)
		return np.array(new_polygon)
	if not isinstance(bbox, np.ndarray):
		bbox = np.array(bbox)
	assert bbox.shape == (4, 2), "The bounding box must be a list of 4 vertices, each has x and y. The expected shape is (4, 2), but the current shape is " + str(bbox.shape)
	ordered_bbox = order_points(bbox)
	return offset_polygon(ordered_bbox, extension)

def random_point_on_polygon_edge(polygon, abs_max=385, max_retry_times=100):
	n = len(polygon)
	retry_times = 0
	while retry_times < max_retry_times:
		i = random.randint(0, n - 1)
		p1 = polygon[i]
		p2 = polygon[(i + 1) % n]
		t = random.random()
		x = p1[0] + t * (p2[0] - p1[0])
		y = p1[1] + t * (p2[1] - p1[1])
		if abs(x) < abs_max and abs(y) < abs_max:
			return (x, y)
		retry_times += 1
	return None

def point_in_polygon(point, polygon):
	x, y = point
	inside = False
	n = len(polygon)
	for i in range(n):
		j = (i - 1) % n
		xi, yi = polygon[i]
		xj, yj = polygon[j]
		if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
			inside = not inside
	return inside

def is_point_enclosed(grid, point, resolution, min_x, min_y, nx, ny):
    from collections import deque

    i = int((point[0] - min_x) / resolution)
    j = int((point[1] - min_y) / resolution)

    # if i < 0 or i >= nx or j < 0 or j >= ny:
    #     print("Point is out of bounds")
    #     return True, (i, j) # not valid

    if grid[i, j] == 1:
        # print("Point is inside an obstacle")
        return True, (i, j) # not valid

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

def generate_obstacle_grid(polygons, resolution, min_x, min_y, nx, ny):
    import taichi as ti
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        print("Obstacle Grid Generation: GPU unavailable or failed. Use CPU.")
        ti.init(arch=ti.cpu)

    max_polygons = 10000
    max_polygon_vertices = 4

    polygon_vertices = ti.Vector.field(2, dtype=ti.f32, shape=(max_polygons, max_polygon_vertices))
    polygon_counts = ti.field(dtype=ti.i32, shape=max_polygons)

    grid = ti.field(dtype=ti.i32, shape=(nx, ny))

    for k, poly in enumerate(polygons):
        polygon_counts[k] = len(poly)
        for l, (x, y) in enumerate(poly):
            polygon_vertices[k, l] = [x, y]

    @ti.func
    def point_in_polygon(p, poly_id):
        inside = 0
        n = polygon_counts[poly_id]
        for i in range(n):
            a = polygon_vertices[poly_id, i]
            b = polygon_vertices[poly_id, (i + 1) % n]
            if ((a.y > p.y) != (b.y > p.y)) and \
               (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-6) + a.x):
                inside = 1 - inside
        return inside

    @ti.kernel
    def generate(min_x: ti.f32, min_y: ti.f32, res: ti.f32, num_polys: ti.i32):
        for i, j in grid:
            cx = min_x + i * res
            cy = min_y + j * res
            p = ti.Vector([cx, cy])
            for k in range(num_polys):
                if point_in_polygon(p, k):
                    grid[i, j] = 1
                    break

    generate(min_x, min_y, resolution, len(polygons))
    obstacle_map = grid.to_numpy()
    return obstacle_map

def is_point_in_extended_bounding_box(x, y, bbox, extension=0.5):
	ext_polygon = compute_extended_polygon(bbox, extension)
	return point_in_polygon((x, y), ext_polygon)

def sample_location_on_extended_bounding_box(bbox, all_bboxes, extension=1, max_retry_times=100):
	retry_times = 0
	extended_polygon = compute_extended_polygon(get_bbox(bbox, None), extension)
	while retry_times < max_retry_times:
		random_point = random_point_on_polygon_edge(extended_polygon)
		if random_point is not None:
			valid = True
			for other_bbox in all_bboxes:
				if other_bbox is not None:
					if is_point_in_extended_bounding_box(random_point[0], random_point[1], get_bbox(other_bbox, None), extension=1):
						valid = False
						break
			if valid:
				return random_point
		retry_times += 1
		# print("retry times:", retry_times)
	return None

def sample_location_on_extended_bounding_box_flood_fill(grid, bbox, resolution, min_x, min_y, nx, ny, extension=1, max_retry_times=100, previous_locations=None):
	retry_times = 0
	extended_polygon = compute_extended_polygon(get_bbox(bbox, None), extension)
	while retry_times < max_retry_times:
		random_point = random_point_on_polygon_edge(extended_polygon)
		if random_point is not None:
			point_enclosed, _ = is_point_enclosed(grid, random_point, resolution, min_x, min_y, nx, ny)
			if not point_enclosed:
				if previous_locations is not None:
					too_close = False
					for loc in previous_locations:
						if np.linalg.norm(np.array(random_point) - np.array(loc)) < 2.0: # a preset threshold, can change to a variable later
							too_close = True
							break
					if too_close:
						retry_times += 1
						continue
					else:
						return random_point
				return random_point
		retry_times += 1
		# print("retry times:", retry_times)
	return None

def deproject_2d_to_3d_from_perspective_camera(pixel_coords, depth, camera_res, camera_fov, camera_extrinsics):
	pixel_x, pixel_y = pixel_coords
	f_x = camera_res[0] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	f_y = camera_res[1] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	intrinsic_K = np.array([[f_x, 0.0, camera_res[0] / 2.0],
							[0.0, f_y, camera_res[1] / 2.0],
							[0.0, 0.0, 1.0]])
	K_inv = np.linalg.inv(intrinsic_K)
	P_image = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
	P_camera = K_inv @ P_image
	P_camera = P_camera * depth
	extrinsic_inv = np.linalg.inv(camera_extrinsics)
	P_camera_h = np.array([P_camera[0], P_camera[1], P_camera[2], 1.0])
	P_world = extrinsic_inv @ P_camera_h
	return P_world[:3]

class LinearNDInterpolatorExt(object):
	def __init__(self, points, values):
		self.funcinterp = scipy.interpolate.LinearNDInterpolator(points, values)
		self.funcnearest = scipy.interpolate.NearestNDInterpolator(points, values)

	def __call__(self, *args: np.ndarray):
		t = self.funcinterp(*args)
		if not np.isnan(t).any():
			return t
		else:
			mask = np.isnan(t)
			nearest_t = self.funcnearest(*[arg[mask] for arg in args])
			t[mask] = nearest_t
			return t

	def single_interp(self, *args):
		t = self.funcinterp(*args)
		if not np.isnan(t):
			return t.item(0)
		else:
			return self.funcnearest(*args)


def load_height_field(file_name: str) -> LinearNDInterpolatorExt:
	height_field = np.load(file_name)
	xs = height_field["plane_coord"][..., 0]
	ys = height_field["plane_coord"][..., 1]
	return LinearNDInterpolatorExt(np.stack([xs, ys * -1], axis=-1), height_field["terrain_alt"])


def get_height_at(height_field: LinearNDInterpolatorExt, x: float, y: float) -> float:
	z = height_field.single_interp(x, y)
	if isinstance(z, np.ndarray):
		return z.item(0)
	return z


def plot_height_field(file_name: str):
	import matplotlib.pyplot as plt

	data = np.load(file_name)
	height_field = load_height_field(file_name)

	xs = data["plane_coord"][..., 0]
	ys = data["plane_coord"][..., 1] * -1
	z = np.hypot(xs, ys)
	X = np.linspace(-500, +500)
	Y = np.linspace(-500, +500)
	X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

	Z = height_field(X, Y)
	plt.pcolormesh(X, Y, Z, shading='auto')
	plt.colorbar()
	plt.scatter(xs, ys, s=1., label="Reference point")
	plt.legend()
	plt.axis("equal")
	plt.show()