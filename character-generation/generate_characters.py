import io
import pickle
import shutil
import os
import sys
import time
from copy import deepcopy
from collections import defaultdict
import copy
import unicodedata

import random, json, argparse, math, re, os
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from initializations import init_scratch

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

from tools.generator import Generator
from tools.utils import *

from tools.constants import google_map_coarse_to_types, ENV_OTHER_METADATA, ASSETS_PATH
from tools.annotate_scene import update_buildings


def create_folders_and_files(structure):
    for folder, files in structure.items():
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")
        else:
            print(f"Folder already exists: {folder}, skipped")

        for file in files:
            # print("file:", file)
            file_path = os.path.join(folder, file["file_name"])
            if not os.path.exists(file_path):
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(file["file_content"], f, indent=4)
                    print(f"File created: {file_path}")
                elif file_path.endswith('.pkl'):
                    with open(file_path, 'wb') as f:
                        pickle.dump(file["file_content"], f)
                    print(f"File created: {file_path}")
            else:
                print(f"File already exists: {file_path}")

def map_lang_colors_to_rgb(lang_colors):
	rgb_list = []
	for lang_color in lang_colors:
		rgb_list.append(mcolors.to_rgb(lang_color))
	return rgb_list

def generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter):
    sorted_ctypes = sorted(places_by_coarse_type.keys())
    col_height = 26
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, col_height))
    ax_left, ax_right = axes[0], axes[1]
    current_y_left = 0
    current_y_right = 0

    accommodation_indicator_mapping = {}
    iter_places = {}
    letter = coarse_type_to_letter["accommodation"]
    i = 1
    for pname in places_by_coarse_type["accommodation"]:
        if "'s room at " in pname and pname.split("'s room at ")[1] in iter_places:
            indicator = iter_places[pname.split("'s room at ")[1]]
        elif pname in iter_places:
            indicator = iter_places[pname]
        else:
            indicator = f"{letter}{i}"
            if "'s room at " in pname:
                iter_places[pname.split("'s room at ")[1]] = indicator
            else:
                iter_places[pname] = indicator
            i += 1
        accommodation_indicator_mapping[pname] = indicator
    places_by_coarse_type["accommodation"].sort(key=lambda pname: accommodation_indicator_mapping[pname])

    def draw_entry(ax, start_y, pname, indicator):
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
        i = 1
        for i, pname in enumerate(places_by_coarse_type[ctype]):
            if "'s room at " in pname and pname.split("'s room at ")[1] in iter_places:
                indicator = iter_places[pname.split("'s room at ")[1]]
            elif pname in iter_places:
                indicator = iter_places[pname]
            else:
                indicator = f"{letter}{i}"
                if "'s room at " in pname:
                    iter_places[pname.split("'s room at ")[1]] = indicator
                else:
                    iter_places[pname] = indicator
                i += 1
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

def find_place_index_in_building(building_meta, place_name):
    for building_key, building in building_meta.items():
        for index, place in enumerate(building.get("places", [])):
            if place["name"] == place_name:
                return index
    raise Exception(f"{place_name} not found in building_meta")

class CharacterGen:
    def __init__(self, args) -> None:
        self.args = args
        self.building_metadata = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "building_metadata.json"), 'r'))
        self.place_metadata = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "place_metadata.json"), 'r'))
        self.all_bboxes = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "all_loaded_building_bboxes.json"), 'r'))
        self.character_name_to_skin_info = json.load(open(os.path.join(ASSETS_PATH, "character2skin.json"), 'r'))
        self.coarse_indoor_scene = json.load(open(os.path.join(ASSETS_PATH, "coarse_type_to_indoor_scene.json"), 'r'))
        if os.path.exists(os.path.join(ASSETS_PATH, "scenes", args.scene, "transit.json")):
            self.transit = json.load(open(f"assets/scenes/{args.scene}/transit.json", 'r'))
        else:
            self.transit = None
        self.character_name_to_image_features = pickle.load(open(os.path.join(ASSETS_PATH, "character_name_to_image_features.pkl"), "rb"))
        self.scene_to_name = {
            "EL_PASO": "El Paso",
            "FORT_WORTH": "Fort Worth",
            "KANSAS_CITY": "Kansas City",
            "LONGISLAND": "Long Island",
            "MADRID2": "Madrid",
            "MIT": "MIT Campus",
            "NY": "New York City",
            "SANFRANCISCO2": "San Francisco",
            "STANFORD": "Stanford Campus",
            "UMASS": "UMass Campus",
            "YALE": "Yale Campus",
            "HARVARD": "Harvard Campus",
            "UCLA": "UCLA Campus",
            "SILICONVALLEY": "Silicon Valley",
            "LASVEGAS": "Las Vegas",
        }
        if args.scene not in self.scene_to_name:
            self.scene_to_name[args.scene] = args.scene.lower().capitalize()
        print("Scene Name:", self.scene_to_name[args.scene])
        if False: # os.path.exists(os.path.join(args.output_dir, "place_metadata.json")):
            self.place_metadata = json.load(open(os.path.join(args.output_dir, "place_metadata.json"), 'r'))
            self.building_metadata = update_buildings(self.building_metadata, self.place_metadata)
            self.assign_scene_to_place_metadata()
        else:
            # counting number of places of types first
            cnt_types = defaultdict(int)
            for place in self.place_metadata:
                cnt_types[self.place_metadata[place]["coarse_type"]] += 1
            if cnt_types["stores"] < 6 or cnt_types["accommodation"] < 1 or cnt_types["food"] < 3 or cnt_types["entertainment"] < 1:
                print(f"Not enough places for stores or accommodation or food or entertainment. The minimum requirements are: stores: 6, accommodation: 1, food: 3, entertainment: 1. Number of stores: {cnt_types['stores']}, accommodation: {cnt_types['accommodation']}, food: {cnt_types['food']}, entertainment: {cnt_types['entertainment']}.")
                exit()
            self.assign_scene_to_place_metadata()
            self.choose_6_stores()

    def assign_scene_to_place_metadata(self):
        for place in self.place_metadata:
            if "scene" in self.place_metadata[place]:
                print(f"Place {place} has pre-assigned scene {self.place_metadata[place]['scene']}. Skipping assignment.")
                continue
            if self.place_metadata[place]["building"] != "open space":
                self.place_metadata[place]["scene"] = random.sample(self.coarse_indoor_scene[self.place_metadata[place]['coarse_type']], 1)[0]
            else:
                self.place_metadata[place]["scene"] = None

    def choose_6_stores(self):
        print("Number of places before choosing stores:", len(self.place_metadata))
        store_places = []
        for building in self.building_metadata:
            for place in self.building_metadata[building]["places"]:
                if place["coarse_type"] == "stores":
                    store_places.append((building, place))
        selected_stores = []
        selected_buildings = set()
        while len(selected_stores) < 6 and store_places:
            building, place = random.choice(store_places)
            if building not in selected_buildings:
                selected_stores.append((building, place))
                selected_buildings.add(building)
            store_places.remove((building, place))
        scene_index = 0
        for building in self.building_metadata:
            chosen_places = []
            for place in self.building_metadata[building]["places"]:
                if place["coarse_type"] != "stores":
                    chosen_places.append(place)
                else:
                    if place["name"] in [store[1]["name"] for store in selected_stores]:
                        updated_place = copy.deepcopy(place)
                        updated_place["scene"] = self.coarse_indoor_scene["stores"][scene_index]
                        chosen_places.append(updated_place)
                        scene_index += 1
            self.building_metadata[building]["places"] = chosen_places
            # Also update the place_metadata
            for place in self.building_metadata[building]["places"]:
                if place["coarse_type"] != "stores":
                    continue
                for store in selected_stores:
                    if place["name"] == store[1]["name"]:
                        self.place_metadata[place["name"]]["scene"] = place["scene"]
                        break
        copy_place_metadata = copy.deepcopy(self.place_metadata)
        for place in copy.deepcopy(copy_place_metadata):
            if copy_place_metadata[place]["coarse_type"] == "stores" and place not in [store[1]["name"] for store in selected_stores]:
                del self.place_metadata[place]
        self.building_metadata = update_buildings(self.building_metadata, self.place_metadata)
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(self.place_metadata, open(os.path.join(args.output_dir, "place_metadata.json"), 'w'), indent=4)
        self.check_store_scenes(self.place_metadata)
        print("Number of places after choosing stores:", len(self.place_metadata))

    def extract_json_blocks(self, text):
        json_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL)
        json_objects = []
        for block in json_blocks:
            try:
                json_obj = json.loads(block.strip())
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        return json_objects
    
    def random_point_on_bbox_edge(self, min_x, min_y, max_x, max_y, abs_max=385):
        found = False
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
                return (x, y)
            print("Retry random point on bbox edge...")

    def is_point_in_bounding_box(self, x, y, bounding_box_3d):
        x_coords = [point[0] for point in bounding_box_3d]
        y_coords = [point[1] for point in bounding_box_3d]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return min_x <= x <= max_x and min_y <= y <= max_y
        
    # def sample_location_on_extended_bounding_box(self, bounding_box_3d, all_bounding_boxes_3d, extension=1): # deprecated, moved to generate_scene_metadata.py
    #     found = False
    #     retry_times = 0
    #     while not found:
    #         x_coords = [point[0] for point in bounding_box_3d]
    #         y_coords = [point[1] for point in bounding_box_3d]
    #         min_x, max_x = min(x_coords), max(x_coords)
    #         min_y, max_y = min(y_coords), max(y_coords)
    #         extended_min_x, extended_max_x = min_x - extension, max_x + extension
    #         extended_min_y, extended_max_y = min_y - extension, max_y + extension
    #         random_point = self.random_point_on_bbox_edge(extended_min_x, extended_min_y, extended_max_x, extended_max_y)
    #         found = True
    #         for each_bounding_box_3d in all_bounding_boxes_3d:
    #             if each_bounding_box_3d is not None:
    #                 if self.is_point_in_bounding_box(random_point[0], random_point[1], each_bounding_box_3d):
    #                     found = False
    #                     break
    #         print("retry in sample_location_on_extended_bounding_box.")
    #     return random_point
    
    def euclidean_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def sample_additional_places(self, metadata, known_places, type, num):
        known_places_locations = []
        for building in metadata:
            building_places = metadata[building]["places"]
            for building_place in building_places:
                if building_place['coarse_type'] != 'transit':
                    if building_place["name"] in known_places:
                        known_places_locations.append(building_place["location"])
        assert len(known_places_locations) > 0
        candidate_places = []
        for building in metadata:
            building_places = metadata[building]["places"]
            for building_place in building_places:
                if building_place['coarse_type'] != 'transit':
                # if (building_place["coarse_type"] == type and
                #     building_place["name"] not in known_places):
                    if building_place["coarse_type"] == type:
                        place_location = building_place["location"]
                        min_distance = min(
                            [self.euclidean_distance(place_location, known_loc) for known_loc in known_places_locations]
                        )
                        candidate_places.append((building_place["name"], min_distance))
        epsilon = 1e-9
        names = []
        weights = []
        for name, min_distance in candidate_places:
            names.append(name)
            weights.append(1 / (min_distance + epsilon))
        weights = np.array(weights)
        total_weight = weights.sum()
        if total_weight == 0:
            probabilities = np.ones(len(weights)) / len(weights)
        else:
            probabilities = weights / total_weight
        assert num <= len(candidate_places)
        names = [name for name, _ in candidate_places]
        selected_names = np.random.choice(names, size=num, replace=False, p=probabilities)

        return list(selected_names)
    
    def polish_verbalization(self, verbalization: str) -> str:
        polish_prompt = """Polish the following text to make it more coherent and fluent. You can make changes to the text, but do not change the meaning of the text. Make sure the text is grammatically correct and free of errors. NOTE: Please use a first-person tone.""" + '\n\n'
        polish_prompt += verbalization
        polished_output = self.generator.generate(polish_prompt)
        return polished_output
    
    def character_names_to_skin_names(self, character_names, skin_names):
        curr_skin_names = skin_names.copy()
        skin_names_ordered_list = []
        for character_name in character_names:
            curr_skin_names_formatted = ','.join(curr_skin_names)
            curr_skin_names_text = f"[{curr_skin_names_formatted}]"
            prompt = f"Given a person's name is {character_name}, output only the most likely pseudonym in the following names list: {curr_skin_names_text}. Output: "
            raw_gpt_response = self.generator.generate(prompt)
            if raw_gpt_response in curr_skin_names:
                skin_names_ordered_list.append(raw_gpt_response)
                curr_skin_names.remove(raw_gpt_response)
            else:
                raise ValueError(f"Generated name {raw_gpt_response} is not in curr_skin_names_text {curr_skin_names_text}")
        return skin_names_ordered_list
    
    def check_store_scenes(self, place_metadata):
        print('-' * 20)
        for place in place_metadata:
            if place_metadata[place]["coarse_type"] == "stores":
                print(f"Place {place} has scene {place_metadata[place]['scene']}")
        print('-' * 20)

    def needs_unicode_escape_decode(self, s):
        return bool(re.search(r'\\u[0-9a-fA-F]{4}', s))

    def execute(self):
        # Step 1: Generate groups and characters using LLM
        groups = None
        characters = None

        if not args.event:
            middle_path = f"{args.scene}"
            base_path = f"assets/scenes/{args.scene}/gpt_cache/g{args.num_groups}c{args.num_characters}"
        else:
            middle_path = os.path.join(f"{args.scene}", "events", args.event)
            base_path = os.path.join(f"assets/scenes/{args.scene}", "gpt_cache", "events", f"{args.event}", f"g{args.num_groups}c{args.num_characters}")
        
        groups_json_path = os.path.join(base_path, "groups.json")
        characters_json_path = os.path.join(base_path, "characters.json")
        print("groups_json_path:", groups_json_path)
        print("characters_json_path:", characters_json_path)
        raw_gpt_response = None
        if os.path.exists(groups_json_path) and os.path.exists(characters_json_path) and not args.regenerate:
            print("groups_json and characters_json exist, no generation.")
            groups = json.load(open(groups_json_path, 'r'))
            characters = json.load(open(characters_json_path, 'r'))
            character_names = list(characters.keys())
            grounding_success = True
        else:
            mixamo_names = [name for name in self.character_name_to_skin_info.keys() if self.character_name_to_skin_info[name]["skin_file"][:6] == "mixamo"]
            custom_names = [name for name in self.character_name_to_skin_info.keys() if self.character_name_to_skin_info[name]["skin_file"][:6] == "custom"]
            print(f"# Total Available Characters: {len(mixamo_names)} Mixamo + {len(custom_names)} Custom")
            random.shuffle(mixamo_names)
            if args.event:
                character_names = custom_names # mixamo_names[:13] + ["Liam Novak", "Yara Mbatha"]
            elif args.num_characters == 1:
                character_names = ["James Thompson"]
            else:
                num_custom_skin = args.num_characters // 3
                num_mixamo = args.num_characters - num_custom_skin
                random.shuffle(mixamo_names)
                selected_mixamo_names = mixamo_names[:num_mixamo]
                custom_skin_names = [name for name in self.character_name_to_skin_info.keys() if name not in mixamo_names]
                random.shuffle(custom_skin_names)
                selected_custom_skin_names = custom_skin_names[:num_custom_skin]
                character_names = selected_mixamo_names + selected_custom_skin_names

            character_names = sorted(character_names)
            print("Selected Characters:", character_names)

            characters_initial_information = {}
            characters_initial_information_verbalized = "Characters: \n"
            
            for character_name in character_names:
                # characters_initial_information.append(f"Character Name: {character_name}, Character Age: {self.character_name_to_skin_info[character_name]}")
                characters_initial_information[character_name] = {} 
                characters_initial_information[character_name]["age"] = self.character_name_to_skin_info[character_name]["age"]
                if character_name in mixamo_names:
                    characters_initial_information[character_name]["famous"] = False
                else:
                    characters_initial_information[character_name]["famous"] = True
            # characters_initial_information_verbalized += "; ".join(characters_initial_information)
            characters_initial_information_verbalized += repr(characters_initial_information)

            places_information_verbalized = "Places: \n"
            if self.args.use_max_120:
                required_places_per_type = {}
                place_type_stats = json.load(open(f"assets/scenes/{self.args.scene}/raw/type_stats_accessible.json", 'r'))
                for coarse_type, count in place_type_stats.items():
                    required_places_per_type[coarse_type] = min(20, count)
                places_by_type = {}
                for building in self.building_metadata:
                    building_places = self.building_metadata[building]["places"]
                    for building_place in building_places:
                        if building_place['coarse_type'] != 'transit':
                            coarse_type = building_place["coarse_type"]
                            places_by_type.setdefault(coarse_type, []).append(building_place)
                places_information = {}
                num_places = 0
                for coarse_type, required_num in required_places_per_type.items():
                    places_list = places_by_type.get(coarse_type, [])
                    selected_places = places_list[:required_num]
                    for building_place in selected_places:
                        if num_places < 120:
                            places_information[building_place["name"]] = {
                                "coarse_type": building_place["coarse_type"],
                                "fine_types": building_place["fine_types"],
                            }
                            num_places += 1
                        else:
                            break
                if num_places < 120:
                    for building in self.building_metadata:
                        building_places = self.building_metadata[building]["places"]
                        for building_place in building_places:
                            if building_place['coarse_type'] != 'transit':
                                name = building_place["name"]
                                if name not in places_information:
                                    if num_places < 120:
                                        places_information[name] = {
                                            "coarse_type": building_place["coarse_type"],
                                            "fine_types": building_place["fine_types"],
                                        }
                                        num_places += 1
            else:
                places_information = {}
                num_places = 0
                for building in self.building_metadata:
                    building_places = self.building_metadata[building]["places"]
                    for building_place in building_places:
                        if building_place['coarse_type'] != 'transit':
                            places_information[building_place["name"]] = {}
                            places_information[building_place["name"]]["coarse_type"] = building_place["coarse_type"]
                            places_information[building_place["name"]]["fine_types"] = building_place["fine_types"]
                            num_places += 1
            places_information_verbalized += repr(places_information)
            print("Number of places:", num_places)

            grounding_success = False
            chat_history = []
            prompt_template_generate_instructions = open('assets/prompt_character_gen_generate_instructions.txt', 'r').read()
            generate_instructions = prompt_template_generate_instructions.replace("$places_information_verbalized$", places_information_verbalized)
            generate_instructions = generate_instructions.replace("$characters_initial_information_verbalized$", characters_initial_information_verbalized)
            generate_instructions = generate_instructions.replace("$num_groups$", str(args.num_groups))
            generate_instructions = generate_instructions.replace("$num_characters$", str(args.num_characters))
            generate_instructions = generate_instructions.replace("$scene_name$", self.scene_to_name[args.scene])
            generate_instructions = generate_instructions.replace("$group_least_members$", str(args.num_characters // args.num_groups))

            if not args.event:
                prompt = f"You will be given a real-world scene '{self.scene_to_name[args.scene]}' and available places (dictionary) as well as a dictionary of characters information situated in this scene. " + '\n' + generate_instructions
            else:
                event_description = "Liam Novak and Yara Mbatha want to make friends with other characters in the scene."
                prompt = f"You will be given a real-world scene '{self.scene_to_name[args.scene]}' and {event_description} is happening in this scene, and available places (dictionary) as well as a dictionary of characters information situated in this scene. " + '\n' + generate_instructions
            
            if args.only_copy_prompt:
                import pyperclip
                pyperclip.copy(prompt)
                print("Prompt copied to clipboard. Exiting...")
                exit()

            if args.force_check_grounding:
                print("force_check_grounding option is not allowed when gpt_cache does not exist. Exiting...")
                exit()
            print("groups_json and characters_json not exist or you used regenerate option, start generation.")
            self.generator = Generator(lm_source=args.lm_source, lm_id='gpt-4o', max_tokens=16384, temperature=0, top_p=1, logger=None)
            os.makedirs(base_path, exist_ok=True)
            raw_gpt_response = self.generator.generate(prompt)
            # print("Debug: raw_gpt_response:", raw_gpt_response)
            dicts_returned = self.extract_json_blocks(raw_gpt_response)
            characters = dicts_returned[0]
            groups = dicts_returned[1]
            # json.dump(characters, open(characters_json_path, 'w'), indent=4)
            # json.dump(groups, open(groups_json_path, 'w'), indent=4)
            for group in groups:
                groups[group]["members"] = sorted(groups[group]["members"])
                
            max_allowable_retries = 15
            num_retries = 0
            # Step 1.5: Validate whether the generated jsons are consistent and grounded to the scene
            while not grounding_success or args.force_check_grounding:

                if num_retries > max_allowable_retries:
                    print("Number of retries exceeds max allowable retries. Exiting...")
                    exit()
                
                # sort characters according to names
                characters = dict(sorted(characters.items()))

                error_messages = []
                for group_name in groups.keys():
                    # print(groups[group_name])
                    if groups[group_name]["place"] not in places_information.keys():
                        error_messages.append(f"Group {group_name}: place {groups[group_name]['place']} not exists in the scene's places.")
                # Groups validation completes
                # Add groups into character
                for group_name in groups.keys():
                    group = groups[group_name]
                    for member_name in group["members"]:
                        if "groups" not in characters[member_name]:
                            characters[member_name]["groups"] = []
                        if group_name not in characters[member_name]["groups"]:
                            characters[member_name]["groups"].append(group_name)
                every_character_has_group = True
                for character_name in characters.keys():
                    if "groups" not in characters[character_name]:
                        every_character_has_group = False
                        error_messages.append(f"{character_name} does not have any group. Every character should have only one group.")
                if every_character_has_group:
                    for character_name in characters.keys():
                        if len(characters[character_name]["groups"]) > 1:
                            error_messages.append(f"{character_name} has more than one group. Every character should have only one group.")
                        for group_in_character in characters[character_name]["groups"]:
                            # if group_in_character not in groups.keys():
                            #     error_messages.append(f"Character {character_name}: group {group_in_character} not exists in the groups.")
                            if groups[group_in_character]["place"] not in characters[character_name]["known_places"]:
                                characters[character_name]["known_places"].append(groups[group_in_character]["place"])
                        if characters[character_name]["living_place"] not in places_information.keys():
                            error_messages.append(f"Character {character_name}: living_place {characters[character_name]['living_place']} not exists in the scene's places.")
                        elif places_information[characters[character_name]["living_place"]]["coarse_type"] != "accommodation":
                            error_messages.append(f"Character {character_name}: living_place {characters[character_name]['living_place']} is not accommodation.")
                        for other_place in characters[character_name]["known_places"]:
                            if other_place not in places_information.keys():
                                error_messages.append(f"Character {character_name}: known_place {other_place} not exists in the scene's places.")
                        if characters[character_name]["working_place"] is not None and characters[character_name]["working_place"] not in places_information.keys():
                            error_messages.append(f"working_place {characters[character_name]['working_place']} not exists in the scene's places.")
                        working_place_candidates = []
                        for place in places_information.keys():
                            if place in characters[character_name]["learned"]:
                                working_place_candidates.append(place)
                        found_one_valid_working_place = False
                        if len(working_place_candidates) > 0:
                            for working_place in working_place_candidates:
                                if working_place == characters[character_name]["working_place"]:
                                    found_one_valid_working_place = True
                        if found_one_valid_working_place:
                            if characters[character_name]["working_place"] not in characters[character_name]["known_places"]:
                                characters[character_name]["known_places"].append(characters[character_name]["working_place"])
                        if len(working_place_candidates) > 0 and found_one_valid_working_place == False:
                            error_messages.append(f"Character {character_name}: the working_place does not match any working place in [{', '.join(working_place_candidates)}] generated from the learned information.")
                        if len(working_place_candidates) == 0:
                            if characters[character_name]["working_place"] is not None and characters[character_name]["working_place"] not in characters[character_name]["known_places"]:
                                characters[character_name]["known_places"].append(characters[character_name]["working_place"])
                    # Check # group members >= 2
                    for group_name in groups.keys():
                        # groups_info.append(f"{group_name}:{groups[group_name]['description']}")
                        group_member_names = groups[group_name]["members"]
                        if len(group_member_names) < (args.num_characters // args.num_groups):
                            error_messages.append(f"Group {group_name}: only has {len(group_member_names)} member: [{', '.join(group_member_names)}]. Every group must have at least {args.num_characters // args.num_groups} members. Consider regenerate groups and assign characters more evenly.")
                    # Check # groups
                    if len(groups.keys()) != args.num_groups:
                        error_messages.append(f"Number of groups generated is {len(groups.keys())}, not equal to the expected number of groups {args.num_groups}. Consider regenerate groups.")
                # Characters validation completes
                if len(error_messages) > 0:
                    if args.force_check_grounding:
                        print("You used force_check_grounding option, and your error messages for the config are:")
                        print("Error messages:", error_messages)
                        print("Exiting...")
                        exit()        
                    num_retries += 1
                    if raw_gpt_response is not None:
                        chat_history.append({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": prompt
                            }]
                        })
                        chat_history.append({
                            "role": "system",
                            "content": [{
                                "type": "text",
                                "text": raw_gpt_response
                            }]
                        })
                        error_messages_verbalize = "Validation fails because of these error messages: \n" + '\n'.join(error_messages) + '\n' + \
                                                f"Instruction: revise the output. Only return 2 JSON objects (both are dictionary of dictionaries). The first JSON contains {args.num_characters} characters (don't wrap it in a 'characters' key, the keys of the first JSON are character names) and the second JSON contains {args.num_groups} groups (don't wrap it in a 'groups' key, the keys of the second JSON are group names)."
                        prompt = error_messages_verbalize
                        print("Retrying...prompt:", prompt)
                        # print("chat_history:", chat_history)
                        raw_gpt_response = self.generator.generate(prompt, chat_history=chat_history)
                    else:
                        raw_gpt_response = self.generator.generate(prompt)
                    # print("Debug: raw_gpt_response:", raw_gpt_response)
                    dicts_returned = self.extract_json_blocks(raw_gpt_response)
                    characters = dicts_returned[0]
                    groups = dicts_returned[1]
                else:
                    print("Passed grounding validator!")
                    if args.force_check_grounding:
                        print("You used force_check_grounding option, grounding is successful.")
                        args.force_check_grounding = False
                    json.dump(characters, open(characters_json_path, 'w'), indent=4)
                    json.dump(groups, open(groups_json_path, 'w'), indent=4)
                    print("Saved characters.json with groups.")
                    grounding_success = True
        
        # Step 2: Initiate dining places to characters
        if not args.regenerate and "entertainment_places" in characters[character_names[0]]:
            print("dining, store, and entertainment places already exist, no resample.")
        else:
            for character_name in characters.keys():
                characters[character_name]["dining_places"] = self.sample_additional_places(metadata=self.building_metadata, known_places=[characters[character_name]["living_place"]]+characters[character_name]["known_places"], type="food", num=3)
                characters[character_name]["store_places"] = self.sample_additional_places(metadata=self.building_metadata, known_places=[characters[character_name]["living_place"]]+characters[character_name]["known_places"]+characters[character_name]["dining_places"], type="stores", num=2)
                characters[character_name]["entertainment_places"] = self.sample_additional_places(metadata=self.building_metadata, known_places=[characters[character_name]["living_place"]]+characters[character_name]["known_places"]+characters[character_name]["dining_places"]+characters[character_name]["store_places"], type="entertainment", num=1)
            json.dump(characters, open(characters_json_path, 'w'), indent=4)
            print("Saved characters.json with dining, store, and entertainment places.")

        # Step 3: Modify stores names
        store_new_name_mapping = {
            "Beverages-Snacks": "Beverages and Snacks Store",
            "Fresh-DM": "Fresh and DM Store",
            "Snacks-DM": "Snacks and DM Store",
            "Beverages-DM": "Beverages and DM Store",
            "Fresh-Snacks": "Fresh and Snacks Store",
            "Fresh-Beverages": "Fresh and Beverages Store",
        }
        # Assign scene to building_metadata
        for building in self.building_metadata:
            for place_i in range(len(self.building_metadata[building]["places"])):
                if "scene" in self.place_metadata[self.building_metadata[building]["places"][place_i]["name"]]:
                    self.building_metadata[building]["places"][place_i]["scene"] = self.place_metadata[self.building_metadata[building]["places"][place_i]["name"]]["scene"]
        replaced_stores = []
        place_metadata_copy = copy.deepcopy(self.place_metadata)
        for place in place_metadata_copy:
            if place_metadata_copy[place]["coarse_type"] == "stores":
                store_new_name = store_new_name_mapping[place_metadata_copy[place]["scene"].split('/store-')[-1].split(".json")[0]]
                if place == store_new_name:
                    continue
                replaced_stores.append((place, store_new_name))
                self.place_metadata[store_new_name] = place_metadata_copy[place]
                del self.place_metadata[place]
        print("Replaced stores:", replaced_stores)
        building_metadata_copy = copy.deepcopy(self.building_metadata)
        for building in building_metadata_copy:
            temp_places = []
            for building_place in building_metadata_copy[building]["places"]:
                if building_place["coarse_type"] == "stores":
                    temp_place = copy.deepcopy(building_place)
                    temp_place["name"] = store_new_name_mapping[building_place["scene"].split('/store-')[-1].split(".json")[0]]
                    temp_places.append(temp_place)
                else:
                    temp_places.append(building_place)
            self.building_metadata[building]["places"] = temp_places
        
        # Replace store names in characters_json and groups_json
        with open(characters_json_path, 'r') as f:
            characters_json = f.read()
            if self.needs_unicode_escape_decode(characters_json):
                characters_json = characters_json.encode().decode("unicode_escape") # Note: this **may** cause bugs of string matching in dict rarely (if generating the characters from cache)
            for place, store_new_name in replaced_stores:
                characters_json = characters_json.replace(place, store_new_name)
            with open(characters_json_path, 'w') as f:
                f.write(characters_json)
        with open(groups_json_path, 'r') as f:
            groups_json = f.read()
            if self.needs_unicode_escape_decode(groups_json):
                groups_json = groups_json.encode().decode("unicode_escape") # Note: this **may** cause bugs of string matching in dict rarely (if generating the characters from cache)
            for place, store_new_name in replaced_stores:
                groups_json = groups_json.replace(place, store_new_name)
            with open(groups_json_path, 'w') as f:
                f.write(groups_json)
        groups = json.load(open(groups_json_path, 'r'))
        characters = json.load(open(characters_json_path, 'r'))

        #! Step 4: Generate characters specifications
        print('==> Generate characters specifications...')
        agent_poses = []
        character_spawn_poses = []
        structure_to_create = {}
        initial_cash_values = []
        all_character_scratch = []
        all_known_places_set = set()
        all_colors = ["red", "green", "blue", "orange", "purple", "yellow", "pink", "cyan", "magenta", "teal", "maroon", "navy", "olive", "lime", "aqua", "coral", "beige", "plum", "lavender", "indigo", "salmon", "turquoise", "peru", "gold", "silver"]
        num_agents = len(character_names)
        stores = {}
        for place_name, place_dict in self.place_metadata.items():
            if place_dict["coarse_type"] == "stores":
                store_type = place_dict["scene"].split('/')[-1].split('.json')[0]
                stores[place_name] = {
                    "building": place_dict["building"],
                    "scene": place_dict["scene"],
                    "type": store_type,
                    "goods": [
                        {
                            "name": good_name,
                            "price": price,
                            "quantity": 5
                        } for good_name, price in ENV_OTHER_METADATA["goods_price_per_store"][store_type].items()
                    ]
                }
        this_config = {"sim_name": f"{args.scene}_agents_num_{num_agents}",
                        "agent_names": character_names,
                        "step": 0, "agent_infos": [],
                        "curr_time": "October 1, 2024, 09:00:00",
                        "start_time": "October 1, 2024, 09:00:00",
                        "agent_poses": agent_poses,
                        "locator_colors": all_colors[:num_agents],
                        "locator_colors_rgb": map_lang_colors_to_rgb(all_colors[:num_agents]),
                        "num_agents": num_agents,
                        "sec_per_step": 1,
                        "agent_skins": [],
                        "stores": stores,
                        "dt_control": [1.0] * num_agents,
                        "dt_visual_obs": [1.0] * num_agents,
                       }
        # for group_name in characters[character_name]["groups"]:
        #     groups[group_name]["daily_activity"] = f"Group member should go to {groups[group_name]['place']} for {groups[group_name]['daily_activity']} in {group_name}."
        previous_locations = []
        for idx, character_name in enumerate(character_names):
            character_scratch = dict(init_scratch)
            character_scratch["daily_requirement"] = ""
            # character_scratch["daily_requirement"] += daily_activity_verbalize
            character_scratch["name"] = character_name
            character_scratch["first_name"] = character_name.split(' ')[0]
            character_scratch["last_name"] = character_name.split(' ')[-1]
            character_scratch["age"] = characters[character_name]["age"]
            character_scratch["innate"] = ", ".join(characters[character_name]["values"])
            character_scratch["learned"] = characters[character_name]['learned']
            character_scratch["working_place"] = characters[character_name]['working_place']
            character_scratch["groups"] = []
            character_scratch["currently"] = characters[character_name]['currently']
            character_scratch["lifestyle"] = characters[character_name]["lifestyle"]
            character_scratch["start_time"] = this_config["start_time"]
            character_scratch["curr_time"] = this_config["curr_time"]

            if args.event and args.event == "campaign":
                if character_name == "Liam Novak":
                    character_scratch["innate"] = "power, achievement, security"
                    character_scratch["currently"] = "I am here for making friends."
                    character_scratch["lifestyle"] = "I go to bed around midnight, wake up around 08:00, eat dinner around 18:00"
                elif character_name == "Yara Mbatha":
                    character_scratch["innate"] = "universalism, benevolence, achievement"
                    character_scratch["currently"] = "I am here for making friends."
                    character_scratch["lifestyle"] = "I go to bed around midnight, wake up around 08:00, eat dinner around 18:00"

            seed_knowledge = {}
            seed_knowledge_feature = {}
            # groups_info = []
            for group_name in characters[character_name]["groups"]:
                # groups_info.append(f"{group_name}:{groups[group_name]['description']}")
                other_character_names = []
                for other_character_name in characters:
                    if other_character_name != character_name:
                        if group_name in characters[other_character_name]["groups"]:
                            other_character_names.append(other_character_name)
                            if other_character_name not in seed_knowledge:
                                appearance = f"assets/imgs/avatars/{other_character_name}.png"
                                seed_knowledge[other_character_name] = {
                                                                        "age": characters[other_character_name]["age"],
                                                                        # "innate": ", ".join(characters[other_character_name]["values"]),
                                                                        # "learned": characters[other_character_name]['learned'],
                                                                        # "currently": characters[other_character_name]['currently'],
                                                                        # "lifestyle": characters[other_character_name]["lifestyle"],
                                                                        "living_place": characters[other_character_name]["living_place"],
                                                                        "appearance": appearance,
                                                                        "groups": []
                                                                        }
                                if other_character_name in self.character_name_to_image_features:
                                    seed_knowledge_feature[other_character_name] = self.character_name_to_image_features[other_character_name]
                                else:
                                    seed_knowledge_feature[other_character_name] = None
                                    print(f"Warning: {other_character_name} not in character_name_to_image_features.")
                            seed_knowledge[other_character_name]["groups"].append(group_name)
                seed_knowledge[group_name] = groups[group_name]
                if args.event and args.event == "campaign":
                    other_character_names = [other_character_name for other_character_name in other_character_names if other_character_name != "Liam Novak"]
                    other_character_names = [other_character_name for other_character_name in other_character_names if other_character_name != "Yara Mbatha"]
                group_info = groups[group_name].copy()
                group_info = {"name": group_name, **group_info}
                character_scratch["groups"].append(group_info)

            if args.event and args.event == "campaign":
                if character_name == "Liam Novak" or character_name == "Yara Mbatha":
                    character_scratch["groups"] = []

            initial_cash_values.append(characters[character_name]["initial_cash_value"])
            living_place = characters[character_name]["living_place"]
            living_building = self.place_metadata[living_place]["building"]

            private_living_place = f"{character_name}'s room at {living_place}"
            all_known_places_set.add(living_place)
            private_living_place_dict = deepcopy(self.place_metadata[living_place])
            private_living_place_dict["name"] = private_living_place
            private_living_place_dict["location"][-1] = -4 * len(self.building_metadata[living_building]["places"]) - 4
            private_living_place_dict["bounding_box"] = self.building_metadata[living_building]["bounding_box"]

            self.place_metadata[private_living_place] = private_living_place_dict
            self.building_metadata[living_building]["places"].append(private_living_place_dict)

            character_scratch["living_place"] = private_living_place
            characters[character_name]["living_place"] = private_living_place
            spatial_memory = {}
            # add other groups' places to known_places
            for group_name in groups.keys():
                group_place = groups[group_name]["place"]
                if group_place not in characters[character_name]["known_places"]:
                    characters[character_name]["known_places"].append(group_place)
            known_places = [characters[character_name]["living_place"]] + characters[character_name]["known_places"] + characters[character_name]["dining_places"] + characters[character_name]["store_places"] + characters[character_name]["entertainment_places"]
            for place in known_places:
                try:
                    spatial_memory[place] = deepcopy(self.place_metadata[place])
                    spatial_memory[place]["bounding_box"] = self.building_metadata[spatial_memory[place]["building"]]["bounding_box"]
                    all_known_places_set.add(place)
                    seed_knowledge[place] = spatial_memory[place]
                except KeyError:
                    try:
                        place = place.encode('latin1').decode('utf-8')
                        spatial_memory[place] = deepcopy(self.place_metadata[place])
                        spatial_memory[place]["bounding_box"] = self.building_metadata[spatial_memory[place]["building"]]["bounding_box"]
                        all_known_places_set.add(place)
                        seed_knowledge[place] = spatial_memory[place]
                    except KeyError as e:
                        print(f"{e.__class__.__name__}: {e} {traceback.format_exc()}")
                        print(f"KeyError: {place} not in place_metadata. Have tried the latin1 encoding/decoding fix but failed.")
                        raise
            # add transit places to seed_knowledge
            for bus_stop, bus_stop_info in self.transit["bus"]["stops"].items():
                location = bus_stop_info["position"]
                rotation = bus_stop_info["target_rad"]
                seed_knowledge[bus_stop] = {
                    "building": "open space",
                    "coarse_type": "transit",
                    "fine_type": "bus stop",
                    "location": location,
                    "rotation": rotation,
                    "bounding_box": None,
                }
            for bike_station in self.transit["bicycle"]["stations"]:
                seed_knowledge[bike_station] = {
                    "building": "open space",
                    "coarse_type": "transit",
                    "fine_type": "bike station",
                    "location": self.transit["bicycle"]["stations"][bike_station],
                    "bounding_box": None,
                }
                
            seed_knowledge["transit_schedule"] = self.transit["bus"]["schedule"]

            if args.event and args.event == "campaign":
                if character_name == "Liam Novak" or character_name == "Yara Mbatha":
                    spatial_memory = {}
                    seed_knowledge = {}
                    seed_knowledge_feature = {}
                    for place in self.place_metadata.keys():
                        spatial_memory[place] = deepcopy(self.place_metadata[place])
                        spatial_memory[place]["bounding_box"] = self.building_metadata[spatial_memory[place]["building"]]["bounding_box"]
                        all_known_places_set.add(place)
                        seed_knowledge[place] = spatial_memory[place]
            
            os.makedirs(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/{character_name}", exist_ok=True)
            # print(character_scratch["living_place"])
            # print(spatial_memory[character_scratch["living_place"]])
            character_spawn_location_xy = sample_location_on_extended_bounding_box_flood_fill(obstacle_grid, spatial_memory[character_scratch["living_place"]]["bounding_box"], obstacle_grid_parameters["resolution"], obstacle_grid_parameters["min_x"], obstacle_grid_parameters["min_y"], obstacle_grid_parameters["nx"], obstacle_grid_parameters["ny"], previous_locations=previous_locations)
            hx, hy = character_spawn_location_xy[0], character_spawn_location_xy[1]
            previous_locations = [hx, hy]
            new_height = get_height_at(height_field, hx, hy)
            if isinstance(new_height, np.ndarray):
                new_height = new_height.item()
            character_spawn_poses.append([character_spawn_location_xy[0], character_spawn_location_xy[1], new_height, 0.0, 0.0, 0.0])
            structure_to_create[f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/{character_name}"] = [{"file_name": "scratch.json", "file_content": character_scratch}, {"file_name": "seed_knowledge.json", "file_content": seed_knowledge}, {"file_name": "seed_knowledge_feature.pkl", "file_content": seed_knowledge_feature}]
            all_character_scratch.append(character_scratch)
            x, y, z = self.place_metadata[characters[character_name]["living_place"]]["location"]
            agent_poses.append([x, y, z, 0.0, 0.0, 0.0])
            

        for i in range(num_agents):
            this_config["agent_infos"].append({
                "cash": initial_cash_values[i],
                "held_objects": [None, None],
                # "physical_health": 1,
                # "mental_health": 1,
                # "condition": {"physical": None, "mental": None},
                "outdoor_pose": character_spawn_poses[i],
                "current_building": self.place_metadata[all_character_scratch[i]["living_place"]]["building"],
                "current_place": all_character_scratch[i]["living_place"],
                "current_vehicle": None,
            })
        this_config["agent_poses"] = agent_poses
        this_config["groups"] = groups

        for character_name in character_names:
            this_config["agent_skins"].append(f"avatars/models/{self.character_name_to_skin_info[character_name]['skin_file']}")
        structure_to_create[f"assets/scenes/{middle_path}/agents_num_{len(character_names)}"] = [{"file_name": "config.json", "file_content": this_config}]
        create_folders_and_files(structure_to_create)
        # all_known_places_set_dict = {}
        # for place in all_known_places_set:
        #     x, y, _ = self.place_metadata[place]["location"]
        #     all_known_places_set_dict[place] = [x, y]
        # json.dump(all_known_places_set_dict, open(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/known_places.json", "w"), indent=4)
        new_place_metadata = {}
        for place in all_known_places_set:
            new_place_metadata[place] = self.place_metadata[place]
            if "bounding_box" in new_place_metadata[place]:
                del new_place_metadata[place]["bounding_box"]
        scene_place_metadata = json.load(open(f"assets/scenes/{args.scene}/place_metadata.json", 'r'))
        for transit_place in scene_place_metadata.keys():
            if scene_place_metadata[transit_place]["coarse_type"] == "transit":
                if transit_place not in new_place_metadata:
                    new_place_metadata[transit_place] = scene_place_metadata[transit_place]
                    location_z = get_height_at(height_field, new_place_metadata[transit_place]["location"][0], new_place_metadata[transit_place]["location"][1])
                    new_place_metadata[transit_place]["location"].append(location_z)
                # This appended code is for solving the bug in event generation where some agents have access to all places, so known places = all places, but in this case, we still need to add the location_z.
                if transit_place in new_place_metadata and len(new_place_metadata[transit_place]["location"]) == 2:
                    location_z = get_height_at(height_field, new_place_metadata[transit_place]["location"][0], new_place_metadata[transit_place]["location"][1])
                    new_place_metadata[transit_place]["location"].append(location_z)
        json.dump(new_place_metadata, open(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/place_metadata.json", "w"), indent=4)
        json.dump(self.building_metadata, open(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/building_metadata.json", "w"), indent=4)
        # Also replace the schedule config if exists
        if os.path.exists(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}_with_schedules/place_metadata.json"):
            json.dump(new_place_metadata, open(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}_with_schedules/place_metadata.json", "w"), indent=4)
            json.dump(self.building_metadata, open(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}_with_schedules/building_metadata.json", "w"), indent=4)
            print("Replaced the place_metadata.json and building_metadata.json in schedule config.")
            for root, dirs, files in os.walk(f"assets/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}_with_schedules"):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read()
                        for place, store_new_name in replaced_stores:
                            content = content.replace(place, store_new_name)
                        with open(file_path, 'w') as f:
                            f.write(content)
            print("Replaced the store names in schedule config.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_characters", "-n", type=int, default=15)
    parser.add_argument("--num_groups", "-ng", type=int, default=4)
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--only_copy_prompt", action="store_true")
    parser.add_argument("--event", type=str)
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--use_max_120", action="store_true") # only use this if you fail many times to generate jsons that are grounded correctly
    parser.add_argument("--force_check_grounding", action="store_true") # only use this if there's existing gpt cache that may not be grounded correctly
    parser.add_argument("--filter_distance_square", type=float, default=300.0)
    parser.add_argument("--lm_source", type=str, default='azure', choices=['azure', 'openai'])
    args = parser.parse_args()
    print("args:", args)
    random.seed(hash(args.scene))

    if not args.event:
        middle_path = f"{args.scene}"
    else:
        middle_path = os.path.join(f"{args.scene}", "events", args.event)
    args.output_dir = f"assets/scenes/{middle_path}/agents_num_{args.num_characters}"

    if args.overwrite and os.path.exists(args.output_dir):
        print(f"Overwrite the output directory: {args.output_dir}")
        keep_file = "place_metadata.json"
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if item == keep_file:
                continue
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        # shutil.rmtree(args.output_dir)

    height_field_path = os.path.join(get_assets_dir(), f"ViCo/scene/v1/{args.scene}/height_field.npz")
    height_field = load_height_field(height_field_path)
    print("Loaded height field.")

    global_cam_parameters = json.load(open(f"assets/scenes/{args.scene}/global_cam_parameters.json", 'r'))
    print("Loaded global camera parameters.")

    obstacle_grid_save = pickle.load(open(f"assets/scenes/{args.scene}/obstacle_grid.pkl", 'rb'))
    obstacle_grid = obstacle_grid_save["grid"]
    obstacle_grid_parameters = obstacle_grid_save["parameters"]
    print("Loaded obstacle grid.")

    gen = CharacterGen(args)
    gen.execute()