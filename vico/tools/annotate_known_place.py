import io
import pickle
import shutil
import os
from collections import defaultdict

from copy import deepcopy
import random, json, argparse, os
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from vico.tools.utils import *
from vico.tools.annotate_scene import stitch_images_horizontally, generate_diverse_colors
from vico.tools.constants import google_map_coarse_to_types, ASSETS_PATH

wonderful_colors = [
    ("#FFA6A6"),
    ("#FFD782"),
    ("#C3FF82"),
    ("#82FFCF"),
    ("#9DBFFF"),
    ("#E29BF8")
]

def generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter, groups):
    sorted_ctypes = sorted(places_by_coarse_type.keys())
    col_height = 26
    info_height = 22
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, col_height))
    ax_left, ax_right = axes[0], axes[1]
    current_y_left = 0
    current_y_right = 0

    iter_places = {}
    # Sort accommodation: private rooms ('s room at ) first, then others alphabetically.
    if "accommodation" in places_by_coarse_type:
        places_by_coarse_type["accommodation"].sort(
            key=lambda pname: (0 if "'s room at " in pname else 1, pname.lower())
        )
    # Sort all other coarse types alphabetically.
    for c in places_by_coarse_type.keys():
        if c != "accommodation":
            places_by_coarse_type[c].sort(key=lambda pname: pname.lower())

    pname2color = {}
    def draw_entry(ax, start_y, pname, indicator):
        color = type_to_color[ctype]
        pname2color[pname] = color
        ax.add_patch(plt.Rectangle((0, start_y), 0.15, 0.15, facecolor=color, edgecolor="black", linewidth=1))
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
        counter = 1
        for pname in places_by_coarse_type[ctype]:
            if "'s room at " in pname and pname.split("'s room at ")[1] in iter_places:
                indicator = iter_places[pname.split("'s room at ")[1]]
            elif pname in iter_places:
                indicator = iter_places[pname]
            else:
                indicator = f"{letter}{counter}"
                if "'s room at " in pname:
                    iter_places[pname.split("'s room at ")[1]] = indicator
                else:
                    iter_places[pname] = indicator
                counter += 1
            if draw_on_left:
                current_y_left = draw_entry(ax_left, current_y_left, pname, indicator)
            else:
                current_y_right = draw_entry(ax_right, current_y_right, pname, indicator)
                increase_y_for_ctype = True
            if draw_on_left and ((info_height - current_y_left) < 1.5):
                draw_on_left = False
                increase_y_for_ctype = False
        if draw_on_left:
            current_y_left += 0.6
        else:
            if increase_y_for_ctype:
                current_y_right += 0.6

    # Draw groups' info

    current_y_left = info_height
    axes[0].hlines(y=info_height + 0.1, xmin=0, xmax=20, color='black', linewidth=2)
    ax_left.text(0.0, current_y_left, f"Groups", fontsize=14, fontweight='bold')
    current_y_left += 0.3
    for group in groups:
        pname = groups[group]['place']
        color = pname2color[pname]
        indicator = iter_places[pname]
        ax_left.add_patch(plt.Rectangle((0, current_y_left), 0.15, 0.15, facecolor=color, edgecolor="black", linewidth=1))
        ax_left.text(0.35, current_y_left + 0.1, f"{pname} ({indicator})", va='center', ha='left', fontsize=12)
        ax_left.text(0.35, current_y_left + 0.32, groups[group]['description'], va='center', ha='left', fontsize=10)
        current_y_left += 0.6


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

def overlay_locations_desp_on_image(place_metadata, building_metadata, save_path, known_places, groups):
    image_path = f"{ASSETS_PATH}/scenes/{args.scene}/global.png"
    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    coarse_types = list(google_map_coarse_to_types.keys())
    type_to_color = dict(zip(coarse_types, wonderful_colors[:len(coarse_types)]))
    coarse_types.append("transit")
    type_to_color["transit"] = "white"
    coarse_type_to_letter = {}
    padding = 5
    font = ImageFont.truetype(f"{ASSETS_PATH}/arial.ttf", args.font_size)
    for i, ctype in enumerate(sorted(coarse_types)):
        coarse_type_to_letter[ctype] = chr(ord('A') + i)
    places_by_coarse_type = defaultdict(list)
    place_indicators = {}
    for place_name in known_places:
        place = place_metadata[place_name]
        ctype = place["coarse_type"]
        if place_name in places_by_coarse_type[ctype]:
            continue
        places_by_coarse_type[ctype].append(place_name)
    # Ensure the same sorting as legend: accommodation rooms first, others alphabetically
    if "accommodation" in places_by_coarse_type:
        places_by_coarse_type["accommodation"].sort(
            key=lambda pname: (0 if "'s room at " in pname else 1, pname.lower())
        )
    for c in places_by_coarse_type.keys():
        if c != "accommodation":
            places_by_coarse_type[c].sort(key=lambda pname: pname.lower())
    for ctype in sorted(places_by_coarse_type.keys()):
        letter = coarse_type_to_letter[ctype]
        counter = 1
        for place_name in places_by_coarse_type[ctype]:
            base_name = place_name.split("'s room at ")[1] if "'s room at " in place_name else place_name
            if base_name in place_indicators:
                continue
            indicator = f"{letter}{counter}"
            place_indicators[base_name] = indicator
            counter += 1
    
    drew_indicators = []
    for place in known_places:
        x, y, _ = place_metadata[place]["location"]
        pixel_x, pixel_y = project_3d_to_2d_from_perspective_camera(np.array([x, y, get_height_at(height_field, x, y)]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"]))
        if place_metadata[place]["building"] != "open space":
            pixel_y += find_place_index_in_building(building_metadata, place) * (args.font_size + padding * 2 + 1)
        if place in place_indicators:
            text = place_indicators[place]
        else:
            text = place_indicators[place.split("'s room at ")[1]]
        if text in drew_indicators:
            continue
        color = type_to_color[place_metadata[place]["coarse_type"]]
        bbox = draw.textbbox((pixel_x, pixel_y), text, font=font)
        draw.rounded_rectangle(
            [(bbox[0] - padding, bbox[1] - padding), (bbox[2] + padding, bbox[3] + padding)],
            radius=5,
            fill=color,
            outline="black",
            width=1
        )
        draw.text((pixel_x, pixel_y), text, font=font, fill="black")
        drew_indicators.append(text)

    legend_image = generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter, groups)
    annotated_image = stitch_images_horizontally(annotated_image, legend_image)
    annotated_image.save(save_path)

class CharacterGen:
    def __init__(self, args) -> None:
        self.args = args

    def execute(self):
        if not args.event:
            middle_path = f"{args.scene}"
        else:
            middle_path = os.path.join(f"{args.scene}", "events", args.event)

        groups = None
        characters = None

        if not args.event:
            base_path = f"{ASSETS_PATH}/scenes/{args.scene}/gpt_cache/g{args.num_groups}c{args.num_characters}"
        else:
            base_path = os.path.join(f"{ASSETS_PATH}/scenes/{args.scene}", "gpt_cache", "events", f"{args.event}", f"g{args.num_groups}c{args.num_characters}")
        
        groups_json_path = os.path.join(base_path, "groups.json")
        characters_json_path = os.path.join(base_path, "characters.json")
        if os.path.exists(groups_json_path) and os.path.exists(characters_json_path):
            groups = json.load(open(groups_json_path, 'r'))
            characters = json.load(open(characters_json_path, 'r'))
            character_names = list(characters.keys())
        else:
            assert 0, "Please generate the config first before known place annotation."
          
        with open(f"{ASSETS_PATH}/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/place_metadata.json", "r") as f:
            self.place_metadata = json.load(f)
        with open(f"{ASSETS_PATH}/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/building_metadata.json", "r") as f:
            self.building_metadata = json.load(f)
        
        all_known_places_set = set(self.place_metadata.keys())
        for idx, character_name in enumerate(character_names):

            living_place = characters[character_name]["living_place"]
            living_building = self.place_metadata[living_place]["building"]

            private_living_place = f"{character_name}'s room at {living_place}"
            private_living_place_dict = deepcopy(self.place_metadata[living_place])
            private_living_place_dict["name"] = private_living_place
            private_living_place_dict["location"][-1] = -4 * len(self.building_metadata[living_building]["places"]) - 4
            private_living_place_dict["bounding_box"] = self.building_metadata[living_building]["bounding_box"]

            self.place_metadata[private_living_place] = private_living_place_dict
            self.building_metadata[living_building]["places"].append(private_living_place_dict)
        # import pdb
        # pdb.set_trace()
        for group in groups:
            groups[group]["members"] = sorted(groups[group]["members"])
            all_known_places_set.add(groups[group]['place'])
        overlay_locations_desp_on_image(place_metadata=self.place_metadata, building_metadata=self.building_metadata, save_path=f"{ASSETS_PATH}/scenes/{middle_path}/agents_num_{len(list(characters.keys()))}/known_places.png", known_places=all_known_places_set, groups=groups)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_characters", "-n", type=int, default=15)
    parser.add_argument("--num_groups", "-ng", type=int, default=4)
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--event", type=str)
    parser.add_argument("--font_size", type=int, default=16)
    args = parser.parse_args()
    random.seed(hash(args.scene))

    if not args.event:
        middle_path = f"{args.scene}"
    else:
        middle_path = os.path.join(f"{args.scene}", "events", args.event)
    args.output_dir = f"{ASSETS_PATH}/scenes/{middle_path}/agents_num_{args.num_characters}"

    height_field_path = os.path.join(get_assets_dir(), f"ViCo/scene/v1/{args.scene}/height_field.npz")
    height_field = load_height_field(height_field_path)
    global_cam_parameters = json.load(open(f"{ASSETS_PATH}/scenes/{args.scene}/global_cam_parameters.json", 'r'))

    gen = CharacterGen(args)
    print(f"Starting to generate known places annotation for {args.scene}...")
    gen.execute()
    print(f"Known places annotation for {args.scene} is generated and saved in {args.output_dir}/known_places.png.")