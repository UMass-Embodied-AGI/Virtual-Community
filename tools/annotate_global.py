import os
import sys
from multiprocessing import Pool
import io
from collections import defaultdict
import cv2
import json
import random
import argparse
import colorsys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import copy

import genesis as gs

current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from tools.constants import google_map_coarse_to_types, ASSETS_PATH
from tools.utils import *

# wonderful_colors = [
#     ("#F59E9E", "#130C2F"),
#     ("#B4F59D", "#601717"),
#     ("#7BFFF6", "#120A53"),
#     ("#191251", "#59FFD3"),
#     ("#52124F", "#F1FF59"),
#     ("#470E22", "#9CFF4C")
# ]

import matplotlib.colors as mcolors
wonderful_colors = [
    ("#FFA6A6"),
    ("#FFD782"),
    ("#C3FF82"),
    ("#82FFCF"),
    ("#9DBFFF"),
    ("#E29BF8")
]

def stitch_images_horizontally(image1, image2):
    combined_width = image1.width + image2.width
    combined_height = image1.height
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    return combined_image

def overlay_locations_desp_on_image():
    global_cam_parameters = json.load(open(os.path.join(ASSETS_PATH, "scenes", args.scene, "global_cam_parameters.json"), 'r'))
    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    coarse_types = list(google_map_coarse_to_types.keys())
    type_to_color = dict(zip(coarse_types, wonderful_colors[:len(coarse_types)]))
    coarse_types.append("transit")
    type_to_color["transit"] = "white"
    font_size = 14
    padding=2
    font=ImageFont.truetype("assets/arial.ttf", font_size)
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
            pixel_x, pixel_y = project_3d_to_2d_from_perspective_camera(np.array([x, y, get_height_at(height_field, x, y)]), np.array(global_cam_parameters["camera_res"]), np.array(global_cam_parameters["camera_fov"]), np.array(global_cam_parameters["camera_extrinsics"]))
            indicator_text = place_indicators[(building, place_name)]
            ctype = place["coarse_type"]
            color = type_to_color[ctype]

            # if ctype == "transit":
            #     print(f"Transit place {place_name} in building {building} at ({x}, {y}, {get_height_at(height_field, x, y)})")

            text_width, text_height = draw.textbbox((0, 0), indicator_text, font=font)[2:]
            if building != "open space":
                px, py = pixel_x, pixel_y + i * (font_size + padding * 2 + 1)
            else:
                px, py = pixel_x, pixel_y
            box_x1 = px - padding
            box_y1 = py - padding
            box_x2 = px + text_width + padding
            box_y2 = py + text_height + padding 
            draw.rounded_rectangle(
                [(box_x1, box_y1), (box_x2, box_y2)],
                radius=5,
                fill=color,
                outline="black",
                width=1
            )
            draw.text((px, py), indicator_text, font=font, fill="black")

    legend_image = generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter)
    annotated_image = stitch_images_horizontally(annotated_image, legend_image)
    annotated_image.save(os.path.join(ASSETS_PATH, "scenes", args.scene, "global_annotated.png"))

def generate_places_legend(places_by_coarse_type, type_to_color, coarse_type_to_letter):
    sorted_ctypes = sorted(places_by_coarse_type.keys())
    col_height = 26
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, col_height))
    ax_left, ax_right = axes[0], axes[1]
    current_y_left = 0
    current_y_right = 0

    def draw_entry(ax, start_y, pname, i):
        indicator = f"{letter}{i+1}"
        color = type_to_color[ctype]
        ax.add_patch(plt.Rectangle((0, start_y), 0.15, 0.15, facecolor=mcolors.to_rgb(color), edgecolor="black", linewidth=1))
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

def get_building_to_places():

    with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "inaccessible_buildings.json"), 'r') as f:
        inaccessible_buildings = json.load(f)

    with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "raw", "building_to_places.json"), 'r') as f:
        building_to_places = json.load(f)

    with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "place_metadata.json"), 'r') as f:
        place_metadata = json.load(f)

    with open(os.path.join(ASSETS_PATH, "scenes", args.scene, "building_metadata.json"), 'r') as f:
        building_metadata = json.load(f)

    return place_metadata, building_metadata, inaccessible_buildings, building_to_places


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("args:", args)
    random.seed(args.seed)
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

    final_folder_mapping = {"newyork": "NY", "elpaso": "EL_PASO_ok"}
    if args.scene not in final_folder_mapping:
        final_folder_mapping[args.scene] = args.scene.upper()
    height_field_path=os.path.join(get_assets_dir(), f"ViCo/scene/v1/{final_folder_mapping[args.scene]}/height_field.npz")
    
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

    height_field = load_height_field(height_field_path)

    image_path = f"assets/scenes/{args.scene}/global.png"
    if not os.path.exists(image_path):
        white_global_image = Image.new("RGB", (2000, 2000), "white")
        white_global_image.save(image_path)
    with open(f"assets/scenes/{args.scene}/raw/places_full.json", 'r') as file:
        places_dict = json.load(file)
    with open(f"assets/scenes/{args.scene}/raw/building_to_osm_tags.json", 'r') as file:
        building_to_osm_tags = json.load(file)
    place_metadata, building_metadata, inaccessible_buildings, building_to_places = get_building_to_places()
    overlay_locations_desp_on_image()