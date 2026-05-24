import os
import re
from collections import defaultdict

import cv2
import json
import textwrap
import argparse
import subprocess
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
from moviepy import TextClip, ImageClip, CompositeVideoClip, concatenate_videoclips

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from vico.tools.utils import *

def render_topdown_locators(image, locator_positions, colors, circle_radii, camera_parameters):
    f_x = camera_parameters["camera_res"][0] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    f_y = camera_parameters["camera_res"][1] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    intrinsic_K = np.array([[f_x, 0.0, camera_parameters["camera_res"][0]/2.0],
                            [0.0, f_y, camera_parameters["camera_res"][1]/2.0],
                            [0.0, 0.0, 1.0]])
    extrinsic = np.array(camera_parameters["camera_extrinsics"])
    extrinsic = extrinsic[:3, :4]
    for pos, color, radius in zip(locator_positions, colors, circle_radii):
        P_world = np.append(pos, 1.0)
        P_camera = extrinsic @ P_world
        P_image = intrinsic_K @ P_camera
        pixel_x = int(P_image[0] / P_image[2])
        pixel_y = int(P_image[1] / P_image[2])
        cv2.circle(image, (pixel_x, pixel_y), radius, (color[2]*255, color[1]*255, color[0]*255), -1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)\.(?:png|jpe?g)$', filename)
    return int(match.group(1)) if match else -1

def images_to_video_using_ffmpeg(input_dir_path, output_path, fps=30, threads=8, codec="mpeg4"):
    images = sorted([img for img in os.listdir(input_dir_path) if img.endswith(('.png', '.jpg', '.jpeg'))], key=extract_frame_number)
    if not images:
        raise ValueError(f"No images in {input_dir_path}")
    with open("images_list.txt", "w") as f:
        for image in images:
            f.write(f"file '{os.path.join(input_dir_path, image)}'\n")
    if codec == "mpeg4":
        ffmpeg_cmd_generate = [
            "ffmpeg", 
            "-f", "concat",
            "-safe", "0",
            "-r", str(fps),
            "-i", "images_list.txt",
            "-vcodec", "mpeg4",
            "-threads", str(threads),
            output_path
        ]
    elif codec == "h264":
        ffmpeg_cmd_generate = [
            "ffmpeg", 
            "-f", "concat",
            "-safe", "0",
            "-r", str(fps),
            "-i", "images_list.txt",
            "-vcodec", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-threads", str(threads),
            output_path
        ]
    else:
        print(f"Codec {codec} not supported.")
        exit()
    subprocess.run(ffmpeg_cmd_generate, check=True)
    os.remove("images_list.txt")

def rgb_to_bgr255(color):
    rgb = mcolors.to_rgb(color)
    # gbr = (rgb[2]*255, rgb[1]*255, rgb[0]*255)
    rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return rgb

def add_colored_dot(frame, position, radius, color):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    x, y = position
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=color, outline=color)
    return frame

def process_frame_agents(frame_idx, args, names_order, name_to_color, agent_cam_images, global_images, demo_folder, last_text_actions):
    
    frame_image_path = os.path.join(demo_folder, f"frame_{frame_idx:06}.png")
    if not args.overwrite and os.path.exists(frame_image_path):
        return ImageClip(frame_image_path, duration=1/args.fps)
    selected_agents = selected_initial_agents.copy()
    none_action_agents = [name for name in selected_agents if json.load(open(os.path.join(args.output_dir, 'steps', name, f'{frame_idx:06d}.json')))["action"] is None]
    action_agents = [name for name in names_order if json.load(open(os.path.join(args.output_dir, 'steps', name, f'{frame_idx:06d}.json')))["action"] is not None and name not in selected_agents]
    
    for none_agent in none_action_agents:
        if len(action_agents) == 0:
            break
        if none_agent in selected_agents:
            selected_agents[selected_agents.index(none_agent)] = action_agents.pop(0)
    
    wrapped_text_upper = {}
    wrapped_text_bottom = {}
    current_time = None
    
    for name in selected_agents:
        steps_data_name_frame_idx = json.load(open(os.path.join(args.output_dir, 'steps', name, f'{frame_idx:06d}.json')))
        current_schedule = str(steps_data_name_frame_idx["action_desp"]) if "action_desp" in steps_data_name_frame_idx else None
        current_action = steps_data_name_frame_idx["action"]["type"] if "action" in steps_data_name_frame_idx and steps_data_name_frame_idx["action"] is not None else "None"
        current_place = str(steps_data_name_frame_idx["obs"]["current_place"]) if "current_place" in steps_data_name_frame_idx["obs"] and steps_data_name_frame_idx["obs"]["current_place"] is not None else "Open Space"
        current_time = steps_data_name_frame_idx["curr_time"]
        
        if current_schedule:
            last_text_actions[name] = current_schedule
        else:
            current_schedule = last_text_actions[name] if last_text_actions[name] else "None"
        
        wrapped_text_upper[name] = textwrap.fill(current_schedule, width=45)
        current_action = name + ' ' + current_action + " at " + current_place
        wrapped_text_bottom[name] = textwrap.fill(current_action, width=45)
    
    dynamic_clips = [
        ImageClip(global_images[frame_idx]).with_position((930, 600)).with_duration(1/args.fps).resized(width=700)
    ]

    avatar_positions = []

    for i in range(0, 12):
        avatar_positions.append([10 + i * 222, 60])
    for i in range(0, 13):
        avatar_positions.append([10 + i * 204, 1760])
    
    text_clips = []
    for i, name in enumerate(names_order):
        pos = avatar_positions[i]
        dynamic_clips.append(ImageClip(avatar_images[name]).with_position((pos[0], pos[1])).with_duration(1/args.fps).resized(width=130))
        pos = avatar_positions[i]
        text_clips.append(
            TextClip(font=os.path.join(os.path.dirname(__file__), "misc/OpenSans-Regular.ttf"), text=name, font_size=20, color=(name_color_bgr_255[name][0], name_color_bgr_255[name][1], name_color_bgr_255[name][2]))
            .with_position((pos[0] - 10, pos[1] - 50))
            .with_duration(1/args.fps)
        )

    
    selected_agents_positions = [
        (50, 300), (50, 780), (50, 1260),
        (500, 300), (500, 780), (500, 1260),
        (1700, 300), (1700, 780), (1700, 1260),
        (2150, 300), (2150, 780), (2150, 1260),
    ]
    
    for i, name in enumerate(selected_agents):
        pos = selected_agents_positions[i]
        dynamic_clips.append(ImageClip(agent_cam_images[name][frame_idx]).with_position((pos[0], pos[1])).with_duration(1/args.fps).resized(width=350))
        # text_clips.append(
        #     TextClip(font=os.path.join(os.path.dirname(__file__), "misc/OpenSans-Regular.ttf"), text=wrapped_text_upper[name], font_size=15, color=(255, 255, 255))
        #     .with_position((pos[0], pos[1] - 50))
        #     .with_duration(1/args.fps)
        # )
        text_clips.append(
            TextClip(font=os.path.join(os.path.dirname(__file__), "misc/OpenSans-Regular.ttf"), text=wrapped_text_bottom[name], font_size=15, color=(255, 255, 255))
            .with_position((pos[0], pos[1] + 350))
            .with_duration(1/args.fps)
        )
    
    text_clips.append(
        TextClip(font=os.path.join(os.path.dirname(__file__), "misc/OpenSans-Regular.ttf"), text="Time: " + str(current_time), font_size=50, color=(255, 255, 255))
        .with_position((890, 400))
        .with_duration(1/args.fps)
    )
    
    frame_clip = CompositeVideoClip(dynamic_clips + text_clips, size=(2560, 1920))
    frame = frame_clip.get_frame(0)
    
    for i, name in enumerate(names_order):
        pos = avatar_positions[i]
        this_color = (int(name_color_bgr_255[name][0]), int(name_color_bgr_255[name][1]), int(name_color_bgr_255[name][2]))
        frame = add_colored_dot(frame, (pos[0] + 60, pos[1] - 15), 10, this_color)
    
    frame.save(frame_image_path)
    
    return ImageClip(frame_image_path, duration=1/args.fps)

def make_global_image(i, args, names_order, camera_parameters):
    global_img_path = os.path.join(args.output_dir, 'global', f'rgb_{i:06d}.png')
    global_images[i] = (global_img_path)
    if os.path.exists(global_img_path) and not args.overwrite:
        return True
    agent_poses = []
    for name in names_order:
        step_data = json.load(open(os.path.join(args.output_dir, 'steps', name, f'{i:06d}.json')))
        agent_poses.append(step_data["obs"]["pose"])
    global_image_copy = global_image.copy()
    global_image_with_agents = render_topdown_locators(global_image_copy, [np.array(agent_pose[:3]) for agent_pose in agent_poses], agent_locator_colors, circle_radii=[15 for _ in agent_poses], camera_parameters=camera_parameters)
    Image.fromarray(global_image_with_agents).save(global_img_path)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default='output')
    parser.add_argument("--scene", type=str, default='NY')
    parser.add_argument("--config", type=str, default='agents_num_15')
    parser.add_argument("--agent_type", type=str, choices=['tour_agent'], default='tour_agent')
    parser.add_argument("--data_dir", "-d", type=str)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--no_output_video", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cam_type", choices=['ego', 'tp'], default='ego')
    parser.add_argument("--videowriter", choices=['default', 'ffmpeg'], default='default')
    parser.add_argument("--codec", choices=['mpeg4', 'h264'], default='mpeg4') # For server, use mpeg4. For local, use h264, which is better.
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.rstrip('/')
        args.agent_type = args.data_dir.split('/')[-1]
        args.scene = args.data_dir.split('/')[-2].split('_')[0]
        args.output_dir = args.data_dir
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")
    demo_folder = os.path.join(args.output_dir, 'demo')
    os.makedirs(demo_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'global'), exist_ok=True)

    config_path = os.path.join(args.output_dir, 'curr_sim', "config.json")
    with open(config_path, 'r') as file:
        config = json.load(file)

    names_order = config["agent_names"] # by default, can change later
    num_agents = config["num_agents"]
    name_to_color = {}
    last_text_actions = {}
    last_goal_locations = {}
    for agent_name, agent_skin, locator_color in zip(config["agent_names"], config["agent_skins"], config["locator_colors_rgb"]):
        name_to_color[agent_name] = locator_color
        last_text_actions[agent_name] = None
        last_goal_locations[agent_name] = None

    num_steps = config["step"]
    if args.steps:
        num_steps = min(num_steps, args.steps)
    duration = num_steps / args.fps

    avatar_images = {}
    agent_cam_images = defaultdict(dict)

    global_images = {}

    global_image_path = os.path.join('ViCo', 'assets', 'scenes', args.scene, "global.png")
    global_image = cv2.imread(global_image_path)

    camera_parameters = json.load(open(os.path.join('ViCo', 'assets', 'scenes', args.scene, "global_cam_parameters.json")))
    agent_locator_colors = map_lang_colors_to_rgb(config["locator_colors"])

    for name in names_order:
        # import pdb; pdb.set_trace()
        for frame_idx in range(num_steps):
            agent_cam_image_path = os.path.join(args.output_dir, args.cam_type, name, f'rgb_{frame_idx:06d}.png')
            assert os.path.exists(agent_cam_image_path), f"Image {agent_cam_image_path} does not exist."
            agent_cam_images[name][frame_idx] = agent_cam_image_path
        # agent_cam_images[name] = sorted(agent_cam_images[name])
        avatar_images[name] = os.path.join('ViCo', 'assets', 'imgs', 'avatars', f'{name}.png')
        # if not os.path.exists(avatar_images[name]):
        #     avatar_images[name] = os.path.join('ViCo', 'assets', 'imgs', 'default.png')
        # avatar_images[name] = imageio.imread(avatar_images[name]).resized((512, 512))

    if num_agents >= 12:
        selected_initial_agents = names_order[:12]
    else:
        selected_initial_agents = names_order[:num_agents]

    name_color_bgr_255 = {}
    for name in names_order:
        name_color = name_to_color[name]
        name_color_bgr_255[name] = rgb_to_bgr255(name_color)

    # for i in tqdm.tqdm(range(0, num_steps, args.fps)):
    #     global_img_path = os.path.join(args.output_dir, 'global', f'rgb_{i:06d}.png')
    #     global_images[i] = (global_img_path)
    #     if os.path.exists(global_img_path) and not args.overwrite:
    #         continue
    #     agent_poses = []
    #     for name in names_order:
    #         step_data = json.load(open(os.path.join(args.output_dir, 'steps', name, f'{i:06d}.json')))
    #         agent_poses.append(step_data["obs"]["pose"])
    #     global_image_copy = global_image.copy()
    #     global_image_with_agents = render_topdown_locators(global_image_copy, [np.array(agent_pose[:3]) for agent_pose in agent_poses], agent_locator_colors, circle_radii=[15 for _ in agent_poses], camera_parameters=camera_parameters)
    #     Image.fromarray(global_image_with_agents).save(global_img_path)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        global_image_generated = partial(make_global_image, args=args, names_order=names_order, camera_parameters=camera_parameters)
        tqdm.tqdm(executor.map(global_image_generated, range(0, num_steps, args.fps)), total=num_steps//args.fps)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        process_frame_partial = partial(process_frame_agents, args=args, names_order=names_order, name_to_color=name_to_color, agent_cam_images=agent_cam_images, global_images=global_images, demo_folder=demo_folder, last_text_actions=last_text_actions)
        clips = list(tqdm.tqdm(executor.map(process_frame_partial, range(0, num_steps, args.fps)), total=num_steps//args.fps))

    clips = [clip for clip in clips if clip is not None]
    if not args.no_output_video:
        if args.videowriter == "ffmpeg":
            images_to_video_using_ffmpeg(demo_folder, os.path.join(demo_folder, "demo.mp4"), fps=args.fps, threads=args.threads, codec=args.codec)
        else:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(os.path.join(demo_folder, "demo.mp4"), fps=args.fps)