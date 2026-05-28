import json

import cv2
import numpy as np
import genesis as gs
import os
from tqdm import tqdm

from vico.modules.avatar.avatar_controller import AvatarController
from vico.modules.vehicle.vehicle_controller import VehicleController
from vico.tools.constants import LIGHTS, ASSETS_PATH

class Env:
    def __init__(self, scene):
        self.avatars = []
        self.scene = scene
        
    def add_avatar(self, name, motion_data_path, skin_options, ego_view_options, frame_ratio, enable_collision):
        avatar = AvatarController(
            env=self, 
            name=name, 
            motion_data_path=motion_data_path, 
            skin_options=skin_options, 
            ego_view_options=ego_view_options, 
            frame_ratio=frame_ratio, 
            enable_collision=enable_collision
        )
        self.avatars.append(avatar)
        return avatar
    
    def add_entity(self, type, name, morph, material=None, surface=None, visualize_contact=False, vis_mode=None):
        entity = self.scene.add_entity(morph=morph, material=material, surface=surface, visualize_contact=visualize_contact, vis_mode=vis_mode)
        return entity


def main():
    char2skin = json.load(open(f"{ASSETS_PATH}/character2skin.json", "r"))
    img_save_dir = f"{ASSETS_PATH}/imgs"
    os.makedirs(img_save_dir, exist_ok=True)
    skin_not_working_characters = {}
    
    gs.init(logging_level='info', backend=gs.gpu)

    p_trans = np.array([0.0, 0.0, 0.0])
    camera_poses = [
        [0.0, 1.2, 0.95],  
        [0.0, -1.2, 0.95],  
        [1.2, 0.0, 0.95],
        [-1.2, 0.0, 0.95],
    ]
    camera_lookat = [
        [0.0, -1.0, 0.9],  
        [0.0, 1.0, 0.9],  
        [-1.0, 0.0, 0.9],
        [1.0, 0.0, 0.9],
    ]

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(512, 512),
            camera_pos=np.array([62.0, 71.0, -27.0]),
            camera_lookat=np.array([0.0, 0.0, 0.0]),
            camera_fov=40,
        ),
        show_viewer=False,
        sim_options=gs.options.SimOptions(),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
            enable_collision=False,
        ),
        avatar_options=gs.options.AvatarOptions(
            enable_collision=False,
        ),
        renderer=gs.renderers.Rasterizer(),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            segmentation_level="entity",
            lights=LIGHTS
        ),
    )

    cam_0 = scene.add_camera(
        res=(1920, 1920),
        pos=(1.2, 0.0, 0.95),
        lookat=(-1.0, 0.0, 0.9),
        fov=90,
        GUI=False,
        far=1600.0,
    )

    scene = Env(scene)
    avatars = {}
    for char_name, skin_dict in tqdm(char2skin.items(), desc="Loading characters"):
        try:
            glb_path = f'avatars/models/{skin_dict["skin_file"]}'
            humanoid = scene.add_avatar(
                name=char_name,
                motion_data_path="avatars/motions/motion.pkl",
                skin_options={
                    'glb_path': glb_path,
                    'euler': (-90, 0, 90),
                    'pos': (0, 0, -0.959008030), 
                },
                ego_view_options={
                    "res": (512, 512),
                    "fov": 90,
                    "GUI": False,
                    "far": 16000.0,
                },
                frame_ratio=1.0,
                enable_collision=False,
            )
            avatars[char_name] = humanoid
        except Exception as e:
            print(f"Error loading {char_name}: {e}")
            skin_not_working_characters[char_name] = e

    scene.scene.build()
    scene.scene.reset()

    ########################## take photos ##########################
    for char_name in tqdm(avatars.keys(), desc="Taking photos"):
        save_check_path = os.path.join(img_save_dir, f"{char_name}_0_rgb.png")
        if os.path.exists(save_check_path):
            print(f"Skipped {char_name} (already exists).")
            continue

        print(f"Processing character: {char_name}")

        for other_name, avatar in avatars.items():
            if other_name == char_name:
                avatar.reset(np.array([0.0, 0.0, 0.0]))
            else:
                avatar.reset(np.array([1e5, 1e5, 1e5]))

        scene.scene.step()

        for idx, (pos, lookat) in enumerate(zip(camera_poses, camera_lookat)):
            cam_0.set_pose(pos=pos, lookat=lookat)
            rgb, depth, _, _ = cam_0.render(depth=True)

            depth_threshold = 10.0
            mask = depth <= depth_threshold
            rgb = rgb.copy()
            rgb[~mask] = 0
            rgba = rgb[:, :, [2, 1, 0]]

            relative_img_save_path = os.path.join(img_save_dir, f"{char_name}_{idx}_rgb.png")
            success = cv2.imwrite(relative_img_save_path, rgba)
            if success:
                print(f"Saved {relative_img_save_path}")
            else:
                print(f"Failed to save {relative_img_save_path}")

    gs.destroy()

    ########################## error report ##########################
    if skin_not_working_characters:
        print("Some characters failed to load:", skin_not_working_characters)


if __name__ == '__main__':
    main()