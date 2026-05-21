#!/usr/bin/env python3
"""
Scene Renderer from Config
==========================

This script loads a scene from config.json and renders it from agent poses using the Rasterizer.
Based on the configuration, it:
1. Loads the scene from the specified GLB file
2. Creates cameras at agent poses
3. Renders the scene from each agent's perspective
4. Saves the rendered images
"""

import json
import os
import argparse
import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
import glob
from tools.utils import get_assets_dir
from utils import load_height_field, get_height_at
# from modules.avatar.avatar_robot import AvatarRobot


def load_config(config_path):
    """Load and parse the configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_scene(config):
    """Initialize the Genesis scene with Rasterizer renderer."""
    # Initialize Genesis
    gs.init(seed=0, precision="32", logging_level="info")
    
    buildings = {}

    # Get scene path from config
    scene_path = config.get("scene", "")
    scene_offset = config.get("scene_offset", [0.0, 0.0, 0.0])
    
    # Create scene with Rasterizer
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(0.0, 0.0, 5.0),  # Default camera position
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=50,
        ),
        show_viewer=False,  # Headless rendering
        rigid_options=gs.options.RigidOptions(
            enable_collision=False, 
            gravity=(0, 0, 0)
        ),
        renderer=gs.renderers.Rasterizer()  # Use Rasterizer as requested
    )
    
    # Add the scene mesh with offset
    if scene_path:
        # print(f"Loading scene: {scene_path}")
        # scene.add_entity(
        #     morph=gs.morphs.Mesh(
        #         file=scene_path,
        #         pos=scene_offset,
        #         euler=(90.0, 0, 0),
        #         scale=1.0,
        #         fixed=True,
        #         collision=False,
        #     ),
        #     surface=gs.surfaces.Default()
        # )
        scene.add_entity(
            material=gs.materials.Rigid(
				sdf_min_res=4,
				sdf_max_res=4,
			),
            morph=gs.morphs.Mesh(
                file=os.path.join(scene_path,"terrain.glb"),
                pos=scene_offset,
                euler=(90.0, 0, 0),   # keep same orientation you used for terrain
                scale=1.0,
                fixed=True,
                collision=False,
				merge_submeshes_for_collision=False,
				group_by_material=True,
            ),
            surface=gs.surfaces.Default()
        )

        paths = [scene_path]
        if os.path.isdir(scene_path):
            paths = sorted(glob.glob(os.path.join(scene_path, "**/*.glb"), recursive=True))

        for p in paths:
            if 'terrain.glb' not in p:
                print(f"Loading scene mesh: {p}")
                scene.add_entity(
                    material=gs.materials.Rigid(
								sdf_min_res=4,
								sdf_max_res=4,
					),
                    morph=gs.morphs.Mesh(
                        file=p,
                        pos=scene_offset,
                        euler=(90.0, 0, 0),
                        fixed=True,
                        collision=True,
                        group_by_material=True,
                        decompose_object_error_threshold=0.15,
                    ),
                    surface=gs.surfaces.Default()
                )
    else:
        print("Warning: No scene path specified in config")
    
    return scene


def add_agent_skins(scene, config, hf):
    """Add agent skin entities at their poses."""
    agent_poses = config.get("agent_poses", [])
    agent_skins = config.get("agent_skins", [])
    agent_names = config.get("agent_names", [])
    
    if not agent_skins or not agent_poses:
        print("Warning: No agent skins or poses found in config")
        return []
    
    avatar_entities = []
    
    for i, (pose, skin_path) in enumerate(zip(agent_poses, agent_skins)):
        if len(pose) >= 6:  # Need at least position and rotation
            # Extract position and rotation from pose
            pos = np.array(pose[:3], dtype=np.float64)
            euler_angles = np.array(pose[3:6], dtype=np.float64)
            
            # Convert euler angles to rotation matrix
            # rot_matrix = geom_utils.euler_to_R(np.degrees(euler_angles))
            
            z_ground = float(get_height_at(hf, pos[0], pos[1]))
            pos[2] = z_ground + 1.2      # ~1.2 m keeps feet above ground

            # keep the pose in config in sync so cameras use the same z
            config["agent_poses"][i][2] = float(pos[2])

            # Get agent name or use default
            agent_name = agent_names[i] if i < len(agent_names) else f"agent_{i}"
            
            print(f"Adding agent skin {i} ({agent_name}) at position {pos} with rotation {euler_angles}")
            
            # Create avatar material
            mat_avatar = gs.materials.Avatar()

            # Add the agent skin entity
            avatar_entity = scene.add_entity(
                # name=agent_name,
                material=mat_avatar,
                morph=gs.morphs.Mesh(
                    file=skin_path,
                    pos=pos,
                    euler=euler_angles,  # Default avatar orientation
                    decimate=False,
                    convexify=False,
                    collision=False,
                    group_by_material=False,
                )
            )
            avatar_entities.append(avatar_entity)
        else:
            print(f"Warning: Agent pose {i} has insufficient data (need 6 values: x,y,z,rx,ry,rz)")
    
    return avatar_entities


def create_cameras_from_agent_poses(scene, config):
    """Create cameras positioned at agent poses."""
    agent_poses = config.get("agent_poses", [])
    camera_configs = config.get("camera_config", [])
    
    cameras = []
    # Use camera config if available, otherwise use agent pose
    # camera_pos = camera_configs[0].get("pos", [])
    camera_pos = [agent_poses[0][0]-3, agent_poses[0][1], agent_poses[0][2] + 0.9]
    camera_lookat = [agent_poses[0][0], agent_poses[0][1], agent_poses[0][2]]
            
    print(f"Creating camera {0} at position {camera_pos}, looking at {camera_lookat}")
            
    # Add camera to scene
    camera = scene.add_camera(
        res=(1024, 1024),
        pos=camera_pos,
        lookat=camera_lookat,
        fov=90,
        far=16000.0,
        GUI=False,
    )
    cameras.append(camera)
    
    return cameras


def render_scene(scene, cameras, output_dir="rendered_output"):
    """Render the scene from all camera positions and save images."""
    # Build the scene
    scene.build()
    scene.reset()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering scene from {len(cameras)} camera positions...")
    
    for i, camera in enumerate(cameras):
        print(f"Rendering from camera {i}...")
        
        # Render the scene
        rgb_img, depth_img, seg_img, normal_img = camera.render(
            rgb=True, 
            depth=False, 
            segmentation=False, 
            normal=False
        )
        
        if rgb_img is not None:
            # Save RGB image
            output_path = os.path.join(output_dir, f"camera_{i:03d}_rgb.png")
            gs.tools.save_img_arr(rgb_img, output_path)
            print(f"Saved RGB image: {output_path}")
        else:
            print(f"Warning: Failed to render RGB image for camera {i}")
    
    print(f"Rendering complete! Images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Render scene from config.json using agent poses")
    parser.add_argument("--config", "-c", default="examples/rendering/config.json", 
                       help="Path to config.json file")
    parser.add_argument("--output", "-o", default="rendered_output", 
                       help="Output directory for rendered images")
    parser.add_argument("--vis", action="store_true", 
                       help="Show viewer (for debugging)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        hf = load_height_field(os.path.join(get_assets_dir(), "ViCo/scene/v1/NY/height_field.npz"))

        # Setup scene
        print("Setting up scene...")
        scene = setup_scene(config)
        
        # Add agent skins at their poses
        print("Adding agent skins at their poses...")
        avatar_entities = add_agent_skins(scene, config, hf)
        
        # Create cameras from agent poses
        print("Creating cameras from agent poses...")
        cameras = create_cameras_from_agent_poses(scene, config)
        
        if not cameras:
            print("Error: No valid cameras created. Check agent poses in config.")
            return
        
        # Render scene
        render_scene(scene, cameras, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        gs.destroy()


if __name__ == "__main__":
    main()
