import os
import traceback
from collections import defaultdict
from ...tools.asset_utils import ensure_asset

import argparse
import logging
from argparse import ArgumentParser
from PIL import Image
import genesis as gs
import pickle
import random
import imageio
import psutil
import copy
import time
import json
import os
import numpy as np
import trimesh

wall_window_bottom = 1.2
wall_window_top = 1.8
door_width = 0.8
door_height = 2.2
wall_gap = 0.01

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

def save_intrinsic_and_pose(cam, work_dir):
    intrinsic_K = opengl_projection_matrix_to_intrinsics(
        cam.node.camera.get_projection_matrix(), width=cam.res[0], height=cam.res[1]
    )

    T_OPENGL_TO_OPENCV = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    cam_pose = cam.node.matrix @ T_OPENGL_TO_OPENCV

    np.save(os.path.join(work_dir, "intrinsic_K.npy"), intrinsic_K)
    np.save(os.path.join(work_dir, "cam_pose.npy"), cam_pose)

def generate_mesh_obj_trimesh_with_uv(x_l, x_r, y_l, y_r, a, b, filename="floor.obj", rep=4, remove_region=None, along_axis='z'):
    # Generate grid points for vertices
    gx = np.linspace(x_l, x_r, a)
    gy = np.linspace(y_l, y_r, b)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_z = np.zeros_like(grid_x)

    # Create vertices array
    vertices = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

    # Generate faces indices
    faces = []
    for j in range(b - 1):
        for i in range(a - 1):
            # Indices of vertices in the current quad
            v1 = j * a + i
            v2 = j * a + (i + 1)
            v3 = (j + 1) * a + (i + 1)
            v4 = (j + 1) * a + i
            # Add two triangles for each quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Convert faces to numpy array for easier manipulation
    faces = np.array(faces)

    # Create UV coordinates
    uv_x = np.linspace(0, 1, a)
    uv_y = np.linspace(0, 1, b)
    uv_grid_x, uv_grid_y = np.meshgrid(uv_x, uv_y)
    uvs = np.vstack([uv_grid_x.flatten(), uv_grid_y.flatten()]).T

    if remove_region:
        a1, b1, a2, b2 = remove_region
        # Mask for vertices outside the removal region
        mask_x = (grid_x.flatten() < a1) | (grid_x.flatten() > a2)
        mask_y = (grid_y.flatten() < b1) | (grid_y.flatten() > b2)
        mask = mask_x | mask_y

        # Filter out vertices inside the removal region
        vertices = vertices[mask]
        uvs = uvs[mask]

        # Find the indices of the remaining vertices
        remaining_indices = np.where(mask)[0]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_indices)}

        # Filter and remap faces
        new_faces = []
        for face in faces:
            if all(idx in index_map for idx in face):
                new_faces.append([index_map[idx] for idx in face])
        faces = np.array(new_faces)

    # Create the mesh with vertices, faces, and uv coordinates
    if along_axis == 'z':
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif along_axis == 'y':
        vertices = vertices[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        vertices = vertices[:, [2, 1, 0]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Export to OBJ file
    mesh.export(filename)

def add_wall(env, x_l, x_r, y_l, y_r, z, height=3.5, remove_region=None, texture=None, id=0):
    if texture is None:
        texture = ensure_asset("objects/indoor_objects/566fc160-d286-4c4c-96ab-6359881e1a51/1.jpg")
    elif not texture.startswith('/'):
        texture = ensure_asset(texture)

    z_l, z_r = z, z + height
    length, width, height = x_r - x_l, y_r - y_l, z_r - z_l
    wall_path = ensure_asset('objects/indoor_objects/wall.obj')
    env.add_entity(
        type="structure",
        name="wall",
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=wall_path,
            scale=(length / 2.0, width / 2.0, height / 2.0),
            pos=(x_l + length / 2, y_l + width / 2, z_l + height / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=texture)
        ),
    )

def add_floor(env, x_l, x_r, y_l, y_r, z, texture=None, id=0):
    if texture is None:
        texture = ensure_asset("objects/indoor_objects/0da58c97-71df-479b-9f69-f084b76937f8/Black_marble.jpg")
    elif not texture.startswith('/'):
        texture = ensure_asset(texture)
    z_l, z_r = z - 0.2, z
    length, width, height = x_r - x_l, y_r - y_l, z_r - z_l
    wall_path = ensure_asset('objects/indoor_objects/wall.obj')
    env.add_entity(
        type="structure",
        name="floor",
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=wall_path,
            scale=(length / 2.0, width / 2.0, height / 2.0),
            pos=(x_l + length / 2, y_l + width / 2, z_l + height / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=texture)
            # color=(0.3, 0.3, 0.3),
        ),
    )

def add_ceiling(env, x_l, x_r, y_l, y_r, z, texture=None, id=0):
    if texture is None:
        texture = ensure_asset("objects/indoor_objects/28a9d2d5-2fa6-4c70-a46f-f6974547832e/1.jpg")
    elif not texture.startswith('/'):
        texture = ensure_asset(texture)
    z_l, z_r = z + 3.5, z + 3.7
    length, width, height = x_r - x_l, y_r - y_l, z_r - z_l
    wall_path = ensure_asset('objects/indoor_objects/wall.obj')
    env.add_entity(
        type="structure",
        name="ceiling",
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=wall_path,
            scale=(length / 2.0, width / 2.0, height / 2.0),
            pos=(x_l + length / 2, y_l + width / 2, z_l + height / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=texture)
        ),
    )

def add_room_camera(x_l, x_r, y_l, y_r, z_l=0, z_r=3.5):
    pose0 = ((x_l+0.1, y_l+0.1, z_r-0.35), (x_r-0.1, y_r-0.1, z_l+0.3))
    pose1 = ((x_l+0.1, y_r-0.1, z_r-0.35), (x_r-0.1, y_l+0.1, z_l+0.3))
    pose2 = ((x_r-0.1, y_l+0.1, z_r-0.35), (x_l+0.1, y_r-0.1, z_l+0.3))
    pose3 = ((x_r-0.1, y_r-0.1, z_r-0.35), (x_l+0.1, y_l+0.1, z_l+0.3))
    pose4 = (((x_l+x_r)/2, (y_l+y_r)/2, z_r), ((x_l+x_r)/2, (y_l+y_r)/2, z_l))
    return [pose0, pose1, pose2, pose3, pose4]

def load_indoor_scene(env, place, offset_x=0, offset_y=0, offset_z=0, size_x=30, size_y=30, no_objects=False):
    room_center = [size_x/2, size_y/2]

    mat_rigid = gs.materials.Rigid()

    room_camera_poses = []
    for room in place:
        offset = [room['left']+offset_x, room['top']+offset_y, offset_z]
        size = [room['width'], room['height']]
        center = [offset[0]+size[0]/2, offset[1]+size[1]/2]
        floor_texture = None
        wall_texture = 'objects/indoor_objects/Porcelain_White_Mat.png'
        add_wall(env, offset[0] - 0.2, offset[0], offset[1], offset[1]+size[1], offset[2], texture=wall_texture)
        add_wall(env, offset[0]+size[0], offset[0]+size[0]+0.2, offset[1], offset[1]+size[1], offset[2], texture=wall_texture)
        add_wall(env, offset[0], offset[0]+size[0], offset[1]-0.2, offset[1], offset[2], texture=wall_texture)
        add_wall(env, offset[0], offset[0]+size[0], offset[1]+size[1], offset[1]+size[1]+0.2, offset[2], texture=wall_texture)
        add_floor(env, offset[0], offset[0]+size[0], offset[1], offset[1]+size[1], offset[2], texture=floor_texture)
        add_ceiling(env, offset[0], offset[0]+size[0], offset[1], offset[1]+size[1], offset[2], texture=wall_texture)
        
        room_camera_poses.append((room['name'], add_room_camera(offset[0], offset[0]+size[0], offset[1], offset[1]+size[1], offset[2], offset[2]+3.5)))

        if not no_objects:
            for obj in room['objects']:
                path = obj['filename']
                if not path.startswith('/'):
                    if "grocery_assets" in path:
                        path = path.replace("grocery_assets/", "", 1)
                    path = ensure_asset(f'objects/indoor_objects/{path}')
                scale = obj['scale']
                pos = obj['pos']
                pos[0] += offset[0]
                pos[1] += offset[1]
                pos[2] += offset[2]
                euler = obj['euler']
                if type(scale) is list:
                    scale = tuple(scale)
                if path.endswith('.urdf'):
                    env.add_entity(
                        type="object",
                        name=obj["name"],
                        material=mat_rigid,
                        morph=gs.morphs.URDF(
                            file=path,
                            scale=scale,
                            pos=(pos[0], pos[1], pos[2]),
                            euler=(euler[0], euler[1], euler[2]),
                            collision=False,
                        ),
                    )
                else:
                    try:
                        env.add_entity(
                            type="object",
                            name=obj["name"],
                            material=mat_rigid,
                            morph=gs.morphs.Mesh(
                                file=path,
                                scale=scale,
                                pos=(pos[0], pos[1], pos[2]),
                                euler=(euler[0], euler[1], euler[2]),
                                fixed=True,
                                collision=False
                            ),
                        )
                    except Exception as e:
                        print(f"Failed to load {path} with error {e} with traceback {traceback.format_exc()}")

    return room_camera_poses