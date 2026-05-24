from typing import Tuple, Union

import numpy as np
import pickle
import ctypes
import time
from dataclasses import dataclass
from ....tools.utils import atomic_save
from .builtin import lib_builder

def convex_hull(points: np.ndarray) -> np.ndarray:
    size = np.zeros(1, dtype=np.int32)
    hull = lib_builder.convex_hull(points.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   points.shape[0],
                                   size.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    return np.ctypeslib.as_array(hull, (size[0], 2))

def dist_to_hull(point: np.ndarray, hull: np.ndarray) -> float:
    return lib_builder.dist_to_hull(point.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                    hull.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                    hull.shape[0])

@dataclass
class VolumeGridBuilderConfig:
    voxel_size: float = 0.1
    depth_bound: float = 30.0
    nav_grid_size: float = 0.5
    thread_num: int = 1

class VolumeGridBuilder:

    def __init__(self, conf = VolumeGridBuilderConfig()):
        self.conf = conf
        self.vg_backend = lib_builder.init_volume_grid(conf.voxel_size, conf.thread_num)
    
    @staticmethod
    def _img_to_pcd(rgb: np.ndarray, depth: np.ndarray, label: np.ndarray, fov: float, camera_ext: np.ndarray):
        size = (label >= -1).sum()
        points = np.zeros((size, 3), dtype=np.float32)
        colors = np.zeros((size, 3), dtype=np.uint8)
        lb = np.zeros(size, dtype=np.int32)
        lib_builder.image_to_pcd(np.ascontiguousarray(rgb).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                     depth.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     label.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     colors.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                     lb.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     rgb.shape[1], rgb.shape[0], fov,
                                     camera_ext.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return points, colors, lb

    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, label: np.ndarray, fov: float, camera_ext: np.ndarray):
        label[(depth < 0) | (depth > self.conf.depth_bound)] = -100 # remove invalid depth
        lib_builder.volume_grid_add_frame(self.vg_backend,
                                          np.ascontiguousarray(rgb).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                          depth.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          label.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                          rgb.shape[1], rgb.shape[0], fov,
                                          camera_ext.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def align(self, x):
        return np.floor(x / self.conf.voxel_size)

    def align_nav(self, x):
        return np.floor(x / self.conf.nav_grid_size)

    def add_points(self, points: np.ndarray, colors: np.ndarray, label: np.ndarray):
        lib_builder.volume_grid_insert_from_numpy(self.vg_backend,
                                              points.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                              colors.astype(np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                              label.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                              points.shape[0])
    
    def navigate(self, start: np.ndarray, goal: np.ndarray, ref_path: Union[np.ndarray, None] = None):
        """
        start: [2]
        goal: [H, 2] a convex hull
        ref_path: [R, 2] a tip path(optional)
        """
        size = np.zeros(1, dtype=np.int32)
        ref_path_c = None
        if ref_path is not None:
            ref_path_c = ref_path.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = lib_builder.navigate(self.vg_backend,
                                start.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                goal.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                goal.shape[0],
                                int(self.conf.nav_grid_size / self.conf.voxel_size),
                                size.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                ref_path_c,
                                ref_path.shape[0] if ref_path is not None else 0)
        if size[0] == 0:
            return None
        return np.ctypeslib.as_array(ret, (size[0], 2))
    
    def get_size(self):
        return lib_builder.volume_grid_size(self.vg_backend)
    
    def get_memory_size(self):
        return lib_builder.volume_grid_memory_size(self.vg_backend)

    def get_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = lib_builder.volume_grid_size(self.vg_backend)
        points = np.zeros((size, 3), dtype=np.float32)
        colors = np.zeros((size, 3), dtype=np.uint8)
        labels = np.zeros(size, dtype=np.int32)
        lib_builder.volume_grid_to_numpy(self.vg_backend,
                                             points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             colors.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                             labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        return points, colors, labels

    def save(self, path: str):
        atomic_save(path, pickle.dumps(self.get_points()))
    
    def load(self, path: str):
        try:
            with open(path, 'rb') as f:
                points, colors, labels = pickle.load(f)
                start_time = time.time()
                lib_builder.volume_grid_insert_from_numpy(self.vg_backend,
                                                  points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                  colors.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                                  labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                  points.shape[0])
        except Exception as e:
            print(e)
            print(f"Failed to load Volume Grid from {path}")

    
    def get_height(self, x: float, y: float, radius: float) -> Union[float, None]:
        z = lib_builder.volume_grid_get_z(self.vg_backend, x, y, radius)
        if z < -100: # never visited
            return None
        return z - 2 * self.conf.voxel_size
    
    def get_label(self, x: float, y: float, z: float, radius: float) -> int:
        return lib_builder.volume_grid_get_label(self.vg_backend, x, y, z, radius)
    
    def get_bound(self) -> Tuple[np.ndarray, np.ndarray]:
        min = np.zeros(3, dtype=np.float32)
        max = np.zeros(3, dtype=np.float32)
        lib_builder.volume_grid_get_bound(self.vg_backend,
                                           min.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return min, max
    
    def get_overlap(self, other: 'VolumeGridBuilder', radius=0.1) -> float:
        return lib_builder.volume_grid_get_overlap(self.vg_backend, other.vg_backend, radius)
    
    def get_occ_map(self, agent_pos: np.ndarray=None, save_path: str=None) -> tuple[np.ndarray, int, int, int, int]:
        # occ map: 1 unknown, 2 obstacle, 3 road
        x_min = np.zeros(1, dtype=np.int32)
        y_min = np.zeros(1, dtype=np.int32)
        x_max = np.zeros(1, dtype=np.int32)
        y_max = np.zeros(1, dtype=np.int32)
        occ = lib_builder.get_occurancy_map(self.vg_backend,
                                            int(self.conf.nav_grid_size / self.conf.voxel_size),
                                            x_min.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            y_min.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            x_max.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            y_max.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        w = x_max[0] - x_min[0] + 1
        h = y_max[0] - y_min[0] + 1
        occ_map = np.ctypeslib.as_array(occ, (h, w))
        if save_path is not None:
            draw_map = np.zeros((h, w, 3), dtype=np.uint8)
            draw_map[np.where(occ_map == 1)] = [0, 0, 0]
            draw_map[np.where(occ_map == 2)] = [255, 255, 255]
            draw_map[np.where(occ_map == 3)] = [0, 0, 255]

            def draw_point(x, y, color):
                x = int(self.align_nav(x))
                y = int(self.align_nav(y))
                ax = x - x_min[0]
                ay = y - y_min[0]
                if 0 <= ax < w and 0 <= ay < h:
                    draw_map[ay, ax] = color
            
            if agent_pos is not None:
                agent_pos = np.array(agent_pos)
                if len(agent_pos.shape) == 1:
                    draw_point(agent_pos[0], agent_pos[1], [0, 255, 0])
                else:
                    for pos in agent_pos:
                        draw_point(pos[0], pos[1], [0, 255, 0])
            from PIL import Image
            Image.fromarray(draw_map).save(save_path)
        return occ_map, x_min[0], y_min[0], x_max[0], y_max[0]
    
    def radius_denoise(self, min_points: int, radius: float):
        lib_builder.volume_grid_radius_denoise(self.vg_backend, min_points, radius)
    
    def has_obstacle(self, bbox: np.ndarray) -> bool:
        if lib_builder.has_obstacle(self.vg_backend,
                                    int(self.conf.nav_grid_size / self.conf.voxel_size),
                                    bbox.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))) != 0:
            return True
        return False

    def visualize(self):
        points, colors, labels = self.get_points()
        import requests
        requests.post("http://localhost:8000", data=pickle.dumps({"points": points, "labels": labels,
                      "path": "pointcloud_view.png", "special_point": np.array([0, 0, 50])}))
        # import open3d
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(points)
        # label_color = np.random.rand(int(labels.max() + 1), 3)
        # labels[labels < 0] = 0
        # # mask = np.zeros(label_color.shape[0], dtype=bool)
        # label_color[0] = 0
        # pcd.colors = open3d.utility.Vector3dVector(label_color[labels])
        # # pcd.colors = open3d.utility.Vector3dVector(colors / 255)
        # # open3d.visualization.draw_geometries([pcd])
        # # 创建一个不可见窗口用于渲染
        # vis = open3d.visualization.Visualizer()
        # vis.create_window(width=800, height=600)
        # vis.add_geometry(pcd)

        # ctr = vis.get_view_control()
        # extrinsic = np.array([[ 9.40441821e-01,  3.18287866e-01, -1.19423680e-01, -1.83882787e+01],
        #                       [ 2.05940741e-02, -4.03987675e-01, -9.14532582e-01,  4.21537649e+01],
        #                       [-3.39330319e-01,  8.57605266e-01, -3.86481749e-01,  5.34274431e+02],
        #                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        # param = ctr.convert_to_pinhole_camera_parameters()
        # param.extrinsic = extrinsic
        # ctr.convert_from_pinhole_camera_parameters(param, True)

        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image("pointcloud_view.png")
        # vis.destroy_window()
    
    def save_pcd(self, file_name):
        print (f"saving pcd to {file_name}!")
        points, colors, labels = self.get_points()
        import open3d
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors / 255)
        open3d.io.write_point_cloud(file_name, pcd)

    def close(self):
        if self.vg_backend is not None:
            lib_builder.free_volume_grid(self.vg_backend)
    
    def clear(self):
        self.close()
        self.vg_backend = lib_builder.init_volume_grid(self.conf.voxel_size, self.conf.thread_num)

    def __del__(self):
        self.close()
