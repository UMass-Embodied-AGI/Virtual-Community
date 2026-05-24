import os
from threading import Lock, Thread
import time
import numpy as np

from .sg.builder.builder import Builder, BuilderConfig, VolumeGridBuilderConfig

class SceneGraph:
	def __init__(self, storage_path, detect_interval=-1, fov=90.0, debug=False, logger=None):
		self.storage_path = storage_path
		self.detect_interval = detect_interval
		self.fov = fov
		self.debug = debug
		self.logger = logger
		self.scene_graph_dict = {}
		self.current_place = "open space"
		self.get_sg(self.current_place)
		self.num_frames = 0
		self.last_processed_rgb = None
		self.saving_lock = Lock()
		self.explored = {}
		self.SIMILARITY_THRESHOLD = 0.75

	def get_sg(self, place=None):
		if place is None:
			place = "open space"
		if place not in self.scene_graph_dict:
			output_path = f"{self.storage_path}/{place}"
			os.makedirs(output_path, exist_ok=True)
			if place == "open space":
				volume_grid_conf = VolumeGridBuilderConfig(voxel_size=0.1, nav_grid_size=0.5, depth_bound=30.0)
			else:
				volume_grid_conf = VolumeGridBuilderConfig(voxel_size=0.025, nav_grid_size=0.2, depth_bound=30.0)

			self.scene_graph_dict[place] = Builder(
				BuilderConfig(output_path=output_path,
							  volume_grid_conf=volume_grid_conf, fov=self.fov, debug=self.debug, logger=self.logger))
			if os.path.exists(f"{self.storage_path}/{place}/volume_grid.pkl"):
				print(f"Loading volume grid for {place}...")
				self.scene_graph_dict[place].volume_grid_builder.load(f"{self.storage_path}/{place}/volume_grid.pkl")
		if self.current_place is not None and self.current_place != place:
			self.scene_graph_dict[self.current_place].volume_grid_builder.save(f"{self.storage_path}/{self.current_place}/volume_grid.pkl")
		self.current_place = place
		return self.scene_graph_dict[place]

	def update(self, obs):
		while self.saving_lock.locked():
			time.sleep(0.1)
		if obs['rgb'] is None:
			return
		cur_sg = self.get_sg(obs['current_place'])
		labels = -np.ones_like(obs['depth'], dtype=np.int32)
		cur_sg.add_frame(obs['rgb'], obs['depth'], labels, obs['extrinsics'])
		self.num_frames += 1

		if self.current_place is None or self.current_place == "open space":
			occ_map = cur_sg.volume_grid_builder.get_occ_map(obs["pose"][:3])[0]
			self.logger.debug(f"Area of known grids: {np.sum(occ_map != 1)}")
		if self.debug:
			cur_sg.volume_grid_builder.get_occ_map(obs["pose"][:3], os.path.join(self.storage_path, self.current_place, f"occ_map_{cur_sg.num_frames:06d}.png"))
		if self.num_frames % 10 == 0:
			self.save_memory()

	def save_memory(self):
		def _save():
			with self.saving_lock:
				self.scene_graph_dict[self.current_place].volume_grid_builder.save(
					f"{self.storage_path}/{self.current_place}/volume_grid.pkl")

		Thread(target=_save, daemon=True).start()
