import numpy as np


class SharedBicycles:
	def __init__(self, env, bicycle_stations_dict, price_per_minute, num_bicycles, terrain_height_path, debug=False, logger=None):
		self.env = env
		self.bicycle_stations_dict = bicycle_stations_dict
		self.price_per_minute = price_per_minute
		self.num_bicycles = num_bicycles
		self.bicycles = []
		self.timers = []
		self.debug = debug
		self.logger = logger
		self.forward_speed_m_per_s = 3
		self.angular_speed_deg_per_s = 360
		# assume one bicycle for each station at start
		for i in range(self.num_bicycles):
			self.bicycles.append(self.env.add_vehicle(
					name="bicycle",
					vehicle_asset_path="ViCo/cars/bike/bike.urdf",
					ego_view_options=None,
					dt=1e-2,
					forward_speed_m_per_s=self.forward_speed_m_per_s,
            		angular_speed_deg_per_s=self.angular_speed_deg_per_s,
					terrain_height_path=terrain_height_path
				)
			)
			self.timers.append(None)
		self.THRESHOLD = 5.0
		self.not_returned_penalty = 20


	def get_nearest_bicycle(self, pos):
		min_dist = np.inf
		nearest_bicycle = None
		nearest_bicycle_idx = None
		for idx, bicycle in enumerate(self.bicycles):
			if bicycle.occupied:
				continue
			bicycle_pos = bicycle.get_global_xy()
			dist = np.linalg.norm(np.array(pos) - np.array(bicycle_pos))
			if dist < min_dist:
				min_dist = dist
				nearest_bicycle = bicycle
				nearest_bicycle_idx = idx
		if min_dist > self.THRESHOLD:
			nearest_bicycle = None
			nearest_bicycle_idx = None
		return nearest_bicycle, nearest_bicycle_idx

	def start_timer(self, idx, curr_time):
		if idx not in range(self.num_bicycles):
			self.logger.error(f"When starting timer, got invalid bicycle index {idx}")
			return
		self.logger.info(f"Start timer for bicycle {idx} at {curr_time}")
		self.timers[idx] = curr_time

	def end_timer(self, idx, curr_time):
		cost = 0
		if idx not in range(self.num_bicycles):
			self.logger.error(f"When ending timer, got invalid bicycle index {idx}")
		elif self.timers[idx] is None:
			self.logger.error("When ending timer, Timer not started for bicycle %d" % idx)
		else:
			elapsed_time_in_minutes = int((curr_time - self.timers[idx]).total_seconds() / 60)
			self.timers[idx] = None
			cost = elapsed_time_in_minutes * self.price_per_minute
			if not self.check_at_station(idx):
				cost += self.not_returned_penalty
		self.logger.info(f"End timer for bicycle {idx} at {curr_time}, cost: {cost}")
		return cost

	def check_at_station(self, idx):
		if idx not in range(self.num_bicycles):
			self.logger.error(f"When checking at station, got invalid bicycle index {idx}")
			return True
		bicycle = self.bicycles[idx]
		bicycle_pos = bicycle.get_global_xy()
		for station in self.bicycle_stations_dict:
			station_pos = self.bicycle_stations_dict[station]
			dist = np.linalg.norm(np.array(bicycle_pos) - np.array(station_pos))
			if dist < self.THRESHOLD:
				return True
		return False

	def get_riding_bicycle(self, pos):
		min_dist = np.inf
		nearest_bicycle = None
		nearest_bicycle_idx = None
		for idx, bicycle in enumerate(self.bicycles):
			if not bicycle.occupied:
				continue
			bicycle_pos = bicycle.get_global_xy()
			dist = np.linalg.norm(np.array(pos) - np.array(bicycle_pos))
			if dist < min_dist:
				min_dist = dist
				nearest_bicycle = bicycle
				nearest_bicycle_idx = idx
		if min_dist > self.THRESHOLD:
			nearest_bicycle = None
			nearest_bicycle_idx = None
		return nearest_bicycle, nearest_bicycle_idx





