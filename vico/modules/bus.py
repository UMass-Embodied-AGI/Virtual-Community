import datetime
import numpy as np
import copy

class Bus:
	def __init__(self, bus, forward_speed, rotation_speed, current_time, route, stop_names, stop_indices, travel_time, bus_start_time, bus_end_time, frequency, debug=False, logger=None):
		self.bus = bus
		self.forward_speed = forward_speed
		self.rotation_speed = rotation_speed
		self.route = route
		self.stop_names = stop_names
		self.stop_indices = stop_indices
		self.debug = debug
		self.nav_last_target = None
		self.nav_trans_history = []
		self._action_queue = []
		self.logger = logger
		self.travel_time = travel_time
		self.frequency = frequency
		self.schedule, self.schedule_reversed = self.get_schedule(
			bus_travel_time=self.travel_time,
			bus_start_time=bus_start_time,
			bus_end_time=bus_end_time,
			frequency=self.frequency,
			bus_stop_indices=self.stop_indices,
			bus_stop_names=self.stop_names
		)
		self.stop_at_this_step = None
		self.current_stop_name = None
		self.op_earliest_time = datetime.datetime.strptime(self.schedule[0]["arrival_times"][0], "%H:%M:%S").time()
		self.op_latest_time = datetime.datetime.strptime(self.schedule_reversed[0]["arrival_times"][-1], "%H:%M:%S").time()
		self.update_at_time(current_time)

	def get_schedule(self, bus_travel_time, bus_start_time, bus_end_time, frequency, bus_stop_indices, bus_stop_names, stop_time=30):
		start_time = datetime.datetime.strptime(bus_start_time, "%H:%M:%S")
		end_time = datetime.datetime.strptime(bus_end_time, "%H:%M:%S")
		arrival_interval = datetime.timedelta(minutes=frequency)
		schedule = []
		schedule_reversed = []
		for i, stop_index in enumerate(bus_stop_indices):
			stop_schedule = {
				'stop_name': bus_stop_names[i],
				'arrival_times': [],
				'departure_times': []
			}
			schedule.append(copy.deepcopy(stop_schedule))
			schedule_reversed.append(copy.deepcopy(stop_schedule))
		current_arrival_time = start_time
		while current_arrival_time <= end_time:
			this_time = current_arrival_time
			for i in range(len(bus_stop_indices)-1):
				schedule[i]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
				this_time += datetime.timedelta(seconds=stop_time)
				schedule[i]['departure_times'].append(this_time.strftime("%H:%M:%S"))
				this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i]}->{bus_stop_indices[i+1]}"]))
			schedule[len(bus_stop_indices)-1]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
			schedule_reversed[len(bus_stop_indices)-1]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
			this_time += datetime.timedelta(seconds=stop_time)
			schedule[len(bus_stop_indices)-1]['departure_times'].append(this_time.strftime("%H:%M:%S"))
			schedule_reversed[len(bus_stop_indices)-1]['departure_times'].append(this_time.strftime("%H:%M:%S"))
			for i in range(len(bus_stop_indices)-2, 0, -1):
				this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[i+1]}->{bus_stop_indices[i]}"]))
				schedule_reversed[i]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
				this_time += datetime.timedelta(seconds=stop_time)
				schedule_reversed[i]['departure_times'].append(this_time.strftime("%H:%M:%S"))
			this_time += datetime.timedelta(seconds=int(bus_travel_time[f"{bus_stop_indices[1]}->{bus_stop_indices[0]}"]))
			schedule_reversed[0]['arrival_times'].append(this_time.strftime("%H:%M:%S"))
			current_arrival_time += arrival_interval
		return schedule, schedule_reversed
		
	def reset(
		self, 
		global_trans,
        global_rot
		):
		self._action_queue = []
		if global_trans is not None and global_rot is not None:
			self.bus.reset(np.array([global_trans[0], global_trans[1], 3.0]), global_rot)
		self.pose = self.bus.get_global_pose()
		self.logger.debug(f"Bus reset to {self.pose}, global position xy: {self.bus.get_global_xy()}")

	def should_bus_stop(self, current_time, current_waypoint_index):
		if current_waypoint_index not in self.stop_indices:
			return False
		current_stop_indices_loc = self.stop_indices.index(current_waypoint_index)
		if self.route_reversed:
			stop_schedule_info = self.schedule_reversed[current_stop_indices_loc]
		else:
			stop_schedule_info = self.schedule[current_stop_indices_loc]
		arrival_times = [datetime.datetime.strptime(arrival_time, "%H:%M:%S").time() for arrival_time in stop_schedule_info["arrival_times"]]
		departure_times = [datetime.datetime.strptime(departure_time, "%H:%M:%S").time() for departure_time in stop_schedule_info["departure_times"]]
		if len(departure_times) == 0:
			return True
		if current_time.time() < self.op_earliest_time or current_time.time() > self.op_latest_time:
			return True
		# Count the cycle first for bus reaching to the end of the rounded trip at the starting stop
		cycle_num = int((datetime.datetime.combine(datetime.date.today(), current_time.time()) - \
						datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(self.schedule[0]["arrival_times"][0], "%H:%M:%S").time())) / datetime.timedelta(minutes=self.frequency))
		last_arrival_time = self.schedule_reversed[0]["arrival_times"][cycle_num]
		if current_time.time() >= datetime.datetime.strptime(last_arrival_time, "%H:%M:%S").time():
			return True
		else:
			for arrival_time, departure_time in zip(arrival_times, departure_times):
				if arrival_time <= current_time.time() <= departure_time:
					return True
			else:
				return False
	
	def get_all_times(self):
		all_times = []
		for i in range(0, len(self.schedule)):
			for j in range(0, len(self.schedule[i]["arrival_times"])):
				all_times.append(datetime.datetime.strptime(self.schedule[i]["arrival_times"][j], "%H:%M:%S").time())
			for j in range(0, len(self.schedule[i]["departure_times"])):
				all_times.append(datetime.datetime.strptime(self.schedule[i]["departure_times"][j], "%H:%M:%S").time())
		for i in range(0, len(self.schedule_reversed)):
			for j in range(0, len(self.schedule_reversed[i]["arrival_times"])):
				all_times.append(datetime.datetime.strptime(self.schedule_reversed[i]["arrival_times"][j], "%H:%M:%S").time())
			for j in range(0, len(self.schedule_reversed[i]["departure_times"])):
				all_times.append(datetime.datetime.strptime(self.schedule_reversed[i]["departure_times"][j], "%H:%M:%S").time())
		return all_times
	
	def calculate_pose_helper_given_current_pos_xy(self, current_pos_xy): # make sure to update self.next_waypoint_index first!
		next_waypoint_pos_xy = self.route[self.next_waypoint_index]
		target_rad = np.arctan2(next_waypoint_pos_xy[1] - current_pos_xy[1], next_waypoint_pos_xy[0] - current_pos_xy[0])
		return current_pos_xy + [3.0] + [0.0, 0.0, target_rad]
	
	def calculate_pose_helper_given_current_waypoint_index(self, current_waypoint_index):
		if self.route_reversed:
			self.next_waypoint_index = current_waypoint_index - 1
		else:
			self.next_waypoint_index = current_waypoint_index + 1
		current_pos_xy = self.route[current_waypoint_index]
		next_waypoint_pos_xy = self.route[self.next_waypoint_index]
		target_rad = np.arctan2(next_waypoint_pos_xy[1] - current_pos_xy[1], next_waypoint_pos_xy[0] - current_pos_xy[0])
		return current_pos_xy + [3.0] + [0.0, 0.0, target_rad]
		
	def update_at_time(self, current_time):
		# print(f"Bus: update at time: {current_time}")
		self.route_reversed = None
		if current_time.time() < self.op_earliest_time or current_time.time() > self.op_latest_time:
			# print("Bus:Scenario 1: out of operating time")
			self.route_reversed = False
			self.current_stop_name = self.stop_names[0]
			self.stop_at_this_step = True
			return self.calculate_pose_helper_given_current_waypoint_index(current_waypoint_index=0)
		else:
			for i in range(len(self.schedule[0]["arrival_times"])):
				if datetime.datetime.strptime(self.schedule[0]["arrival_times"][i], "%H:%M:%S").time() <= current_time.time() < datetime.datetime.strptime(self.schedule_reversed[-1]["arrival_times"][i], "%H:%M:%S").time():
					self.route_reversed = False
		if self.route_reversed == None:
			self.route_reversed = True
		schedule = self.schedule_reversed[::-1] if self.route_reversed else self.schedule
		last_departure_stop_index = None
		last_departure_time = None
		for i in range(0, len(schedule[0]["arrival_times"])):
			for stop_info in schedule:
				# print(stop_info["stop_name"])
				arrival_times = stop_info["arrival_times"]
				departure_times = stop_info["departure_times"]
				if departure_times:
					arrival_time = arrival_times[i]
					departure_time = departure_times[i]
					this_arrival_time = datetime.datetime.strptime(arrival_time, "%H:%M:%S").time()
					this_departure_time = datetime.datetime.strptime(departure_time, "%H:%M:%S").time()
					if this_arrival_time <= current_time.time() <= this_departure_time:
						current_stop_name = stop_info["stop_name"]
						current_stop_index = self.stop_names.index(current_stop_name)
						current_waypoint_index = self.stop_indices[current_stop_index]
						self.current_stop_name = current_stop_name
						self.stop_at_this_step = True
						return self.calculate_pose_helper_given_current_waypoint_index(current_waypoint_index=current_waypoint_index)
					if this_departure_time < current_time.time():
						last_departure_stop_name = stop_info["stop_name"]
						last_departure_time = this_departure_time
					else:
						break
				else:
					# Count the cycle first
					cycle_num = int((datetime.datetime.combine(datetime.date.today(), current_time.time()) - \
					  				datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(self.schedule[0]["arrival_times"][0], "%H:%M:%S").time())) / datetime.timedelta(minutes=self.frequency))
					last_arrival_time = self.schedule_reversed[0]["arrival_times"][cycle_num]
					if current_time.time() >= datetime.datetime.strptime(last_arrival_time, "%H:%M:%S").time():
						self.current_stop_name = self.stop_names[0]
						self.stop_at_this_step = True
						self.next_waypoint_index = 1
						self.route_reversed = False
						return self.route[0] + [3.0] + [0.0, 0.0, np.arctan2(self.route[0][1] - self.route[1][1], self.route[0][0] - self.route[1][0])]
		# print("Bus:Scenario 4: running between stops")
		last_departure_stop_index = self.stop_names.index(last_departure_stop_name)
		if self.route_reversed:
			next_stop_index = last_departure_stop_index - 1
		else:
			next_stop_index = last_departure_stop_index + 1
		last_stop_waypoint_index = self.stop_indices[last_departure_stop_index]
		next_stop_waypoint_index = self.stop_indices[next_stop_index]
		travel_key = f"{last_stop_waypoint_index}->{next_stop_waypoint_index}"
		current_time_in_today = datetime.datetime.combine(datetime.date.today(), current_time.time())
		last_departure_time_in_today = datetime.datetime.combine(datetime.date.today(), last_departure_time)
		time_elapsed = (current_time_in_today - last_departure_time_in_today).total_seconds()
		if self.route_reversed:
			waypoints_in_between = list(range(last_stop_waypoint_index, next_stop_waypoint_index - 1, -1))
		else:
			waypoints_in_between = list(range(last_stop_waypoint_index, next_stop_waypoint_index + 1))
		cumulative_time = 0
		proportion_traveled = None
		for i in range(len(waypoints_in_between) - 1):
			current_waypoint_index = waypoints_in_between[i]
			next_waypoint_index = waypoints_in_between[i + 1]
			travel_key = f"{current_waypoint_index}->{next_waypoint_index}"
			segment_time = self.travel_time[travel_key]
			# print(f"segment_time from {current_waypoint_index} to {next_waypoint_index}:", segment_time)
			if cumulative_time + segment_time >= time_elapsed:
				delta_time_elapsed = time_elapsed - cumulative_time
				proportion_traveled = min(delta_time_elapsed / segment_time, 1)
				break
			cumulative_time += segment_time
		last_waypoint_pos_xy = self.route[current_waypoint_index]
		next_waypoint_pos_xy = self.route[next_waypoint_index]
		current_x = last_waypoint_pos_xy[0] + (next_waypoint_pos_xy[0] - last_waypoint_pos_xy[0]) * proportion_traveled
		current_y = last_waypoint_pos_xy[1] + (next_waypoint_pos_xy[1] - last_waypoint_pos_xy[1]) * proportion_traveled
		self.next_waypoint_index = next_waypoint_index
		self.current_stop_name = None
		self.stop_at_this_step = False
		return self.calculate_pose_helper_given_current_pos_xy(current_pos_xy=[current_x, current_y])

	def step(self, current_time):
		self.current_stop_name = None
		self.stop_at_this_step = False
		round_trip_ended_and_waiting_for_next = False
		# print("D towards next waypoint index:", np.sqrt(sum(np.power((np.array(self.route[self.next_waypoint_index]) - np.array(self.bus.get_global_xy()[:2])), 2))))
		if np.sqrt(sum(np.power((np.array(self.route[self.next_waypoint_index]) - np.array(self.bus.get_global_xy()[:2])), 2))) < 1:
			current_waypoint_index = self.next_waypoint_index
			self.stop_at_this_step = self.should_bus_stop(current_time, current_waypoint_index)
			if self.next_waypoint_index == len(self.route) - 1:
				self.route_reversed = True
			elif self.next_waypoint_index == 0:
				if current_time.time() in [datetime.datetime.strptime(arrival_time, "%H:%M:%S").time() for arrival_time in self.schedule[0]["arrival_times"]]:
					self.route_reversed = False
					# print("In arrive times...")
				else:
					round_trip_ended_and_waiting_for_next = True
					# print("Not in arrive times... Waiting...")
				# print("Route Reversed?", self.route_reversed)
			if not round_trip_ended_and_waiting_for_next:
				if self.route_reversed:
					self.next_waypoint_index -= 1
				else:
					self.next_waypoint_index += 1
		# print("next waypoint index:", self.next_waypoint_index)
		# print("route reversed:", self.route_reversed)
		else:
			# not reaching to the next waypoint index, but currently at current waypoint index, still should go into check if bus should stop
			if self.route_reversed:
				current_waypoint_index = self.next_waypoint_index + 1
			else:
				current_waypoint_index = self.next_waypoint_index - 1
			if np.sqrt(sum(np.power((np.array(self.route[current_waypoint_index]) - np.array(self.bus.get_global_xy()[:2])), 2))) < 1:
				# print("at first")
				self.stop_at_this_step = self.should_bus_stop(current_time, current_waypoint_index)

		if not self.stop_at_this_step:
			next_waypoint = np.array(self.route[self.next_waypoint_index])
			self.pose = self.bus.get_global_pose()
			nav_result = self.navigate(self.bus.get_global_pose()[:2], self.bus.robot.global_rot, next_waypoint, self.next_waypoint_index)
			if len(self._action_queue) == 0:
				if nav_result is None:
					return None
				elif isinstance(nav_result, list):
					assert len(nav_result) > 0
					self._action_queue.extend(nav_result)
				else:
					self._action_queue.append(nav_result)
			bus_action = self._action_queue.pop(0)
			if bus_action is not None:
				self.logger.debug(f"bus action: {bus_action}")
				if bus_action['type'] == 'move_forward':
					self.bus.move_forward(target_pos=bus_action['arg1'], speed=bus_action['arg2'])
				elif bus_action['type'] == 'turn_left_schedule':
					self.bus.turn_left_schedule(angle=bus_action['arg1'], speed=bus_action['arg2'])
				elif bus_action['type'] == 'turn_right_schedule':
					self.bus.turn_right_schedule(angle=bus_action['arg1'], speed=bus_action['arg2'])
				else:
					raise NotImplementedError(f"bus action type {bus_action['type']} is not supported")
			self.current_stop_name = None
		else:
			assert current_waypoint_index in self.stop_indices # sanity check (bus must stop at stops)
			stop_index_in_names = self.stop_indices.index(current_waypoint_index)
			self.current_stop_name = self.stop_names[stop_index_in_names]

	def navigate(self, cur_trans, cur_rot, cur_goal, next_waypoint_index):
		if cur_goal is None:
			return None
		# move
		target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
		R = cur_rot
		cur_rad = np.arctan2(R[1, 0], R[0, 0])
		delta_rad = target_rad - cur_rad
		# print("delta angle:", np.rad2deg(delta_rad))
		if delta_rad > np.pi:
			delta_rad -= 2 * np.pi
		elif delta_rad < -np.pi:
			delta_rad += 2 * np.pi
		if delta_rad > 0:
			action = [{
				'type': 'turn_left_schedule',
				'arg1': np.rad2deg(delta_rad),
				'arg2': self.rotation_speed,
				'arg3': cur_goal,
				'arg4': next_waypoint_index
			}]
		else:
			action = [{
				'type': 'turn_right_schedule',
				'arg1': np.rad2deg(-delta_rad),
				'arg2': self.rotation_speed,
				'arg3': cur_goal,
				'arg4': next_waypoint_index
			}]
		action.append({
			'type': 'move_forward',
			'arg1': cur_goal,
			'arg2': self.forward_speed, # np.linalg.norm(cur_goal - cur_trans),
			'arg3': cur_goal,
			'arg4': next_waypoint_index
		})
		return action