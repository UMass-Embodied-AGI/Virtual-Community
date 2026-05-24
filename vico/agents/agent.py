import json
import os
from datetime import datetime
import numpy as np
import traceback
import logging
# import torch.multiprocessing as mp
# ctx = mp.get_context('spawn')
import multiprocessing as mp # todo: replace with torch.multiprocessing

from ..tools.utils import atomic_save, json_converter

class Agent:
	def __init__(self, name, pose: list, info, sim_path, no_react=False, debug=False, logger=None):
		self.name = name
		self.pose = pose
		self.storage_path = f"{sim_path}/{name}"
		self.scratch = json.load(open(os.path.join(self.storage_path, "scratch.json"), "r"))
		self.curr_time: datetime = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
		self.seed_knowledge = json.load(open(os.path.join(self.storage_path, "seed_knowledge.json"), "r"))
		if logger:
			self.logger = logger
		else:
			self.logger = AgentLogger(self.name, "INFO", os.path.join(self.storage_path, "logs.log"))
		
		self.no_react = no_react
		self.debug = debug
		self.WALK_SPEED = 1.0 # m/s
		self.BIKE_SPEED = 3.0 # m/s
		self.fov = 90.0 # degrees
		self.current_vehicle = self.scratch['current_vehicle'] if 'current_vehicle' in self.scratch else None
		self.current_place = self.scratch['current_place'] if 'current_place' in self.scratch else None
		self.action_status = None
		self.held_objects = [None, None]
		self.cash = 0
		if info is not None:
			self.cash = info['cash']
			self.held_objects = info['held_objects']
		self.last_path = None
		self.steps = 0
		self.obs = None

	def reset(self, name, pose):
		self.name = name
		self.pose = pose
		self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
		self.current_vehicle = self.scratch['current_vehicle'] if 'current_vehicle' in self.scratch else None
		self.current_place = self.scratch['current_place'] if 'current_place' in self.scratch else None
		self.action_status = None

	def act(self, obs):
		self.steps = obs['steps']
		self.pose = obs['pose']
		self.obs = obs
		if self.curr_time is None or ("start_time" in self.scratch and self.curr_time == self.scratch["start_time"]) or self.curr_time.date() != obs['curr_time'].date():
			self.logger.info(f"a new day.")
			obs['new_day'] = True
		else:
			obs['new_day'] = False
		self.curr_time = obs['curr_time']
		self.current_place = obs['current_place']
		self.current_vehicle = obs['current_vehicle']
		self.action_status = obs['action_status']
		self.held_objects = obs['held_objects']
		self.cash = obs['cash']
		self._process_obs(obs)
		if self.action_status == "ONGOING":
			return None
		action = self._act(obs)
		self.save_scratch()
		return action

	def chat(self, content):
		# override to generate the utterance when chatting
		pass

	def _process_obs(self, obs):
		# override to process observations **each turn**
		pass

	def _act(self, obs):
		# override to perform your own action **when spare**
		pass

	def diagnose(self, text_prompt, rgb, depth, extrinsics, **kwargs):
		# overwrite to perform your own diagnosis with text prompt, rgb, depth, extrinsics. 
		pass

	def save_scratch(self):
		self.scratch['held_objects'] = self.held_objects
		self.scratch['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		self.scratch['current_vehicle'] = self.current_vehicle
		self.scratch['current_place'] = self.current_place
		atomic_save(os.path.join(self.storage_path, "scratch.json"), json.dumps(self.scratch, indent=4, default=json_converter))

	def __str__(self):
		return f"Agent {self.name} at {self.pose}"


class AgentProcess(mp.Process):
	def __init__(self, agent_cls: type[Agent], name: str, **kwargs: dict):
		super().__init__(daemon=True)
		self.agent_cls = agent_cls
		self.name = name
		self.logging_level: str = kwargs.pop("logging_level", "INFO")
		self.multi_process: bool = kwargs.pop("multi_process", True)
		self.sim_path: str = kwargs.get("sim_path", "curr_sim")
		self.kwargs = kwargs
		self.input_queue = mp.Queue()
		self.action_queue = mp.Queue()
		self.utterance_queue = mp.Queue()
		if not self.multi_process:
			self.agent = self.create_agent()
		else:
			self.agent = None

	def create_agent(self):
		logger = AgentLogger(self.name, self.logging_level,
							 os.path.join(self.sim_path.replace('curr_sim', 'logs'), f"{self.name}.log"))
		agent = self.agent_cls(self.name, logger=logger, **self.kwargs)
		return agent

	def log_step_agent_info(self, obs_i, action):
		obs_printable = {k: v for k, v in obs_i.items() if
						 not isinstance(v, np.ndarray) and not isinstance(v, datetime)}
		obs_printable.pop("gt_seg_idxc_to_info", None)
		step_info = {"curr_time": obs_i["curr_time"],
					 "obs": obs_printable,
					 "action": action,  # todo: if ongoing, then log down the last action [which is ongoing]
					 "curr_events": self.agent.curr_events if hasattr(self.agent, "curr_events") else None,
					 "react_mode": self.agent.react_mode if hasattr(self.agent, "react_mode") else None,
					 "chatting_buffer": self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None,
					 "commute_plan": self.agent.commute_plan if hasattr(self.agent, "commute_plan") else None,
					 "commute_plan_idx": self.agent.commute_plan_idx if hasattr(self.agent, "commute_plan_idx") else None, }
		if hasattr(self.agent, "curr_goal_description"):
			step_info["action_desp"] = self.agent.curr_goal_description
			step_info["action_dura"] = self.agent.curr_goal_duration
			step_info["action_location"] = self.agent.curr_goal_address
		elif hasattr(self.agent, "hourly_schedule") and self.agent.curr_schedule_idx < len(self.agent.hourly_schedule):
			step_info["action_desp"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["activity"]
			step_info["action_location"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["place"] if "place" in \
																												self.agent.hourly_schedule[
																													self.agent.curr_schedule_idx] else None

		step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name, f"{self.agent.steps:06d}.json")
		atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))

	def run(self):
		self.agent = self.create_agent()
		self.agent.logger.info(f"Agent {self.name} started")
		while True:
			input = self.input_queue.get()
			if input["type"] == "chat":
				content = input["content"]
				try:
					utterance = self.agent.chat(content)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
					if "Stop the exp immediately" in e.args[0]:
						break # end the agent process immediately
					utterance = None
				self.utterance_queue.put(utterance)
				step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name,
											  f"{self.agent.steps:06d}.json")
				step_info = json.load(open(step_info_path, 'r'))
				step_info["action"]["arg1"] = utterance
				step_info["chatting_buffer"] = self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None
				atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))
			elif input["type"] == "obs":
				obs = input["content"]
				self.agent.logger.debug(f"Agent {self.name} received obs at {obs['curr_time']}")
				try:
					action = self.agent.act(obs)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
					action = None
				self.action_queue.put(action)
				self.log_step_agent_info(obs, action)
			else:
				self.agent.logger.error(f"Agent {self.name} received unknown input: {input}")

	def update(self, obs):
		self.input_queue.put({"type": "obs", "content": obs})

	def act(self):
		if self.multi_process:
			return self.action_queue.get()
		else:
			obs = self.input_queue.get()["content"]
			try:
				action = self.agent.act(obs)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
				action = None
			self.log_step_agent_info(obs, action)
			return action

	def request_chat(self, content):
		self.input_queue.put({"type": "chat", "content": content})

	def get_utterance(self, steps):
		if self.multi_process:
			utterance = self.utterance_queue.get()
		else:
			content = self.input_queue.get()["content"]
			try:
				utterance = self.agent.chat(content)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
				utterance = None

		return utterance
	
	def close(self):
		if self.multi_process:
			if self.agent.is_alive():
				self.agent.terminate()


class AgentLogger(logging.Logger):
    def __init__(self, name: str, level: str, output_path: str = None):
        super().__init__(f"Agent <{name}>")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging._nameToLevel[level.upper()])
        self.addHandler(console_handler)
        if output_path:
            file_handler = logging.FileHandler(output_path)
            file_handler.setLevel(logging.DEBUG)
            self.addHandler(file_handler)
        self.formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Agent <{name}>] %(message)s"
        )
        for handler in self.handlers:
            handler.setFormatter(self.formatter)