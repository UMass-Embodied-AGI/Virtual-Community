import numpy as np
import queue
from .agent import Agent
from .keystroke_counter import KeyCode, KeystrokeCounter


class UserControlledAgent(Agent):
    """Agent controlled by user input, moving forward autonomously when idle."""

    def __init__(self, name, pose, info, sim_path, tour_spatial_memory=None, no_react=False, debug=False, logger=None,
                 move_speed=1.0, turn_speed=45.0):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        
        self.move_speed = move_speed  
        self.turn_speed = turn_speed  
        self.pending_actions = queue.Queue()
        
        # self.logger.info(f"UserControlledAgent {name} initialized with move_speed={move_speed}, turn_speed={turn_speed}")
        # self.logger.info("Controls: up=Forward, left=Turn Left, right=Turn Right, Q=Exit Control")
        
    def process_key_events(self, press_events):
        """Process key events and queue actions - using arrow keys and IJK"""
        from pynput.keyboard import Key, KeyCode
        
        for key_stroke in press_events:
            action = None
            
            # Use arrow keys instead of WAD to avoid Genesis conflicts
            # Handle arrow keys using pynput Key objects
            if key_stroke == Key.up:  # Up arrow
                action = {
                    'type': 'move_forward',
                    'arg1': self.move_speed
                }
                self.logger.debug("Up arrow pressed - Move forward")
                
            elif key_stroke == Key.left:  # Left arrow
                action = {
                    'type': 'turn_left', 
                    'arg1': self.turn_speed
                }
                self.logger.debug("Left arrow pressed - Turn left")
                
            elif key_stroke == Key.right:  # Right arrow
                action = {
                    'type': 'turn_right',
                    'arg1': self.turn_speed
                }
                self.logger.debug("Right arrow pressed - Turn right")  

            # ===== OBJECT INTERACTION =====
            elif key_stroke == KeyCode(char='b'):  # Pick action
                action = {
                    'type': 'pick',
                    'arg1': 0,  # hand id [0,1]
                    'arg2': None  # Defaulted to none for now
                    # Will need special handling to pick specific objects later
                }
                self.logger.info("B pressed - Pick object")
                
            elif key_stroke == KeyCode(char='n'):  # Put action
                action = {
                    'type': 'put',
                    'arg1': 0  # hand id [0,1]
                }
                self.logger.info("N pressed - Put object")

            # ===== VEHICLE CONTROLS =====
            elif key_stroke == KeyCode(char='m'):  # Enter/Exit bus
                # Check current vehicle state from obs
                if hasattr(self, 'obs') and self.obs.get('current_vehicle') == 'bus':
                    action = {
                        'type': 'exit_bus',
                    }
                    self.logger.info("M pressed - Exit bus")
                else:
                    action = {
                        'type': 'enter_bus',
                    }
                    self.logger.info("M pressed - Enter bus")

            elif key_stroke == KeyCode(char=',') or key_stroke == KeyCode(char='<'):  # Enter/Exit bike
                # Check current vehicle state from obs
                if hasattr(self, 'obs') and self.obs.get('current_vehicle') == 'bicycle':
                    action = {
                        'type': 'exit_bike',
                    }
                    self.logger.info("< pressed - Exit bike")
                else:
                    action = {
                        'type': 'enter_bike',
                    }
                    self.logger.info("< pressed - Enter bike")

            # ===== BUILDING CONTROLS =====
            elif key_stroke == KeyCode(char='.') or key_stroke == KeyCode(char='>'):  # Enter/Exit building
                if hasattr(self, 'obs') and self.obs.get('current_building') != 'open space':
                    # Exit to open space
                    action = {
                        'type': 'enter',
                        'arg1': 'open space'
                    }
                    self.logger.info("> pressed - Exit building (enter open space)")
                else:
                    # Enter building 
                    # Check accessible places and enter the first one 
                    if hasattr(self, 'obs') and self.obs.get('accessible_places'):
                        accessible = [p for p in self.obs['accessible_places'] if p != 'open space']
                        if accessible:
                            action = {
                                'type': 'enter',
                                'arg1': accessible[0]  # Enter first accessible place
                            }
                            self.logger.info(f"> pressed - Enter building: {accessible[0]}")
                        else:
                            self.logger.warning("> pressed - No accessible places to enter")
                    else:
                        self.logger.warning("> pressed - No accessible places available")

            elif key_stroke == KeyCode(char='p'):
                # Print status
                self._print_status()  
                
            if action:
                # Clear any pending actions and add the new one
                while not self.pending_actions.empty():
                    try:
                        self.pending_actions.get_nowait()
                    except queue.Empty:
                        break
                self.pending_actions.put(action)
            
    def set_key_events(self, press_events):
        """Called from the main simulation loop to pass key events"""
        self.process_key_events(press_events)
        
    def _act(self, obs):
        """Override base _act method to handle queued user actions"""
        # Store observation for potential use
        self.obs = obs
        
        # Check if we have any pending actions from user input
        if not self.pending_actions.empty():
            try:
                action = self.pending_actions.get_nowait()
                return action
            except queue.Empty:
                pass
            
        return {'type': 'move_forward', 'arg1': self.move_speed}
        
    def _process_obs(self, obs):
        """Process observations"""
        super()._process_obs(obs)
        self.obs = obs

    def _print_status(self):
        """Print current agent status"""
        if hasattr(self, 'obs') and self.obs:
            self.logger.info(f"=== Agent {self.name} Status ===")
            self.logger.info(f"Position: {self.pose[:3]}")
            self.logger.info(f"Orientation: {self.pose[3:]}")
            self.logger.info(f"Current Place: {self.obs.get('current_place', 'None')}")
            self.logger.info(f"Current Building: {self.obs.get('current_building', 'None')}")
            self.logger.info(f"Current Vehicle: {self.obs.get('current_vehicle', 'None')}")
            self.logger.info(f"Cash: {self.obs.get('cash', 0)}")
            self.logger.info(f"Accessible Places: {self.obs.get('accessible_places', [])}")
            self.logger.info(f"Action Status: {self.obs.get('action_status', 'Unknown')}")
            self.logger.info("================================")
        else:
            self.logger.info(f"Agent {self.name} - No observation data available yet")