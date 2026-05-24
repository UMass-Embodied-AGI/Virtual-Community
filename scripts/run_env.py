import os
import sys
import shutil
import errno
import argparse
import json
import time
import numpy as np
import genesis as gs

from vico.env import VicoEnv
from vico.agents import get_agent_cls, AgentProcess
from vico.agents.keystroke_counter import KeyCode, KeystrokeCounter
from vico.tools.utils import get_assets_dir, atomic_save, json_converter

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--precision", type=str, default='32')
parser.add_argument("--logging_level", type=str, default='info')
parser.add_argument("--backend", type=str, default='gpu')
parser.add_argument("--head_less", '-l', action='store_true')
parser.add_argument("--multi_process", '-m', action='store_true')
parser.add_argument("--model_server_port", type=int, default=0)
parser.add_argument("--output_dir", "-o", type=str, default='output')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--challenge", type=str, default='full')

### Simulation configurations
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--enable_collision", action='store_true')
parser.add_argument("--enable_decompose", action='store_true')
parser.add_argument("--skip_avatar_animation", action='store_true')
parser.add_argument("--enable_gt_segmentation", action='store_true')
parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
parser.add_argument("--save_per_seconds", type=int, default=10)
parser.add_argument("--enable_third_person_cameras", action='store_true')
parser.add_argument("--enable_demo_camera", action='store_true')
parser.add_argument("--batch_renderer", action='store_true')
parser.add_argument("--curr_time", type=str)

### Scene configurations
parser.add_argument("--scene", type=str, default='NY')
parser.add_argument("--no_load_indoor_scene", action='store_true')
parser.add_argument("--no_load_indoor_objects", action='store_true')
parser.add_argument("--no_load_outdoor_objects", action='store_true')
parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
parser.add_argument("--no_load_scene", action='store_true')

# Traffic configurations
parser.add_argument("--no_traffic_manager", action='store_true')
parser.add_argument("--tm_vehicle_num", type=int, default=0)
parser.add_argument("--tm_avatar_num", type=int, default=0)
parser.add_argument("--enable_tm_debug", action='store_true')

### Agent configurations
parser.add_argument("--num_agents", type=int, default=15)
parser.add_argument("--config", type=str, default='agents_num_15')
parser.add_argument("--agent_type", type=str, choices=['tour_agent'], default='tour_agent')
parser.add_argument("--detect_interval", type=int, default=1)
parser.add_argument("--user_controlled_agent", type=str, default=None,
                    help="Single agent index to make user-controlled (e.g., '0' or '2')")

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

if args.overwrite and os.path.exists(args.output_dir):
    print(f"Overwrite the output directory: {args.output_dir}")
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
config_path = os.path.join(args.output_dir, 'curr_sim')
if not os.path.exists(config_path):
    seed_config_path = os.path.join(get_assets_dir(), 'scenes', args.scene, args.config)
    print(f"Initiate new simulation from config: {seed_config_path}")
    try:
        shutil.copytree(seed_config_path, config_path)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(seed_config_path, config_path)
        else:
            raise
else:
    print(f"Continue simulation from config: {config_path}")

config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
if args.debug:
    args.enable_third_person_cameras = True
    if args.curr_time is not None:
        config['curr_time'] = args.curr_time
        atomic_save(os.path.join(config_path, 'config.json'), json.dumps(config, indent=4, default=json_converter))

os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

user_controlled_index = None
if args.user_controlled_agent:
    try:
        user_controlled_index = int(args.user_controlled_agent.strip())
        print(f"User-controlled agent: {user_controlled_index}")
        print("Basic controls: Up-arrow=Forward, Left-Arrow=Turn Left, Right-Arrow=Turn Right, P=Print Status, Q=Quit")
    except ValueError:
        print("Error: Invalid format for --user_controlled_agent. Use a single stringified integer.")
        user_controlled_index = None

env = VicoEnv(
    seed=args.seed,
    precision=args.precision,
    logging_level=args.logging_level,
    backend=gs.cpu if args.backend == 'cpu' else gs.gpu,
    head_less=args.head_less,
    resolution=args.resolution,
    challenge=args.challenge,
    num_agents=args.num_agents,
    config_path=config_path,
    scene=args.scene,
    enable_indoor_scene=not args.no_load_indoor_scene,
    enable_indoor_objects=not args.no_load_indoor_objects,
    enable_outdoor_objects=not args.no_load_outdoor_objects,
    outdoor_objects_max_num=args.outdoor_objects_max_num,
    enable_collision=args.enable_collision,
    enable_decompose=args.enable_decompose,
    skip_avatar_animation=args.skip_avatar_animation,
    enable_gt_segmentation=args.enable_gt_segmentation,
    no_load_scene=args.no_load_scene,
    output_dir=args.output_dir,
    enable_third_person_cameras=args.enable_third_person_cameras,
    enable_demo_camera=args.enable_demo_camera,
    no_traffic_manager=args.no_traffic_manager,
    enable_tm_debug=args.enable_tm_debug,
    tm_vehicle_num=args.tm_vehicle_num,
    tm_avatar_num=args.tm_avatar_num,
    save_per_seconds=args.save_per_seconds,
    defer_chat=True,
    debug=args.debug,
    batch_renderer=args.batch_renderer,
    user_controlled_index=user_controlled_index,
)

agents = []
for i in range(args.num_agents):
    basic_kwargs = dict(
        name = env.agent_names[i],
        pose = env.config["agent_poses"][i],
        info = env.agent_infos[i],
        sim_path = config_path,
        debug = args.debug,
        logging_level = args.logging_level,
        multi_process = args.multi_process
    )
    if i == user_controlled_index:
        agent_type_for_this_agent = 'user_controlled_agent'
        robot_type = None
        print(f"Agent {i} ({env.agent_names[i]}) set to user-controlled")
    else:
        agent_type_for_this_agent = args.agent_type
        if 'robot_agent_id_list' in config and i in config['robot_agent_id_list']:
            robot_type = config['robot_types'][config['robot_agent_id_list'].index(i)]
        else:
            robot_type = None

    agent_cls = get_agent_cls(agent_type=agent_type_for_this_agent, robot_type=robot_type)
    agents.append(AgentProcess(agent_cls, **basic_kwargs, tour_spatial_memory=env.building_metadata))

if user_controlled_index is not None:
    env.setup_user_controlled_camera(user_controlled_index)

if args.multi_process:
    gs.logger.info("Start agent processes")
    for agent in agents:
        agent.start()
    gs.logger.info("Agent processes started")

# Simulation loop
obs = env.reset()
agent_list_to_update = obs.pop('agent_list_to_update')
agent_actions = {}
agent_actions_to_print = {}
args.max_steps = args.max_seconds // env.sec_per_step

def update_user_camera(agent_index):
    agent = env.agents[agent_index]
    agent_pos = agent.get_global_pose()[:3]
    agent_rotation = agent.robot.global_rot

    forward_dir = agent_rotation[:, 0]

    camera_distance = 5.0
    camera_height = 3.0
    camera_pos = agent_pos - forward_dir * camera_distance + np.array([0, 0, camera_height])

    lookat_distance = 3.0
    lookat_pos = agent_pos + forward_dir * lookat_distance + np.array([0, 0, 1.0])

    env.scene.viewer.set_camera_pose(pos=camera_pos, lookat=lookat_pos)

key_counter = None
if user_controlled_index is not None:
    _ax_trusted = True
    if sys.platform == 'darwin':
        try:
            from ApplicationServices import AXIsProcessTrusted
            _ax_trusted = AXIsProcessTrusted()
        except Exception:
            _ax_trusted = True
    if not _ax_trusted:
        gs.logger.warning("Keyboard listener requires Accessibility permission on macOS.")
        gs.logger.warning("Go to: System Settings → Privacy & Security → Accessibility → add Terminal.app")
        gs.logger.warning("User control disabled — running in observe-only mode.")
    else:
        try:
            key_counter = KeystrokeCounter()
            key_counter.__enter__()
            gs.logger.info("Keyboard listener started (subprocess). Use arrow keys to control the agent.")
        except Exception as e:
            gs.logger.warning(f"Failed to start keyboard listener: {e}")
            gs.logger.warning("User control disabled — running in observe-only mode.")

try:
    while True:
        lst_time = time.perf_counter()

        if key_counter is not None:
            press_events = key_counter.get_press_events()
            if press_events:
                quit_requested = False
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        print("Quit requested by user (Q key pressed)")
                        quit_requested = True
                        break

                if quit_requested:
                    break

                gs.logger.debug(f"Key events: {press_events}")
                if not args.multi_process and hasattr(agents[user_controlled_index], 'agent'):
                    agents[user_controlled_index].agent.set_key_events(press_events)
                else:
                    gs.logger.warning(f"Key events not dispatched: multi_process={args.multi_process}, has_agent={hasattr(agents[user_controlled_index], 'agent')}")

        for i, agent in enumerate(agents):
            if i in agent_list_to_update:
                agent.update(obs[i])

        for i, agent in enumerate(agents):
            if i in agent_list_to_update:
                agent_actions[i] = agent.act()
                agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None
                if agent_actions[i] is not None and agent_actions[i]['type'] == 'converse':
                    agent_actions[i]['request_chat_func'] = agent.request_chat
                    agent_actions[i]['get_utterance_func'] = agent.get_utterance

        agent_actions['agent_list_to_update'] = agent_list_to_update

        gs.logger.info(f"current time: {env.curr_time}, ViCo steps: {env.steps}, agents actions: {agent_actions_to_print}")
        sps_agent = time.perf_counter() - lst_time
        env.config["sps_agent"] = (env.config["sps_agent"] * env.steps + sps_agent) / (env.steps + 1)
        lst_time = time.perf_counter()
        obs, _, done, info = env.step(agent_actions)
        agent_list_to_update = obs.pop('agent_list_to_update')

        if user_controlled_index is not None:
            update_user_camera(user_controlled_index)

        sps_sim = time.perf_counter() - lst_time
        env.config["sps_sim"] = (env.config["sps_sim"] * (env.steps - 1) + sps_sim) / max(env.steps, 1)
        gs.logger.info(f"Time used: {sps_agent:.2f}s for agents, {sps_sim:.2f}s for simulation, "
                       f"average {env.config['sps_agent']:.2f}s for agents, "
                       f"{env.config['sps_sim']:.2f}s for simulation, "
                       f"{env.config['sps_chat']:.2f}s for post-chatting over {env.steps} steps.")

        if env.steps > args.max_steps:
            break
finally:
    if key_counter:
        key_counter.__exit__(None, None, None)

for agent in agents:
    agent.close()
env.close()
