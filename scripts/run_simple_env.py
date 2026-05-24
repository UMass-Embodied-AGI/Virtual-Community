import os
import sys
import shutil
import errno
import argparse
import json
import time

import genesis as gs

from vico.simple_env import SimpleVicoEnv
from vico.agents.demo_agent import DemoAgent

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", "-o", type=str, default='output/')
parser.add_argument("--backend", "-b", type=str, default='gpu')
parser.add_argument("--step_limit", "-s", type=int, default=1000)
parser.add_argument("--load_indoor_objects", action='store_true')
parser.add_argument("--use_luisa_renderer", action='store_true')
parser.add_argument("--overwrite", action='store_true')

args = parser.parse_args()

if args.overwrite and os.path.exists(args.output_dir):
    print(f"Overwrite the output directory: {args.output_dir}")
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
curr_sim_path = os.path.join(args.output_dir, 'curr_sim')
if not os.path.exists(curr_sim_path):
    print(f"Initiate new simulation from config: {args.config_path}")
    try:
        shutil.copytree(args.config_path, curr_sim_path)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(args.config_path, curr_sim_path)
        else:
            raise
else:
    print(f"Continue simulation from config: {curr_sim_path}")

env = SimpleVicoEnv(config_path=curr_sim_path,
                    output_dir=args.output_dir,
                    backend=gs.cpu if args.backend == 'cpu' else gs.gpu,
                    resolution=512,
                    skip_avatar_animation=False,
                    enable_collision=True,
                    enable_third_person_cameras=True,
                    load_indoor_objects=args.load_indoor_objects,
                    use_luisa_renderer=args.use_luisa_renderer,
                    dt_sim=0.01,
                    head_less=True)

agents = []
config = json.load(open(os.path.join(curr_sim_path, 'config.json'), 'r'))
for agent_id in range(env.num_agents):
    agents.append(DemoAgent(config['agent_actions'][agent_id]))

obs = env.reset()
for _ in range(args.step_limit):
    agent_actions = {i: None for i in range(env.num_agents)}
    for agent_id, agent in enumerate(agents):
        agent_actions[agent_id] = agent.act(obs[agent_id])
    obs, _, done, info = env.step(agent_actions)
env.close()
