import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
import genesis.utils.geom as geom_utils
from PIL import Image
import tqdm
import cv2
import glob
import os

from .modules import *
from .tools.constants import LIGHTS
from .agents.demo_agent import DemoAgent


class SimpleVicoEnv:
    def __init__(self,
                 config_path,
                 output_dir,
                 backend=gs.gpu,
                 seed=0,
                 resolution=512,
                 skip_avatar_animation=False,
                 enable_collision=False,
                 enable_third_person_cameras=False,
                 load_indoor_objects=False,
                 use_luisa_renderer=False,
                 dt_sim=0.01,
                 head_less=True
                 ):
        if not gs._initialized:
            gs.init(seed=seed, backend=backend)
        self.output_dir = output_dir
        self.config_path = config_path
        self.resolution = resolution
        self.dt_sim = dt_sim
        self.config = json.load(open(os.path.join(self.config_path, 'config.json'), 'r'))
        if 'height_field' in self.config:
            terrain_height_path = self.config['height_field']
        else:
            terrain_height_path = None
        frame_ratio = 0.0 if skip_avatar_animation else 5.0
        self.steps = self.config['step']
        self.num_agents = self.config['num_agents']
        self.curr_time: datetime = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
        self.obs = {i: {} for i in range(self.num_agents)}
        self.agent_names = self.config['agent_names']
        self.enable_collision = enable_collision
        self.enable_third_person_cameras = enable_third_person_cameras
        self.agents = []
        self.entity_idx_to_info = defaultdict(dict)
        self.scene = gs.Scene(
            # viewer_options=None,
            viewer_options=gs.options.ViewerOptions(
                res=(1000, 1000),
                camera_pos=np.array([0.0, 0.0, 1000]),
                camera_lookat=np.array([0, 0.0, 0.0]),
                camera_fov=60,
            ),
            rigid_options=gs.options.RigidOptions(
                gravity=(0.0, 0.0, 0.0),
                enable_collision=self.enable_collision,
                max_collision_pairs=400,
                dt=dt_sim
            ),
            avatar_options=gs.options.AvatarOptions(
                enable_collision=self.enable_collision,
            ),
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="textures/indoor_bright.png",
                    ),
                ),
                env_radius=100.0,
                env_euler=(0, 0, 180),
                lights=[],
            ) if use_luisa_renderer else gs.renderers.Rasterizer(),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                segmentation_level="entity",
                lights=LIGHTS
            ),
            show_viewer=not head_less,
        )
        self.load_simple_scene(scene_path=self.config["scene"], offset=self.config['scene_offset'],
                               load_indoor_objects=load_indoor_objects)
        for i in range(self.num_agents):
            self.agents.append(self.add_avatar(name=self.agent_names[i],
                                               motion_data_path='ViCo/avatars/motions/motion.pkl',
                                               skin_options={
                                                   'glb_path': self.config['agent_skins'][i],
                                                   'euler': (-90, 0, 90),
                                                   'pos': (0.0, 0.0, -0.959008030)
                                               },
                                               ego_view_options={
                                                   "res": (self.resolution, self.resolution),
                                                   "fov": 90,
                                                   "GUI": False,
                                               },
                                               frame_ratio=frame_ratio,
                                               terrain_height_path=terrain_height_path,
                                               third_person_camera_resolution=128 if self.enable_third_person_cameras else None,
                                               enable_collision=enable_collision))
        self.demo_cameras = []
        self.demo_image_counter = 0
        if "camera_config" in self.config:
            for camera_id, camera in enumerate(self.config["camera_config"]):
                self.demo_cameras.append(self.scene.add_camera(
                    res=(1024, 1024),
                    pos=camera['pos'],
                    lookat=camera['lookat'],
                    fov=90,
                    GUI=False,
                ))
                os.makedirs(os.path.join(self.output_dir, f'demo_{camera_id}'), exist_ok=True)
        self.scene.build()
        self.scene.reset()

    def load_simple_scene(self, scene_path, offset, load_indoor_objects=False):
        if scene_path.endswith('.glb'):
            # glb assets
            self.add_entity(
                type="structure",
                name="scene",
                material=gs.materials.Rigid(
                    sdf_min_res=4,
                    sdf_max_res=4,
                ),
                morph=gs.morphs.Mesh(
                    file=scene_path,
                    pos=offset,
                    euler=(90.0, 0, 0),
                    fixed=True,
                    collision=False,
                    merge_submeshes_for_collision=False,
                    group_by_material=True,
                ),
            )
        else:
            load_indoor_room(self, scene_path, offset, "demo", load_indoor_objects)

    def add_avatar(
            self,
            name: str,
            motion_data_path: str,
            skin_options=None,
            ego_view_options=None,
            frame_ratio=5.0,
            terrain_height_path=None,
            third_person_camera_resolution=None,
            enable_collision=True,
    ):
        avatar = AvatarController(
            env=self,
            motion_data_path=motion_data_path,
            skin_options=skin_options,
            ego_view_options=ego_view_options,
            frame_ratio=frame_ratio,
            terrain_height_path=terrain_height_path,
            third_person_camera_resolution=third_person_camera_resolution,
            enable_collision=enable_collision,
            name=name
        )
        return avatar

    def add_entity(self, type, name, morph, material=None,
                   surface=None, visualize_contact=False, vis_mode=None, ):
        """
        :param type: One of "structure", "building", "object", "avatar", "avatar_box", "vehicle"
        :param name:
        :param morph:
        :param material:
        :param surface:
        :param visualize_contact:
        :param vis_mode:
        :return:
        """
        entity = self.scene.add_entity(morph=morph, material=material, surface=surface,
                                       visualize_contact=visualize_contact, vis_mode=vis_mode)
        self.entity_idx_to_info[entity.idx] = {"type": type, "name": name}
        return entity

    def perform_action(self, agent_id, action):
        if action is None:
            return
        agent = self.agents[agent_id]
        agent.robot.action_status = ActionStatus.SUCCEED
        # avatar actions
        if action['type'] == 'move_forward':
            agent.move_forward(action['arg1'], 1.0)
        elif action['type'] == 'teleport':
            agent.reset_with_global_xy(np.array(action['arg1']))
        elif action['type'] == 'turn_left':
            agent.turn_left(action['arg1'], turn_sec_limit=1500)
        elif action['type'] == 'turn_right':
            agent.turn_right(action['arg1'], turn_sec_limit=1500)
        elif action['type'] == 'look_at':
            target_pos = action['arg1']
            # make avatar look at target_pos by turn_left or turn_right
            agent_pos = agent.robot.global_trans
            agent_rot = agent.robot.global_rot
            agent_dir = agent_rot[:, 0]
            target_dir = target_pos - agent_pos
            agent_dir[2] = 0
            target_dir[2] = 0
            agent_dir = agent_dir / np.linalg.norm(agent_dir)
            target_dir = target_dir / np.linalg.norm(target_dir)
            cross = np.cross(agent_dir, target_dir)
            dot = np.dot(agent_dir, target_dir)
            angle = np.arccos(dot)
            if cross[2] > 0:
                agent.turn_left(angle, turn_sec_limit=1500)
            else:
                agent.turn_right(angle, turn_sec_limit=1500)
        elif action['type'] == 'sleep':
            agent.sleep()
        elif action['type'] == 'wake':
            agent.wake()
        elif action['type'] == 'pick':  # arg1: hand id [0,1], arg2: position
            pos = np.array(action['arg2'])
            min_volume, entity_idx = 1e10, None
            for j, e in self.entity_idx_to_info.items():
                if "bbox" in e:
                    bbox = e["bbox"]
                    rigid: RigidEntity = self.entities[j]
                    rel_pos = pos - rigid.get_pos().cpu().numpy()
                    if np.all(rel_pos > bbox[0] - 0.02) and np.all(rel_pos < bbox[1] + 0.02):
                        volume = np.prod(bbox[1] - bbox[0])
                        if volume < min_volume:
                            min_volume, entity_idx = volume, j

            if entity_idx is None:
                gs.logger.warning(
                    f"Agent {self.agent_names[agent_id]} cannot pick at {pos} because no entity is found.")
                agent.robot.action_status = ActionStatus.FAIL
                return
            agent.pick(action['arg1'], self.entities[entity_idx])
        elif action['type'] == 'put':  # arg1: hand id [0,1]
            agent.put(action['arg1'], action.get('arg2', None))
        elif action['type'] == 'stand':
            agent.stand_up()
        elif action['type'] == 'sit':
            agent.sit(position=np.array(action['arg1'][0]))
        elif action['type'] == 'drink':
            agent.drink(action['arg1'])
        elif action['type'] == 'eat':
            agent.eat(action['arg1'])
        elif action['type'] == 'play_animation':
            agent.play_animation(name=action['arg1'])
        elif action['type'] == 'wait':
            return
        else:
            raise NotImplementedError(f"agent action type {action['type']} is not supported")

    def scene_step(self):
        self.scene.step()

        for camera_id, camera in enumerate(self.demo_cameras):
            rgb, _, _, _ = camera.render(depth=False)
            Image.fromarray(rgb).save(os.path.join(self.output_dir, f'demo_{camera_id}', f"{self.demo_image_counter:06d}.png"))

        self.demo_image_counter += 1

        if self.agents:
            for agent_id, avatar in enumerate(self.agents):
                avatar.step()

        if self.agents and self.enable_collision:
            collision_pairs = self.scene.rigid_solver.detect_collision()
            for i, avatar in enumerate(self.agents):
                avatar.post_step(collision_pairs)

    def reset(self):
        self.scene.reset()
        for i, agent in enumerate(self.agents):
            agent.reset(np.array(self.config['agent_poses'][i][:3], dtype=np.float64), geom_utils.euler_to_R(
                np.degrees(np.array(self.config['agent_poses'][i][3:], dtype=np.float64))))
        self.scene_step()
        self.steps = self.config['step']
        self.demo_image_counter = 0
        self.curr_time = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
        self.update_obs()
        return self.obs

    def update_obs(self):
        for i, agent in enumerate(self.agents):
            self.obs[i]['pose'] = self.config['agent_poses'][i]
            self.obs[i]['curr_time'] = self.curr_time
            self.obs[i]['steps'] = self.steps
            self.obs[i]['action_status'] = agent.action_status().value

    def step(self, agent_actions):
        for i, agent in enumerate(self.agents):
            action = agent_actions[i]
            self.perform_action(i, action)
        for _ in tqdm.tqdm(range(int(1.0 / self.dt_sim)), desc="simulating", ):
            self.scene_step()
        self.config['step'] = self.steps
        self.config['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
        self.config['agent_poses'] = []
        for i, agent in enumerate(self.agents):
            self.config['agent_poses'].append(agent.get_global_pose().tolist())
        atomic_save(os.path.join(self.config_path, 'config.json'),
                    json.dumps(self.config, indent=4, default=json_converter))
        self.update_obs()
        self.steps += 1
        return self.obs, 0, False, {}

    @property
    def entities(self):
        """All the entities in the scene."""
        return self.scene.entities

    def close(self):
        for camera_id, camera in enumerate(self.demo_cameras):
            image_dir = os.path.join(self.output_dir, f"demo_{camera_id}")
            images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(os.path.join(self.output_dir, f"demo_{camera_id}.mp4"),
                                    fourcc, 100, (width, height))
            for img_path in images:
                img = cv2.imread(img_path)
                video.write(img)
            video.release()

        gs.logger.warning("Close Simple environment")
        import gc
        self.scene = None
        gc.collect()
        gs.destroy()

