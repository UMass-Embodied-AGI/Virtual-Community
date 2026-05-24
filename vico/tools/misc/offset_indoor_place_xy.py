from vico.tools.constants import ASSETS_PATH
import os
import json
import argparse

def offset_places_xy_in_place_metadata(place_metadata, offset):
    for place in place_metadata:
        if len(place_metadata[place]["location"]) == 3 and place_metadata[place]["location"][2] < 0 and place_metadata[place]["building"] != "open space" and place_metadata[place]["location"][0] < (offset / 2) and place_metadata[place]["location"][1] < (offset / 2):
            place_metadata[place]["location"][0] += offset
            place_metadata[place]["location"][1] += offset
    return place_metadata

def offset_places_xy_in_building_metadata(building_metadata, offset):
    for building in building_metadata:
        if building == "open space":
            continue
        for place_i in range(len(building_metadata[building]["places"])):
            if len(building_metadata[building]["places"][place_i]["location"]) == 3 and building_metadata[building]["places"][place_i]["location"][2] < 0 and building_metadata[building]["places"][place_i]["location"][0] < (offset / 2) and building_metadata[building]["places"][place_i]["location"][1] < (offset / 2):
                building_metadata[building]["places"][place_i]["location"][0] += offset
                building_metadata[building]["places"][place_i]["location"][1] += offset
    return building_metadata

def offset_places_xy_in_config_metadata(config_metadata, offset):
    for i in range(len(config_metadata["agent_poses"])):
        if config_metadata["agent_poses"][i][2] < 0 and config_metadata["agent_poses"][i][0] < (offset / 2) and config_metadata["agent_poses"][i][1] < (offset / 2):
            config_metadata["agent_poses"][i][0] += offset
            config_metadata["agent_poses"][i][1] += offset
    return config_metadata

def offset_places_xy_in_knowledge(knowledge, offset):
    for k in knowledge:
        if "location" in knowledge[k]:
            if len(knowledge[k]["location"]) == 3 and knowledge[k]["location"][2] < 0 and knowledge[k]["building"] != "open space" and knowledge[k]["location"][0] < (offset / 2) and knowledge[k]["location"][1] < (offset / 2):
                knowledge[k]["location"][0] += offset
                knowledge[k]["location"][1] += offset
    return knowledge

if __name__ == "__main__":
    base_dir = f"{ASSETS_PATH}/scenes"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--num_agents", "-n", type=int, default=15)
    args = parser.parse_args()
    scene_dir = os.path.join(base_dir, args.scene)
    config_dir = os.path.join(scene_dir, f"agents_num_{str(args.num_agents)}")
    config_schedules_dir = os.path.join(scene_dir, f"agents_num_{str(args.num_agents)}_with_schedules")
    config_dirs = [config_dir, config_schedules_dir]
    for config_dir in config_dirs:
        config_config_path = os.path.join(config_dir, "config.json")
        config_place_metadata_path = os.path.join(config_dir, "place_metadata.json")
        config_building_metadata_path = os.path.join(config_dir, "building_metadata.json")
        if not os.path.exists(config_config_path) or not os.path.exists(config_place_metadata_path) or not os.path.exists(config_building_metadata_path):
            print(f"Configuration files for {args.scene} with {args.num_agents} agents do not exist in {config_dir}. Skipping offsetting.")
            continue
        config_config = json.load(open(config_config_path))
        config_place_metadata = json.load(open(config_place_metadata_path))
        config_building_metadata = json.load(open(config_building_metadata_path))
        json.dump(offset_places_xy_in_config_metadata(config_config, 1000), open(os.path.join(config_dir, "config.json"), "w"), indent=4)
        json.dump(offset_places_xy_in_place_metadata(config_place_metadata, 1000), open(os.path.join(config_dir, "place_metadata.json"), "w"), indent=4)
        json.dump(offset_places_xy_in_building_metadata(config_building_metadata, 1000), open(os.path.join(config_dir, "building_metadata.json"), "w"), indent=4)
        agent_names = config_config["agent_names"]
        for agent_name in agent_names:
            knowledge = json.load(open(os.path.join(config_dir, agent_name, f"seed_knowledge.json")))
            json.dump(offset_places_xy_in_knowledge(knowledge, 1000), open(os.path.join(config_dir, agent_name, f"seed_knowledge.json"), "w"), indent=4)
    print(f"Indoor place xy offsetting for {args.scene} completed.")