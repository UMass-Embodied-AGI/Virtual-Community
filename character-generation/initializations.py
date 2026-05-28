init_scratch = {
    # "vision_r": 8,
    # "att_bandwidth": 8,
    # "retention": 8,
    "name": "", 
    "first_name": "", 
    "last_name": "", 
    "age": "", 
    "innate": "", 
    "learned": "", 
    "currently": "",
    "lifestyle": "",
    "groups": "",
    "daily_requirement": "",
    "curr_time": None,
    "start_time": None,
    "living_place": "",
    "react_freq": 60,
    # "concept_forget": 100,
    # "daily_reflection_time": 180,
    # "daily_reflection_size": 5,
    # "overlap_reflect_th": 4,
    # "kw_strg_event_reflect_th": 10,
    # "kw_strg_thought_reflect_th": 9,
    # "recency_w": 1,
    # "relevance_w": 1,
    # "importance_w": 1,
    # "recency_decay": 0.995,
    "importance_trigger_max": 250,
    "importance_trigger_curr": 250,
    "importance_ele_n": 0,
    # "thought_count": 5,
    "daily_plan": [],
    "hourly_schedule": [],
    "held_objects": [None, None],
    "act_address": None, 
    "act_start_time": None, 
    "act_duration": None, 
    "act_description": None,
    # "act_pronunciatio": None,
    "act_event": "", 
    # "act_obj_description": None,
    # "act_obj_pronunciatio": None,
    # "act_obj_event": [None, None, None],
    "chat": None,
    "chatting_with_buffer": {}, 
    "chatting_end_time": None,
    # "act_path_set": False,
    # "planned_path": []
}

config = {
    "sim_name": "mit_agents_num_5",
    "agent_names": [
        "Brian Carter",
        "Elon Musk",
        "Chad Thompson",
        "Joshua Tenenbaum",
        "Kate Novak"
    ],
    "agent_poses": [
    ],
    "agent_infos": [
        {
            "cash": 700,
            "outdoor_pose": [
                318.4800900758187,
                342.48944091796875,
                3.0000041345912902,
                0.0,
                0.0,
                0.0
            ],
            "current_building": "Proto",
            "current_place": "Proto Kendall Square"
        },

    ],
    "locator_colors": [
        "red",
        "green",
        "blue",
        "orange",
        "purple"
    ],
    "num_agents": 5,
    "start_date": "February 1, 2023",
    "curr_time": "February 1, 2023, 06:00:00",
    "sec_per_step": 1,
    "agent_skins": [
        "avatars/models/mixamo_Brian_merge.glb",
        "avatars/models/celebrity_Elon_Musk.glb",
        "avatars/models/mixamo_Chad_merge.glb",
        "avatars/models/celebrity_Joshua_Tenenbaum.glb",
        "avatars/models/mixamo_Kate_merge.glb"
    ],
    "step": 0
}