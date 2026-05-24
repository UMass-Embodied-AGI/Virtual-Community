import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.colors as mcolors

from utils import merge_step_files

def map_lang_colors_to_rgb(lang_colors):
    return [mcolors.to_rgb(c) for c in lang_colors]


def build_lookup(root_dir: str, agent_names, num_steps, tag: str):
    lookup_path = os.path.join(root_dir, tag,
                               f"{tag}_check_exists_lookup.json")
    if os.path.exists(lookup_path):
        return lookup_path  # nothing to do

    lookup = defaultdict(list)
    for name in tqdm(agent_names, desc=f"Scanning {tag} frames"):
        for i in range(num_steps):
            img_path = os.path.join(root_dir, tag, name, f"rgb_{i:06d}.png")
            if os.path.exists(img_path):
                lookup[name].append(i)

    os.makedirs(os.path.dirname(lookup_path), exist_ok=True)
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, separators=(",", ":"))
    print(f"Generated: {lookup_path}")
    return lookup_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default="output")
    parser.add_argument("--scene", default="newyork")
    parser.add_argument("--config", default="agents_num_15")
    parser.add_argument("-d", "--data_dir", type=str,
                        help="Shortcut: point to an existing "
                             "`output/<scene>_<config>/<agent_type>` folder")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    # Leaving --type so the script can still build ego-only demos
    parser.add_argument("--type", choices=["ego", "tp"], default="ego",
                        help="Main view the HTML will default to")
    args = parser.parse_args()

    if args.data_dir:
        args.data_dir = args.data_dir.rstrip("/\\")
        args.output_dir = args.data_dir


        # infer scene/agent_type only to resolve asset paths
        SCENE_TAGS = ("NY", "DETROIT", "NEWYORK")          # add more if needed
        args.scene = "newyork"                              # safe fallback

        for part in args.data_dir.replace("\\", "/").split("/"):
            for tag in SCENE_TAGS:
                if part.upper().startswith(tag):
                    args.scene = tag
                    break
            else:
                continue      
            break            
    else:
        args.output_dir = os.path.join(
            args.output_dir, f"{args.scene}_{args.config}", "tour_agent")

    os.makedirs(os.path.join(args.output_dir, "global"), exist_ok=True)

    # config
    cfg_path = os.path.join(args.output_dir, "curr_sim", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    num_steps = min(cfg["step"], args.steps) if args.steps else cfg["step"]
    agent_names = cfg["agent_names"]
    cfg["locator_colors_rgb"] = map_lang_colors_to_rgb(cfg["locator_colors"])

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Config updated with RGB colours  ->  {cfg_path}")

    # merge step files
    if not args.skip_merge:
        merge_step_files(os.path.join(args.output_dir, "steps"),
                         agent_names, num_steps, overwrite=args.overwrite)

    # build ego-check lookup
    build_lookup(args.output_dir, agent_names, num_steps, args.type)
    build_lookup(args.output_dir, agent_names, num_steps,
                 "tp" if args.type == "ego" else "ego")

    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    # Convert filesystem paths to server-root-relative URLs for python -m http.server
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def web_path(p: str) -> str:
        return "/" + os.path.relpath(os.path.abspath(p), repo_root).replace(os.sep, "/")

    output_dir_abs = os.path.abspath(args.output_dir)

    substitutions = {
        "$OutputFolderPath$": web_path(output_dir_abs),
        "$CurrSimFolderPath$": web_path(os.path.join(output_dir_abs, "curr_sim")),
        "$ConfigPath$": web_path(cfg_path),
        "$TpegoFolderPath$": web_path(os.path.join(output_dir_abs, "ego")),
        "$tpFolderPath$": web_path(os.path.join(output_dir_abs, "tp")),
        "$StepsFolderPath$": web_path(os.path.join(output_dir_abs, "steps")),
        "$AvatarImgsPath$": web_path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/imgs/avatars")),
        "$GlobalCameraParameterPath$": web_path(os.path.join(os.path.dirname(os.path.dirname(__file__)), f"assets/scenes/{args.scene}/global_cam_parameters.json")),
        "$GlobalImagePath$": web_path(os.path.join(os.path.dirname(os.path.dirname(__file__)), f"assets/scenes/{args.scene}/global.png")),
        "$FPS$": str(args.fps),
    }
    for key, val in substitutions.items():
        html = html.replace(key, val)

    out_html = os.path.join(args.output_dir, "demo.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Demo HTML written to  -> {out_html}\n")

    print("Serve with:")
    print("    python -m http.server")
    print(f"and open:")
    print(
        f"    http://localhost:8000{web_path(output_dir_abs)}/demo.html")


if __name__ == "__main__":
    main()
