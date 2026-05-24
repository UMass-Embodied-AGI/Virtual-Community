import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

def _extract_semantic(prim, extract_from_children=True):
    for attr in prim.GetAuthoredAttributes():
        if attr.GetBaseName() == "semanticData":
            if attr.HasAuthoredValue():
                return attr.Get()

    if extract_from_children:
        for child in prim.GetChildren():
            val = _extract_semantic(child, False)
            if val:
                return val
    return None

def parse_instance_usd(path):
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.Open(path)
    xform_cache = UsdGeom.XformCache()
    instance_lst = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        if len(prim.GetPrimStack()) <= 1:
            continue

        matrix = np.array(xform_cache.GetLocalToWorldTransform(prim)).T
        instance_spec = prim.GetPrimStack()[-1]
        layer_id = instance_spec.layer.identifier
        target = prim.GetPrototype() if prim.IsInstance() else prim
        semantic = _extract_semantic(target)
        if semantic and "/" in semantic:
            semantic = semantic.split("/", 1)[0]
        if semantic in ['plant', 'person']:
            continue
        typ = 'object'
        if any([st in str(prim.GetPath()) for st in ['__default_setting', 'lights', 'Base']]):
            typ = 'structure'

        bbox_min = None
        bbox_max = None
        try:
            imageable = UsdGeom.Imageable(prim)
            time = Usd.TimeCode.Default()
            bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
            bound_range = bound.ComputeAlignedBox()
            bbox_min = np.array(bound_range.min) * 0.01
            bbox_max = np.array(bound_range.max) * 0.01
        except Exception as e:
            bbox_min, bbox_max = None, None
        instance_lst.append({
            'matrix': matrix,
            'layer_id': layer_id,
            'semantic': semantic,
            'typ': typ,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'prim_path': str(prim.GetPath())
        })

    return instance_lst

def compute_support_relationships(objects):
    import bisect
    support_map = {}
    n = len(objects)
    if n == 0:
        return support_map

    bbox_min = np.array([obj["bbox_min"] for obj in objects])
    bbox_max = np.array([obj["bbox_max"] for obj in objects])
    ax1, ax2, up_index = (0, 1, 2)

    height_tol = 1e-2
    overlap_threshold = 0.5

    indices_by_bottom = sorted(range(n), key=lambda i: bbox_min[i, up_index])
    indices_by_top = sorted(range(n), key=lambda i: bbox_max[i, up_index])
    tops_sorted = bbox_max[indices_by_top, up_index]

    for i in range(n):
        support_map = {}
        for idx in range(n):
            bottom = bbox_min[idx, up_index]
            top = bbox_max[idx, up_index]
            min1_x, min1_y = bbox_min[idx, ax1], bbox_min[idx, ax2]
            max1_x, max1_y = bbox_max[idx, ax1], bbox_max[idx, ax2]
            area1 = (max1_x - min1_x) * (max1_y - min1_y)

            low = bottom - height_tol
            high = bottom + height_tol
            j_low = bisect.bisect_left(tops_sorted, low)
            support_candidate_index = None
            best_top_height = -float('inf')
            for j_sorted in range(j_low, len(indices_by_top)):
                j = indices_by_top[j_sorted]
                top_j = bbox_max[j, up_index]
                if top_j > high:
                    break
                if j == idx:
                    continue
                if top_j <= bottom + height_tol and top_j >= bottom - height_tol:
                    min2_x, min2_y = bbox_min[j, ax1], bbox_min[j, ax2]
                    max2_x, max2_y = bbox_max[j, ax1], bbox_max[j, ax2]
                    overlap_x = min(max1_x, max2_x) - max(min1_x, min2_x)
                    overlap_y = min(max1_y, max2_y) - max(min1_y, min2_y)
                    if overlap_x > 0 and overlap_y > 0:
                        overlap_area = overlap_x * overlap_y
                        area2 = (max2_x - min2_x) * (max2_y - min2_y)
                        smaller_area = min(area1, area2) if area1 > 0 and area2 > 0 else area1
                        if smaller_area == 0:
                            continue
                        overlap_frac = overlap_area / smaller_area
                        if overlap_frac >= overlap_threshold:
                            if top_j > best_top_height:
                                support_candidate_index = j
                                best_top_height = top_j
            if support_candidate_index is not None:
                base_idx = support_candidate_index
                if base_idx not in support_map:
                    support_map[base_idx] = []
                support_map[base_idx].append(idx)
        return support_map

def sample_removal_per_category(objects, global_keep_ratio=0.5, category_keep_ratio=None):
    if category_keep_ratio is None:
        category_keep_ratio = {}

    n = len(objects)
    keep_flags = [True] * n

    pts = []
    for obj in objects:
        if obj["bbox_min"] is not None and obj["bbox_max"] is not None:
            p = (np.array(obj["bbox_min"]) + np.array(obj["bbox_max"])) * 0.5
        else:
            p = np.array(obj["matrix"][:3, 3])
        pts.append(p)

    idx_per_cat = {}
    for i, obj in enumerate(objects):
        if obj["typ"] == "structure":
            continue
        cat = obj["semantic"] or "other"
        idx_per_cat.setdefault(cat, []).append(i)

    for cat, idxs in idx_per_cat.items():
        m = len(idxs)
        ratio = category_keep_ratio.get(cat, global_keep_ratio)
        keep_num = max(1, round(m * ratio))
        if keep_num >= m:
            continue

        coords = np.stack([pts[i] for i in idxs])

        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

        keep_local = []
        seed_local = int(np.random.randint(m))
        keep_local.append(seed_local)

        min_dist = dist_mat[seed_local].copy()
        while len(keep_local) < keep_num:
            next_local = int(np.argmax(min_dist))
            keep_local.append(next_local)
            min_dist = np.minimum(min_dist, dist_mat[next_local])

        keep_global_idx = [idxs[k] for k in keep_local]
        for i in idxs:
            if i not in keep_global_idx:
                keep_flags[i] = False

    return keep_flags

def enforce_physical_consistency(keep_flags, support_map):
    """
    Modify the keep_flags list in-place to enforce physical consistency:
    if an object is removed, all objects that sit on top of it (directly or indirectly) are also removed.
    """
    n = len(keep_flags)
    # Use a stack/queue to propagate removals
    to_process = []
    # Initialize the stack with any object that is marked False (removed)
    for idx, keep in enumerate(keep_flags):
        if not keep:
            to_process.append(idx)
    processed = set()
    while to_process:
        base_idx = to_process.pop()
        if base_idx in processed:
            continue
        processed.add(base_idx)
        # If this base object supports others, remove those as well
        if base_idx in support_map:
            for child_idx in support_map[base_idx]:
                if keep_flags[child_idx]:
                    # Child is currently marked to keep, but its support is removed -> remove it
                    keep_flags[child_idx] = False
                    to_process.append(child_idx)

def place_usd_scene_with_ratio(usd_file, scene, global_pos=np.zeros(3), load_objects=True):
    def T_decompose(T):
        pos = T[:3, 3]
        euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        scale = np.linalg.norm(T[:3, :3], axis=0)
        return pos, euler, scale

    instance_list = parse_instance_usd(usd_file)

    support_map = compute_support_relationships(instance_list)

    global_ratio = 0.5
    base_cost_map = {"pen": 0.5, "chair": 3.0,
                     "table": 20.0, "counter": 20.0,
                     "other": 2.0, "desk": 20.0}
    category_keep_ratio = {
        "door": 0.0,
        "refrigerator": 1.0,
        "tv": 1.0,
        "microwave": 1.0,
        "cabinet": 1.0,
        "table": 1.0,
        "teatable": 1.0,
        "desk": 1.0,
        "chair": 0.6,
        "couch": 1.0,
        "sofa_chair": 0.5,
        "shelf": 1.0,
        "bookshelf": 1.0,
        "stool": 0.6,
        "trashcan": 0.2,
        "curtain": 1.0,
        "laptop": 0.3,
        "lamp": 0.3,
        "mirror": 0.3,
        "shoppingtrolley": 0.3,
        "box": 0.3,
        "coffeemaker": 0.3,
        "monitor": 1.0,
        "fan": 0.3,
        "keyboard": 0.3,
        "pen": 0.1,
        "book": 0.1,
        "clock": 0.1,
        "bottle": 0.1,
        "cup": 0.1,
        "tray": 0.1,
        "pot": 0.1,
        "plate": 0.1,
        "decoration": 0.1,
        "picture": 0.5,
        "pillow": 0.4,
        "light": 0.5,
        "other": 0.4
    }

    keep_flags = sample_removal_per_category(
        instance_list,
        global_keep_ratio=0.6,
        category_keep_ratio=category_keep_ratio
    )
    enforce_physical_consistency(keep_flags, support_map)

    for idx, instance in enumerate(instance_list):
        if not keep_flags[idx]:
            continue

        semantic = instance['semantic']
        typ = instance['typ']
        matrix = instance['matrix']
        instance_path = instance['layer_id']

        if semantic in ['person', 'pen', 'other'] and typ != "structure":  # skipping some categories
            continue
        if not load_objects and typ != "structure":
            continue
        init_euler = [0., 0., 0.]
        init_pos = np.array([0., 0., 0.])
        init_scale = 0.01
        init_T = gs.utils.geom.trans_R_to_T(init_pos, R.from_euler("xyz", init_euler, degrees=True).as_matrix())
        init_T[:3, :3] *= init_scale

        final_transform = init_T @ matrix
        pos, euler, scale = T_decompose(final_transform)
        scene.add_entity(
            type=typ,
            name=semantic,
            morph=gs.morphs.Mesh(
                file=instance_path,
                pos=pos + global_pos,
                euler=euler,
                scale=scale,
                fixed=True,
                collision=False,
                decimate=True,
                convexify=False,
            )
        )
    return instance_list, keep_flags