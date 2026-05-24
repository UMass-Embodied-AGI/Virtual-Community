#include "volume_grid.h"
#include "a_star.h"
#include <cassert>

extern "C" {
    VolumeGrid *init_volume_grid(float voxel_res, int num_workers) {
        return new VolumeGrid(voxel_res, num_workers);
    }

    void volume_grid_add_frame(VolumeGrid *vg, uint8_t *rgb, float *depth, int *label, int w, int h, float fov, float *extrinsic) {
        vg->add_frame(rgb, depth, label, w, h, fov, extrinsic);
    }

    void volume_grid_insert_from_numpy(VolumeGrid *vg, float *p, uint8_t *c, int *l, int n) {
        vg->insert(p, c, l, n);
    }

    void free_volume_grid(VolumeGrid *vg) {
        delete vg;
    }

    size_t volume_grid_memory_size(VolumeGrid *vg) {
        return vg->memory_size();
    }

    size_t volume_grid_size(VolumeGrid *vg) {
        return vg->size();
    }

    void volume_grid_to_numpy(VolumeGrid *vg, float *p, uint8_t *c, int *l) {
        vg->get_all_points(p, c, l);
    }

    float volume_grid_get_z(VolumeGrid *vg, float x, float y, float radius) {
        int ix = vg->align(x), iy = vg->align(y);
        int iz = vg->get_height(ix, iy, radius / vg->get_voxel_res());
        return iz * vg->get_voxel_res();
    }

    int volume_grid_get_label(VolumeGrid *vg, float x, float y, float z, float radius) {
        int ix = vg->align(x), iy = vg->align(y), iz = vg->align(z);
        int label = vg->get_label(ix, iy, iz, radius / vg->get_voxel_res());
        return label;
    }

    void volume_grid_get_bound(VolumeGrid *vg, float *min, float *max) {
        int grid_min[3], grid_max[3];
        std::tie(grid_min[0], grid_min[1], grid_min[2], grid_max[0], grid_max[1], grid_max[2]) = vg->get_bound();
        for (int i = 0; i < 3; i++) {
            min[i] = vg->retrieve(grid_min[i]);
            max[i] = vg->retrieve(grid_max[i]);
        }
    }

    float volume_grid_get_overlap(VolumeGrid *self, VolumeGrid *other, float radius) {
        int sz = self->size(), osz = other->size();
        int radius_int = radius / self->get_voxel_res();
        return (float)self->get_overlap(other, radius_int) / sz;
    }

    void volume_grid_radius_denoise(VolumeGrid *vg, int min_points, float radius) {
        vg->radius_denoise(min_points, radius / vg->get_voxel_res());
    }

    float* navigate(VolumeGrid *vg, float *start, float *goal, int goal_size, int base, int *size, float *ref_path, int ref_size) {
        AStar::Point s = { vg->align(start[0]), vg->align(start[1]) };
        AStar::Point *goal_p = new AStar::Point[goal_size];
        for (int i = 0; i < goal_size; i++) {
            goal_p[i] = { vg->align(goal[2 * i]), vg->align(goal[2 * i + 1]) };
        }
        AStar ast(s, goal_p, goal_size, vg, base);
        if (ref_size != 0) {
            std::vector<AStar::Point> ref;
            for (int i = 0; i < ref_size; i++) {
                ref.emplace_back(AStar::Point { vg->align(ref_path[2 * i]), vg->align(ref_path[2 * i + 1]) });
            }
            int first_obs = ast.check_path(ref);
            if (first_obs < 0) {
                // can use the whole ref_path
                *size = ref_size;
                float *p = new float[2 * ref_size];
                memcpy(p, ref_path, 2 * ref_size * sizeof(float));
                delete[] goal_p;
                return p;
            } else if (first_obs > 5) {
                int start_id = first_obs - 5;
                // plan from first_obstacle - 5
                AStar::Point news = ref[start_id];
                AStar newast(news, goal_p, goal_size, vg, base);
                std::vector<AStar::Point> path = newast.search();
                *size = path.size() + start_id;
                float *p = new float[2 * *size];
                for (int i = 0; i < start_id; i++) {
                    p[2 * i] = vg->retrieve(ref[i].x);
                    p[2 * i + 1] = vg->retrieve(ref[i].y);
                }
                for (int i = 0; i < path.size(); i++) {
                    p[2 * (i + start_id)] = vg->retrieve(path[i].x);
                    p[2 * (i + start_id) + 1] = vg->retrieve(path[i].y);
                }
                delete[] goal_p;
                return p;
            }
        }
        // normal plan
        std::vector<AStar::Point> path = ast.search();
        *size = path.size();
        float *p = new float[2 * path.size()];
        for (int i = 0; i < path.size(); i++) {
            p[2 * i] = vg->retrieve(path[i].x);
            p[2 * i + 1] = vg->retrieve(path[i].y);
        }
        delete[] goal_p;
        return p;
    }

    uint8_t* get_occurancy_map(VolumeGrid *vg, int base, int *x_min, int *y_min, int *x_max, int *y_max) {
        int _;
        std::tie(*x_min, *y_min, _, *x_max, *y_max, _) = vg->get_bound();
        AStar::Point p = { *x_min, *y_min };
        AStar ast(p, &p, 1, vg, base);
        std::tie(*x_min, *y_min, *x_max, *y_max) = ast.get_bound();
        int w = *x_max - *x_min + 1, h = *y_max - *y_min + 1;
        uint8_t *ret = new uint8_t[w * h];
        for (int i = *x_min; i <= *x_max; i++) {
            for (int j = *y_min; j <= *y_max; j++) {
                ret[(j - *y_min) * w + (i - *x_min)] = ast.get_map(AStar::Point { i, j });
            }
        }
        return ret;
    }

    uint8_t has_obstacle(VolumeGrid *vg, int base, float *bbox) {
        AStar::Point b_min = { vg->align(bbox[0]), vg->align(bbox[1]) };
        AStar::Point b_max = { vg->align(bbox[2]), vg->align(bbox[3]) };
        AStar::Point goal[2] = { b_min, b_max };
        AStar ast(b_min, goal, 2, vg, base);
        b_min = ast.align(b_min);
        b_max = ast.align(b_max);
        for (int i = b_min.x; i <= b_max.x; i++) {
            for (int j = b_min.y; j <= b_max.y; j++) {
                if (ast.get_map(AStar::Point { i, j }) != AStar::MapEntry::ROAD) {
                    return 1;
                }
            }
        }
        return 0;
    }
}
