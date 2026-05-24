#pragma once

#include "volume_grid.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>

class AStar {
public:
    struct Point {
        int x, y;
        bool operator==(const Point &p) const {
            return x == p.x && y == p.y;
        }
    };

    AStar(const Point &start, const Point *goal, int goal_size, VolumeGrid *vg, int base) : vg(vg), base(base) {
        this->start = align(start);
        // build goal hull
        Point goal_min = {INT32_MAX, INT32_MAX}, goal_max = {INT32_MIN, INT32_MIN};
        int *goal_p = new int[goal_size * 2];
        for (int i = 0; i < goal_size; i++) {
            goal_p[2 * i] = floordiv(goal[i].x, base);
            goal_p[2 * i + 1] = floordiv(goal[i].y, base);
            goal_min.x = std::min(goal_min.x, goal[i].x);
            goal_min.y = std::min(goal_min.y, goal[i].y);
            goal_max.x = std::max(goal_max.x, goal[i].x);
            goal_max.y = std::max(goal_max.y, goal[i].y);
        }
        goal_hull = convex_hull(goal_p, goal_size, &hull_size);
        delete[] goal_p;

        int _;
        std::tie(x_min, y_min, _, x_max, y_max, _) = vg->get_bound();
        x_min = floordiv(std::min(x_min, std::min(start.x, goal_min.x)), base) - 10;
        x_max = floordiv(std::max(x_max, std::max(start.x, goal_max.x)), base) + 10;
        y_min = floordiv(std::min(y_min, std::min(start.y, goal_min.y)), base) - 10;
        y_max = floordiv(std::max(y_max, std::max(start.y, goal_max.y)), base) + 10;

        map = new MapEntry*[x_max - x_min + 1];
        closed = new bool*[x_max - x_min + 1];
        height = new int*[x_max - x_min + 1];
        for (int i = 0; i < x_max - x_min + 1; i++) {
            map[i] = new MapEntry[y_max - y_min + 1];
            closed[i] = new bool[y_max - y_min + 1];
            height[i] = new int[y_max - y_min + 1];
            for (int j = 0; j < y_max - y_min + 1; j++) {
                map[i][j] = NONE;
                closed[i][j] = false;
            }
        }
    }

    ~AStar() {
        for (int i = 0; i < x_max - x_min + 1; i++) {
            delete[] map[i];
            delete[] closed[i];
            delete[] height[i];
        }
        delete[] map;
        delete[] closed;
        delete[] height;
    }

    enum MapEntry : uint8_t {
        NONE,
        UNKNOWN,
        OBSTACLE,
        ROAD,
    };

    MapEntry get_map(const Point &p) {
        MapEntry &m = map[p.x - x_min][p.y - y_min];
        if (m == NONE) {
            Point org = retrieve(p);
            int h = vg->get_height(org.x, org.y, base / 2 + 1); // +1 is important! to avoid cases that the building is at the edge of the grid
            height[p.x - x_min][p.y - y_min] = h;
            if (h == INT32_MIN) {
                return m = UNKNOWN;
            } else if (h == INT32_MAX) {
                return m = OBSTACLE;
            }
            // int grid_num = int(MAX_HEIGHT / vg->get_voxel_res());
            int grid_num = int(100 / vg->get_voxel_res()); // ban going under buildings
            for (int i = 0; i < base; i++)
            for (int j = 0; j < base; j++) {
                int ox = p.x * base + i, oy = p.y * base + j;
                Z_Array *arr = vg->get_z(ox, oy);
                if (arr == nullptr) {
                    continue;
                } else {
                    int obs_grid = arr->valid.count(h + 1, h + grid_num);
                    if (obs_grid > int(0.5 / vg->get_voxel_res())) {
                        return m = OBSTACLE;
                    }
                }
            }
            return m = ROAD;
        }
        return m;
    }

    auto get_bound() const {
        return std::tie(x_min, y_min, x_max, y_max);
    }

    Point align(const Point &p) {
        return {floordiv(p.x, base), floordiv(p.y, base)};
    }

    Point retrieve(const Point &p) {
        return {p.x * base + base / 2, p.y * base + base / 2};
    }

    std::vector<Point> search();
    
    int check_path(const std::vector<Point> &path) {
        for (size_t i = 0; i < path.size(); i++) {
            Point cp = align(path[i]);
            if (cp.x < x_min || cp.x > x_max || cp.y < y_min || cp.y > y_max || get_map(cp) == OBSTACLE) {
                return i;
            }
        }
        return -1;
    }

private:
    struct Node {
        Point p;
        int g, h;
        Node *parent;
        Node(const Point &p, int g, int h, Node *parent) : p(p), g(g), h(h), parent(parent) {}
        bool operator<(const Node &n) const {
            int f = g + h, nf = n.g + n.h;
            return f == nf ? (p.x == n.p.x ? p.y > n.p.y : p.x > n.p.x) : f > nf;
        }
    };
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    Point start;
    int *goal_hull;
    VolumeGrid *vg;
    int base, hull_size;
    int x_min, x_max, y_min, y_max;

    MapEntry **map;
    int **height;
    bool **closed;

    bool &get_closed(const Point &p) {
        return closed[p.x - x_min][p.y - y_min];
    }

    int heuristic(const Point &p) {
        // satisfy triangle inequality
        int ap[2] = {p.x, p.y};
        return round(dist_to_hull(ap, goal_hull, hull_size));
    }

    int has_obstacle_near(const Point &p) {
        int mn_dist = INT32_MAX;
        for (int nx = -2; nx <= 2; nx++) {
            for (int ny = -2; ny <= 2; ny++) {
                Point q = {p.x + nx, p.y + ny};
                if (q.x < x_min || q.x > x_max || q.y < y_min || q.y > y_max) {
                    continue;
                }
                if (get_map(q) == OBSTACLE)
                    mn_dist = std::min(mn_dist, std::max(std::abs(nx), std::abs(ny)));
            }
        }
        return mn_dist;
    }
};

std::vector<AStar::Point> AStar::search() {
    std::vector<AStar::Point> path;
    std::priority_queue<AStar::Node> open;
    std::vector<AStar::Node*> node_list;
    open.push(Node(start, 0, heuristic(start), nullptr));
    while (!open.empty()) {
        Node *current = new Node(open.top()); open.pop();
        if (current->h == 0) {
            for (Node *n = current; n->parent != nullptr; n = n->parent) {
                path.emplace_back(retrieve(n->p));
//                if (n->parent->parent != nullptr && get_map(n->p) == UNKNOWN) {
//                    path.clear(); // truncate path if unknown
//                }
            }
            path.push_back(retrieve(start));
            std::reverse(path.begin(), path.end());
            for (Node *n : node_list) {
                delete n;
            }
            delete current;
            return path;
        }
        bool &c = get_closed(current->p);
        if (c) {
            delete current;
            continue;
        }
        c = true;
        node_list.emplace_back(current);
        MapEntry current_entry = get_map(current->p);
        int h = height[current->p.x - x_min][current->p.y - y_min];
        int current_near_obs = has_obstacle_near(current->p);
        for (int i = 0; i < 4; i++) {
            Point np = {current->p.x + dx[i], current->p.y + dy[i]};
            if (np.x < x_min || np.x > x_max || np.y < y_min || np.y > y_max || get_closed(np)) {
                continue;
            }
            MapEntry entry = get_map(np);
            int nh = height[np.x - x_min][np.y - y_min];
            int n_near_obs = has_obstacle_near(np);
            int extra = 100 / (n_near_obs + 1);
            if (current_near_obs <= n_near_obs) {
                // permit move to any direction which farther from the obstacles if current point is near obstacle
                open.emplace(np, current->g + 1 + extra, heuristic(np), current);
                continue;
            }
            if (entry == OBSTACLE || n_near_obs <= 1) {
                // ban move to any direction which near obstacle
                continue;
            }
            if (entry == UNKNOWN) {
                open.emplace(np, current->g + 5 + extra, heuristic(np), current);
            } else if (current_entry == UNKNOWN) {
                open.emplace(np, current->g + 5 + extra, heuristic(np), current);
            } else if (current_entry == ROAD) {
                if (std::abs(nh - h) <= base)
                    open.emplace(np, current->g + 1 + extra, heuristic(np), current);
                // else if (std::abs(nh - h) <= int(MAX_HEIGHT / 2 / vg->get_voxel_res()))
                //     open.emplace(np, current->g + 20, heuristic(np), current);
            }
        }
    }
    Node *min_goal = nullptr;
    for (Node *n : node_list) {
        bool replace = false;
        if (min_goal == nullptr) {
            replace = true;
        } else if (n->h != min_goal->h) {
            replace = n->h < min_goal->h;
        } else if (n->g != min_goal->g) {
            replace = n->g < min_goal->g;
        } else {
            replace = n->p.x < min_goal->p.x || (n->p.x == min_goal->p.x && n->p.y < min_goal->p.y);
        }
        if (replace) min_goal = n;
    }
    for (Node *n = min_goal; n->parent != nullptr; n = n->parent) {
        path.emplace_back(retrieve(n->p));
    }
    path.push_back(retrieve(start));
    std::reverse(path.begin(), path.end());
    for (Node *n : node_list) {
        delete n;
    }
    return path;
}