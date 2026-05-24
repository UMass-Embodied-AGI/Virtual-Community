#pragma once

#include <stdio.h>
#include <string.h>
#include <random>
#include <time.h>
#include <thread>
#include <algorithm>
#include <cassert>
#include <map>
#include "utils.h"

template<typename T> class DynamicArray {
    friend class VolumeGrid;
    T *arr;
    int mn, mx;
public:
    DynamicArray() {
        arr = nullptr;
    }
    DynamicArray(int mn, int mx) : mn(mn), mx(mx) {
        arr = new T[mx - mn + 1];
    }
    DynamicArray<T>& operator=(DynamicArray<T> &&other) {
        arr = other.arr;
        mn = other.mn;
        mx = other.mx;
        other.arr = nullptr;
        return *this;
    }
    T& operator[](int i) {
        if (arr != nullptr && i >= mn && i <= mx) {
            return arr[i - mn];
        }
        if (arr == nullptr) {
            mn = mx = i;
            arr = new T[1];
            memset(arr, 0, sizeof(T));
            return arr[0];
        }
        T *new_arr;
        if (i < mn) {
            int new_mn = i;
            new_arr = new T[mx - new_mn + 1];
            memset(new_arr, 0, (mn - new_mn) * sizeof(T));
            memcpy(new_arr + (mn - new_mn), arr, (mx - mn + 1) * sizeof(T));
            memset(arr, 0, (mx - mn + 1) * sizeof(T));
            // for (int j = mn; j <= mx; j++) {
            //     new_arr[j - new_mn] = std::move(arr[j - mn]);
            // }
            mn = new_mn;
        } else {
            int new_mx = i;
            new_arr = new T[new_mx - mn + 1];
            memset(new_arr + mx - mn + 1, 0, (new_mx - mx) * sizeof(T));
            memcpy(new_arr, arr, (mx - mn + 1) * sizeof(T));
            memset(arr, 0, (mx - mn + 1) * sizeof(T));
            // for (int j = mn; j <= mx; j++) {
            //     new_arr[j - mn] = std::move(arr[j - mn]);
            // }
            mx = new_mx;
        }
        delete[] arr;
        arr = new_arr;
        return arr[i - mn];
    }
    ~DynamicArray() {
        if (arr != nullptr)
            delete[] arr;
    }
};

template<> class DynamicArray<bool> {
    static const int B_SIZE = 32;
    friend class VolumeGrid;
    uint32_t *arr;
    int mn, mx;
    inline static int blockid(int i) {
        return floordiv(i, B_SIZE);
    }
    void extend(int b) {
        if (arr == nullptr) {
            mn = mx = b;
            arr = new uint32_t[1];
            arr[0] = 0;
            return;
        }
        uint32_t *new_arr;
        if (b < mn) {
            int new_mn = b;
            new_arr = new uint32_t[mx - new_mn + 1];
            memset(new_arr, 0, (mn - new_mn) * sizeof(uint32_t));
            memcpy(new_arr + (mn - new_mn), arr, (mx - mn + 1) * sizeof(uint32_t));
            mn = new_mn;
        } else if (b > mx) {
            int new_mx = b;
            new_arr = new uint32_t[new_mx - mn + 1];
            memset(new_arr + mx - mn + 1, 0, (new_mx - mx) * sizeof(uint32_t));
            memcpy(new_arr, arr, (mx - mn + 1) * sizeof(uint32_t));
            mx = new_mx;
        } else {
            return;
        }
        delete[] arr;
        arr = new_arr;
    }
public:
    DynamicArray() {
        arr = nullptr;
    }
    DynamicArray(int mn, int mx) : mn(blockid(mn)), mx(blockid(mx)) {
        arr = new uint32_t[this->mx - this->mn + 1];
    }
    DynamicArray<bool>& operator=(DynamicArray<bool> &&other) {
        arr = other.arr;
        mn = other.mn;
        mx = other.mx;
        other.arr = nullptr;
        return *this;
    }
    int set(int i) {
        int b = blockid(i), w = i - b * B_SIZE;
        if (arr == nullptr || b < mn || b > mx) {
            extend(b);
        }
        int org = (arr[b - mn] >> w) & 1;
        arr[b - mn] |= 1UL << w;
        return org;
    }
    int reset(int i) {
        int b = blockid(i), w = i - b * B_SIZE;
        if (arr == nullptr || b < mn || b > mx) {
            extend(b);
        }
        int org = (arr[b - mn] >> w) & 1;
        arr[b - mn] &= ~(1UL << w);
        return org;
    }
    void reset_all() {
        if (arr == nullptr)
            return;
        memset(arr, 0, (mx - mn + 1) * sizeof(uint32_t));
    }
    int test(int i) {
        int b = blockid(i), w = i - b * B_SIZE;
        if (arr == nullptr || b < mn || b > mx)
            return 0;
        return (arr[b - mn] >> w) & 1;
    }
    int test(int l, int r) {
        if (arr == nullptr || l >= r)
            return 0;
        l = std::max(mn * B_SIZE, l);
        r = std::min((mx + 1) * B_SIZE, r);
        int bl = blockid(l), br = blockid(r);
        uint32_t ml = arr[bl - mn] & ~((1UL << (l - bl * B_SIZE)) - 1);
        uint32_t mr = br > mx ? 0 : arr[br - mn] & ((1UL << (r - br * B_SIZE)) - 1);
        int c = 0;
        if (bl == br) {
            return (ml & mr) != 0;
        } else if (bl < br) {
            if (ml != 0 || mr != 0) return 1;
            for (int i = bl + 1; i < br; i++) {
                if (arr[i - mn] != 0) {
                    return 1;
                }
            }
        }
        return 0;
    }
    int get_first() {
        if (arr == nullptr)
            return 0;
        for (int i = 0; i <= mx - mn; i++) {
            if (arr[i] != 0) {
                return (i + mn) * B_SIZE + __builtin_ctz(arr[i]);
            }
        }
        return (mx + 1) * B_SIZE;
    }
    int get_next(int i) {
        if (arr == nullptr)
            return -1;
        int b = blockid(i), w = i - b * B_SIZE;
        uint32_t t = arr[b - mn] >> (w + 1);
        if (w + 1 < B_SIZE && t != 0) {
            return i + __builtin_ctz(t) + 1;
        }
        for (int j = b - mn + 1; j <= mx - mn; j++) {
            if (arr[j] != 0) {
                return (j + mn) * B_SIZE + __builtin_ctz(arr[j]);
            }
        }
        return (mx + 1) * B_SIZE;
    }
    int get_last() {
        if (arr == nullptr)
            return 0;
        for (int i = mx - mn; i >= 0; i--) {
            if (arr[i] != 0) {
                return (i + mn) * B_SIZE + 31 - __builtin_clz(arr[i]);
            }
        }
        return mn * B_SIZE - 1;
    }
    int count() {
        if (arr == nullptr)
            return 0;
        int c = 0;
        for (int i = 0; i <= mx - mn; i++) {
            c += __builtin_popcount(arr[i]);
        }
        return c;
    }
    int count(int l, int r) {
        if (arr == nullptr || l >= r)
            return 0;
        l = std::max(mn * B_SIZE, l);
        r = std::min((mx + 1) * B_SIZE, r);
        int bl = blockid(l), br = blockid(r);
        uint32_t ml = arr[bl - mn] & ~((1UL << (l - bl * B_SIZE)) - 1);
        uint32_t mr = br > mx ? 0 : arr[br - mn] & ((1UL << (r - br * B_SIZE)) - 1);
        int c = 0;
        if (bl == br) {
            c = __builtin_popcount(ml & mr);
        } else if (bl < br) {
            c += __builtin_popcount(ml);
            for (int i = bl + 1; i < br; i++) {
                c += __builtin_popcount(arr[i - mn]);
            }
            c += __builtin_popcount(mr);
        }
        return c;
    }
    ~DynamicArray() {
        if (arr != nullptr)
            delete[] arr;
    }
};

struct Voxel {
    int z;
    uint8_t r, g, b;
    int label;
};

class Z_Array {
    friend class VolumeGrid;
    int sz;
public:
    DynamicArray<bool> valid;
    Voxel *voxels;
    int n;
    Z_Array() : voxels(nullptr), n(0), sz(0) {}
    void insert(const Voxel &v) {
        if (valid.set(v.z)) {
            for (int i = 0; i < n; i++) {
                if (voxels[i].z == v.z) {
                    voxels[i] = v;
                    return;
                }
            }
            throw "Invalid";
        }
        if (n < sz) {
            voxels[n++] = v;
        } else {
            sz = sz == 0 ? 1 : sz * 2;
            Voxel *new_voxels = new Voxel[sz];
            if (voxels != nullptr) {
                memcpy(new_voxels, voxels, n * sizeof(Voxel));
                delete[] voxels;
            }
            voxels = new_voxels;
            voxels[n++] = v;
        }
    }
    ~Z_Array() {
        if (voxels != nullptr)
            delete[] voxels;
    }
};

class VolumeGrid {
    const static int X_MAX = 65536;
    DynamicArray<Z_Array*> data[X_MAX * 2];
    float voxel_res;
    int num_workers;
    int x_max, x_min, y_max, y_min, z_max, z_min;
    std::tuple<int, int, int, int, int, int> bound;

    auto _get_bound() const {
        int x_min, x_max, y_min, y_max, z_min, z_max;
        x_min = y_min = z_min = INT32_MAX;
        x_max = y_max = z_max = INT32_MIN;
        for (int i = 0; i < X_MAX * 2; i++) {
            if (data[i].arr == nullptr)
                continue;
            for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                Z_Array *&arr = data[i].arr[j];
                if (arr != nullptr && arr->n > 0) {
                    x_min = std::min(x_min, i - X_MAX);
                    x_max = std::max(x_max, i - X_MAX);
                    y_min = std::min(y_min, j + data[i].mn);
                    y_max = std::max(y_max, j + data[i].mn);
                    z_min = std::min(z_min, arr->valid.get_first());
                    z_max = std::max(z_max, arr->valid.get_last());
                }
            }
        }
        return std::make_tuple(x_min, y_min, z_min, x_max, y_max, z_max);
    }
public:
    VolumeGrid(float voxel_res, int num_workers) :
        voxel_res(voxel_res), num_workers(num_workers) {
        x_max = x_min = y_max = y_min = z_max = z_min = 0;
    }

    int align(float x) {
        return floor(x / voxel_res);
    }

    float retrieve(int x) {
        return (x + 0.5) * voxel_res;
    }

    void add_frame(uint8_t *rgb, float *depth, int *label, int w, int h, float fov, float *extrinsic) {
        int pcd_size = 0;
        float max_depth = 0;
        for (int i = 0; i < w * h; i++) {
            if (label[i] >= -1) {
                pcd_size++;
                max_depth = std::max(max_depth, depth[i]);
            }
        }
        // remove voxels that are disappeared using depth test
        int cur_x = align(extrinsic[3]), cur_y = align(extrinsic[7]);
        int radius = ceil(max_depth * 1.1 / voxel_res);
        float tan_fov = std::tan(fov / 180 * PI / 2) * w / h;
        float fx = w / 2.0 / tan_fov, fy = h / 2.0 / tan_fov;
        float cx = w / 2.0, cy = h / 2.0;
        
        auto run = [&](int id) {
            for (int i = cur_x - radius; i <= cur_x + radius; i++) {
                if ((i + X_MAX) % num_workers != id)
                    continue;
                for (int j = cur_y - radius; j <= cur_y + radius; j++) {
                    Z_Array *arr = get_z(i, j);
                    if (arr == nullptr) {
                        continue;
                    }
                    int cnt = 0;
                    for (int k = 0; k < arr->n; k++) {
                        Voxel &v = arr->voxels[k];
                        float x = retrieve(i) - extrinsic[3], y = retrieve(j) - extrinsic[7], z = retrieve(v.z) - extrinsic[11];
                        float camera_x = extrinsic[0] * x + extrinsic[4] * y + extrinsic[8] * z;
                        float camera_y = extrinsic[1] * x + extrinsic[5] * y + extrinsic[9] * z;
                        float camera_z = -(extrinsic[2] * x + extrinsic[6] * y + extrinsic[10] * z);

                        int pixel_x = round(camera_x * fx / camera_z + cx);
                        int pixel_y = round(h - 1 - (camera_y * fy / camera_z + cy));

                        if (pixel_x >= 0 && pixel_x < w && pixel_y >= 0 && pixel_y < h && camera_z > 0) {
                            float dep = depth[pixel_y * w + pixel_x];
                            if (pixel_x > 0) dep = std::min(dep, depth[pixel_y * w + pixel_x - 1]);
                            if (pixel_x < w - 1) dep = std::min(dep, depth[pixel_y * w + pixel_x + 1]);
                            if (pixel_y > 0) dep = std::min(dep, depth[(pixel_y - 1) * w + pixel_x]);
                            if (pixel_y < h - 1) dep = std::min(dep, depth[(pixel_y + 1) * w + pixel_x]);
                            if (camera_z < dep) {
                                // this voxel disappears
                                arr->valid.reset(v.z);
                                continue;
                            }
                        }
                        arr->voxels[cnt++] = v;
                    }
                    arr->n = cnt;
                }
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; i++) {
            threads.push_back(std::thread(run, i));
        }
        for (int i = 0; i < num_workers; i++) {
            threads[i].join();
        }
        
        // add new voxels
        float *p = new float[3 * pcd_size];
        uint8_t *c = new uint8_t[3 * pcd_size];
        int *l = new int[pcd_size];
        image_to_pcd(rgb, depth, label, p, c, l, w, h, fov, extrinsic);
        insert(p, c, l, pcd_size);
        delete[] p;
        delete[] c;
        delete[] l;
    }

    void insert(float *p, uint8_t *c, int *l, int n) {
        auto run = [&](int id) {
            for (int i = 0; i < n; i++) {
                float px = p[3 * i] / voxel_res, py = p[3 * i + 1] / voxel_res, pz = p[3 * i + 2] / voxel_res;
                int x = floor(px), y = floor(py), z = floor(pz);
                if (id == 0) {
                    x_max = std::max(x_max, x);
                    x_min = std::min(x_min, x);
                    y_max = std::max(y_max, y);
                    y_min = std::min(y_min, y);
                    z_max = std::max(z_max, z);
                    z_min = std::min(z_min, z);
                }
                if ((x + X_MAX) % num_workers != id)
                    continue;
                Z_Array *&arr = data[x + X_MAX][y];
                if (arr == nullptr) {
                    arr = new Z_Array();
                }
                arr->insert(Voxel{z, c[3 * i], c[3 * i + 1], c[3 * i + 2], l[i]});
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; i++) {
            threads.push_back(std::thread(run, i));
        }
        for (int i = 0; i < num_workers; i++) {
            threads[i].join();
        }
        this->bound = _get_bound();
    }

    void get_all_points(float *p, uint8_t *c, int *l) {
        int n = 0;
        for (int i = 0; i < X_MAX * 2; i++) {
            if (data[i].arr == nullptr)
                continue;
            for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                Z_Array *&arr = data[i].arr[j];
                if (arr != nullptr) {
                    for (int k = 0; k < arr->n; k++) {
                        const Voxel &v = arr->voxels[k];
                        p[3 * n] = retrieve(i - X_MAX);
                        p[3 * n + 1] = retrieve(j + data[i].mn);
                        p[3 * n + 2] = retrieve(v.z);
                        c[3 * n] = v.r;
                        c[3 * n + 1] = v.g;
                        c[3 * n + 2] = v.b;
                        l[n] = v.label;
                        n++;
                    }
                }
            }
        }
    }

    Z_Array *get_z(int x, int y) {
        if (x < -X_MAX || x >= X_MAX)
            return nullptr;
        auto &d = data[x + X_MAX];
        if (d.arr == nullptr || d.mn > y || d.mx < y)
            return nullptr;
        Z_Array *ret = d.arr[y - d.mn];
        if (ret == nullptr || ret->n == 0) {
            return nullptr;
        }
        return d.arr[y - d.mn];
    }

    size_t size() const {
        size_t size = 0;
        for (int i = 0; i < X_MAX * 2; i++) {
            if (data[i].arr == nullptr)
                continue;
            for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                Z_Array *&arr = data[i].arr[j];
                if (arr != nullptr) {
                    size += arr->n;
                }
            }
        }
        return size;
    }

    size_t memory_size() const {
        size_t size = sizeof(data);
        for (int i = 0; i < X_MAX * 2; i++) {
            if (data[i].arr == nullptr)
                continue;
            size += (data[i].mx - data[i].mn + 1) * sizeof(data[i].arr[0]);
            for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                Z_Array *&arr = data[i].arr[j];
                if (arr != nullptr) {
                    size += (arr->valid.mx - arr->valid.mn + 1) * sizeof(arr->valid.arr[0]);
                    size += arr->sz * sizeof(Voxel);
                }
            }
        }
        return size;
    }

    int get_height(int x, int y, int radius) {
        std::vector<int> zs;
        std::vector<Z_Array*> z_arrs;
        int z_mn = INT32_MAX, z_mx = INT32_MIN;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                Z_Array *arr = get_z(x + i, y + j);
                if (arr != nullptr) {
                    z_arrs.push_back(arr);
                    z_mn = std::min(z_mn, arr->valid.mn);
                    z_mx = std::max(z_mx, arr->valid.mx);
                }
            }
        }
        uint32_t *valid = new uint32_t[z_mx - z_mn + 1];
        for (int i = 0; i <= z_mx - z_mn; i++) {
            valid[i] = 0;
        }
        for (Z_Array *arr : z_arrs) {
            for (int i = arr->valid.mn; i <= arr->valid.mx; i++) {
                valid[i - z_mn] |= arr->valid.arr[i - arr->valid.mn];
            }
        }
        int mn = INT32_MAX;
        for (int i = z_mn; i <= z_mx; i++) {
            if (valid[i - z_mn] != 0) {
                mn = i * 32 + __builtin_ctz(valid[i - z_mn]);
                break;
            }
        }
        if (mn == INT32_MAX) {
            delete[] valid;
            // unknown
            return INT32_MIN;
        }
        while (true) {
            int nxt;
            int b = floordiv(mn, 32), w = mn - b * 32;
            uint32_t t = valid[b - z_mn] >> (w + 1);
            if (w + 1 < 32 && t != 0) {
                nxt = mn + __builtin_ctz(t) + 1;
            } else {
                nxt = INT32_MAX;
                for (int j = b + 1; j <= z_mx; j++) {
                    if (valid[j - z_mn] != 0) {
                        nxt = j * 32 + __builtin_ctz(valid[j - z_mn]);
                        break;
                    }
                }
            }
            if (nxt > int(MAX_HEIGHT / voxel_res) + mn) {
                break;
            }
            mn = nxt;
        }
        delete[] valid;
        for (Z_Array *arr : z_arrs) {
            if (mn > arr->valid.get_first() + radius * 2) {
                // obstacle if height difference is too large
                return INT32_MAX;
            }
        }
        return mn;
    }

    int get_label(int x, int y, int z, int radius) {
        std::map<int, int> labels;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                Z_Array *arr = get_z(x + i, y + j);
                if (arr != nullptr) {
                    for (int k = 0; k < arr->n; k++) {
                        if (arr->voxels[k].z >= z - radius && arr->voxels[k].z <= z + radius && arr->voxels[k].label > -1) {
                            labels[arr->voxels[k].label]++;
                        }
                    }
                }
            }
        }
        if (labels.empty()) {
            return -100;
        }
        int max_label = -1, max_cnt = 0;
        for (auto &p : labels) {
            if (p.second > max_cnt) {
                max_label = p.first;
                max_cnt = p.second;
            }
        }
        return max_label;
    }

    float get_voxel_res() const {
        return voxel_res;
    }

    auto get_bound() const {
        return bound;
    }

    // overlap means number of points which has at least one neighbour(or itself) in other volume grid
    int get_overlap(VolumeGrid *other, int radius) {
        if (std::fabs(voxel_res - other->voxel_res) > 1e-5) {
            // overlap of two volume grids with different voxel resolutions is not supported
            return 0;
        }
        // check bound
        auto [x_min, y_min, z_min, x_max, y_max, z_max] = bound;
        auto [x_min2, y_min2, z_min2, x_max2, y_max2, z_max2] = other->get_bound();
        if (x_min > x_max2 + radius || x_min2 > x_max + radius ||
            y_min > y_max2 + radius || y_min2 > y_max + radius ||
            z_min > z_max2 + radius || z_min2 > z_max + radius) {
            return 0;
        }
        std::vector<int> ans(num_workers, 0);
        auto run = [&](int id) {
            for (int i = 0; i < X_MAX * 2; i++) {
                if (data[i].arr == nullptr || i % num_workers != id)
                    continue;
                for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                    Z_Array *&arr = data[i].arr[j];
                    if (arr != nullptr) {
                        for (int k = 0; k < arr->n; k++) {
                            const Voxel &v = arr->voxels[k];
                            int flag = 0;
                            for (int dx = -radius; dx <= radius && !flag; dx++)
                            for (int dy = -radius; dy <= radius; dy++) {
                                Z_Array *arr2 = other->get_z(i - X_MAX + dx, j + data[i].mn + dy);
                                if (arr2 != nullptr && arr2->valid.test(v.z - radius, v.z + radius + 1)) {
                                    flag = 1;
                                    break;
                                }
                            }
                            ans[id] += flag;
                        }
                    }
                }
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; i++) {
            threads.push_back(std::thread(run, i));
        }
        int sum = 0;
        for (int i = 0; i < num_workers; i++) {
            threads[i].join();
            sum += ans[i];
        }
        return sum;
    }

    void radius_denoise(int min_points, int radius) {
        auto run = [&](int id) {
            for (int i = 0; i < X_MAX * 2; i++) {
                if (data[i].arr == nullptr || i % num_workers != id)
                    continue;
                for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                    Z_Array *&arr = data[i].arr[j];
                    if (arr != nullptr) {
                        int cnt = 0;
                        for (int k = 0; k < arr->n; k++) {
                            const Voxel &v = arr->voxels[k];
                            int nb = 0;
                            for (int dx = -radius; dx <= radius && nb < min_points; dx++)
                            for (int dy = -radius; dy <= radius && nb < min_points; dy++) {
                                Z_Array *arr2 = get_z(i - X_MAX + dx, j + data[i].mn + dy);
                                if (arr2 != nullptr) {
                                    nb += arr2->valid.count(v.z - radius, v.z + radius + 1);
                                }
                            }
                            if (nb >= min_points) {
                                arr->voxels[cnt++] = v;
                            }
                        }
                        arr->n = cnt;
                    }
                }
            }
            for (int i = 0; i < X_MAX * 2; i++) {
                if (data[i].arr == nullptr || i % num_workers != id)
                    continue;
                for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                    Z_Array *&arr = data[i].arr[j];
                    if (arr != nullptr) {
                        arr->valid.reset_all();
                        for (int k = 0; k < arr->n; k++) {
                            arr->valid.set(arr->voxels[k].z);
                        }
                    }
                }
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; i++) {
            threads.push_back(std::thread(run, i));
        }
        for (int i = 0; i < num_workers; i++) {
            threads[i].join();
        }
    }

    ~VolumeGrid() {
        for (int i = 0; i < X_MAX * 2; i++) {
            if (data[i].arr == nullptr)
                continue;
            for (int j = 0; j <= data[i].mx - data[i].mn; j++) {
                if (data[i].arr[j] != nullptr) {
                    delete data[i].arr[j];
                }
            }
        }
    }
};