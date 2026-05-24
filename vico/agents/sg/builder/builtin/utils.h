#include <math.h>

const float PI = 3.14159265358979323846;
const float MAX_HEIGHT = 2.0;

inline int floordiv(int a, int b) {
    return a < 0 ? (a - b + 1) / b : a / b;
}

extern "C" void image_to_pcd(uint8_t *rgb, float *depth, int *label, float *p, uint8_t *c, int *l, int w, int h, float fov, float *extrinsic) {
    const double PI = 3.14159265358979323846;
    float tan_fov = std::tan(fov / 180 * PI / 2) * w / h;
    float fx = w / 2.0 / tan_fov, fy = h / 2.0 / tan_fov;
    float cx = w / 2.0, cy = h / 2.0;
    int n = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float z = depth[y * w + x];
            if (label[y * w + x] >= -1) {
                p[3 * n] = (x - cx) * z / fx;
                p[3 * n + 1] = (h - 1 - y - cy) * z / fy;
                p[3 * n + 2] = -z;
                c[3 * n] = rgb[3 * (y * w + x)];
                c[3 * n + 1] = rgb[3 * (y * w + x) + 1];
                c[3 * n + 2] = rgb[3 * (y * w + x) + 2];
                l[n] = label[y * w + x];
                n++;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        float x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
        p[3 * i] = extrinsic[0] * x + extrinsic[1] * y + extrinsic[2] * z + extrinsic[3];
        p[3 * i + 1] = extrinsic[4] * x + extrinsic[5] * y + extrinsic[6] * z + extrinsic[7];
        p[3 * i + 2] = extrinsic[8] * x + extrinsic[9] * y + extrinsic[10] * z + extrinsic[11];
    }
}

extern "C" int *convex_hull(int *points, int n, int *size) {
    if (n <= 2) {
        int *hull = new int[n * 2];
        for (int i = 0; i < n; i++) {
            hull[2 * i] = points[2 * i];
            hull[2 * i + 1] = points[2 * i + 1];
        }
        *size = n;
        return hull;
    }

    int *p = new int[n * 2];
    memcpy(p, points, n * 2 * sizeof(int));
    // Find point with lowest y-coordinate (and leftmost if tie)
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (p[2 * i + 1] < p[2 * min_idx + 1] || 
            (p[2 * i + 1] == p[2 * min_idx + 1] && p[2 * i] < p[2 * min_idx])) {
            min_idx = i;
        }
    }

    // Swap the lowest point to position 0
    std::swap(p[0], p[2 * min_idx]);
    std::swap(p[1], p[2 * min_idx + 1]);

    // Sort points by polar angle with respect to the lowest point
    auto comp = [&](int i, int j) {
        int ori = (p[2 * i] - p[0]) * (p[2 * j + 1] - p[1]) - 
                  (p[2 * j] - p[0]) * (p[2 * i + 1] - p[1]);
        if (ori == 0) {
            // If collinear, sort by distance from p[0]
            int dist_i = (p[2 * i] - p[0]) * (p[2 * i] - p[0]) + 
                         (p[2 * i + 1] - p[1]) * (p[2 * i + 1] - p[1]);
            int dist_j = (p[2 * j] - p[0]) * (p[2 * j] - p[0]) + 
                         (p[2 * j + 1] - p[1]) * (p[2 * j + 1] - p[1]);
            return dist_i < dist_j;
        }
        return ori > 0; // Counter-clockwise sorting
    };

    int *indices = new int[n];
    for (int i = 0; i < n; i++) indices[i] = i;
    std::sort(indices + 1, indices + n, comp);

    // Graham scan
    int *queue = new int[n];
    int k = 0;
    
    for (int i = 0; i < n; i++) {
        while (k >= 2 && 
              ((p[2 * queue[k-1]] - p[2 * queue[k-2]]) * 
               (p[2 * indices[i] + 1] - p[2 * queue[k-2] + 1]) - 
               (p[2 * indices[i]] - p[2 * queue[k-2]]) * 
               (p[2 * queue[k-1] + 1] - p[2 * queue[k-2] + 1])) <= 0) {
            k--;
        }
        queue[k++] = indices[i];
    }
    
    *size = k;
    int *hull = new int[k * 2];
    for (int i = 0; i < k; i++) {
        hull[2 * i] = p[2 * queue[i]];
        hull[2 * i + 1] = p[2 * queue[i] + 1];
    }
    delete[] queue;
    delete[] indices;
    return hull;
}

extern "C" float dist_to_hull(int *p, int *hull, int n) {
    if (n <= 0) return -1;  // Invalid hull

    // Check if p is inside the hull
    bool inside = true;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // Cross product to determine if point is to the right of edge
        int cross = (hull[2 * j] - hull[2 * i]) * (p[1] - hull[2 * i + 1]) - 
                    (hull[2 * j + 1] - hull[2 * i + 1]) * (p[0] - hull[2 * i]);
        if (cross < 0) {
            inside = false;
            break;
        }
    }
    if (inside) return 0;

    // Find minimum distance to any edge
    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        
        // Edge vector
        float ex = hull[2 * j] - hull[2 * i];
        float ey = hull[2 * j + 1] - hull[2 * i + 1];
        float length_squared = ex * ex + ey * ey;
        
        // If edge length is zero, compute distance to the point
        if (length_squared < 1e-6) {
            float dx = p[0] - hull[2 * i];
            float dy = p[1] - hull[2 * i + 1];
            float dist = std::sqrt(dx * dx + dy * dy);
            min_dist = std::min(min_dist, dist);
            continue;
        }
        
        // Project point onto edge
        float t = ((p[0] - hull[2 * i]) * ex + (p[1] - hull[2 * i + 1]) * ey) / length_squared;
        
        // Find closest point on segment
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        
        // Closest point on the line segment
        float px = hull[2 * i] + t * ex;
        float py = hull[2 * i + 1] + t * ey;
        
        // Distance to closest point
        float dx = p[0] - px;
        float dy = p[1] - py;
        float dist = std::sqrt(dx * dx + dy * dy);
        
        min_dist = std::min(min_dist, dist);
    }
    
    return min_dist;
}