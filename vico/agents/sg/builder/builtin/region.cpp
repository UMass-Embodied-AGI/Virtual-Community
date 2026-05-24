#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>

const int dx[] = {0, 1, 0, -1, 1, 1, -1, -1};
const int dy[] = {1, 0, -1, 0, 1, -1, 1, -1};

void check_min(int &a, int b) {
    a = a < b ? a : b;
}
void check_max(int &a, int b) {
    a = a > b ? a : b;
}

void get_id(int h, int w, uint8_t *map, int *id) {
    int *qx, *qy, cur_id = 0;
    qx = new int[h * w];
    qy = new int[h * w];

    memset(id, 0, h * w * sizeof(int));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            if (!map[i * w + j] || id[i * w + j]) continue;
            int qh = 0, qt = 0;
            qx[qt] = i;
            qy[qt] = j;
            qt++;
            id[i * w + j] = ++cur_id;
            while (qh < qt) {
                int x = qx[qh], y = qy[qh];
                qh++;
                for (int k = 0; k < 8; k++) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx >= 0 && nx < h && ny >= 0 && ny < w && map[nx * w + ny] && !id[nx * w + ny]) {
                        id[nx * w + ny] = cur_id;
                        qx[qt] = nx;
                        qy[qt] = ny;
                        qt++;
                    }
                }
            }
            if (qt < 400) {
                for (int k = 0; k < qt; k++) {
                    map[qx[k] * w + qy[k]] = 0;
                    id[qx[k] * w + qy[k]] = -1;
                }
                cur_id--;
            }
        }
    delete[] qx;
    delete[] qy;
}

extern "C" void smooth(int h, int w, int radius, uint8_t *map) {
    uint8_t *nmap = new uint8_t[h * w];
    memset(nmap, 0, h * w * sizeof(uint8_t));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            for (int x = i - radius; x <= i + radius; x++)
                for (int y = j - radius; y <= j + radius; y++) {
                    if (x >= 0 && x < h && y >= 0 && y < w && map[x * w + y]) {
                        nmap[i * w + j] = 1;
                        break;
                    }
                }
        }
    memcpy(map, nmap, h * w * sizeof(uint8_t));
    delete[] nmap;
}

extern "C" void bfs(int h, int w, uint8_t *map, int *id, int *dist) {
    int *qx, *qy, qh = 0, qt = 0;
    qx = new int[h * w];
    qy = new int[h * w];
    get_id(h, w, map, id);

    memset(dist, -1, h * w * sizeof(int));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            if (map[i * w + j]) {
                qx[qt] = i;
                qy[qt] = j;
                dist[i * w + j] = 0;
                qt++;
            }
        }
    
    while (qh < qt) {
        int x = qx[qh], y = qy[qh];
        qh++;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < h && ny >= 0 && ny < w && dist[nx * w + ny] == -1) {
                dist[nx * w + ny] = dist[x * w + y] + 1;
                id[nx * w + ny] = id[x * w + y];
                qx[qt] = nx;
                qy[qt] = ny;
                qt++;
            }
        }
    }

    delete[] qx;
    delete[] qy;
}

// mat: (max(id), max(id))
extern "C" void adj_matrix(int h, int w, int *id, int *dist, int *mat) {
    int max_id = 0;
    for (int i = 0; i < h * w; i++)
        max_id = max_id > id[i] ? max_id : id[i];
    memset(mat, 0x3f, max_id * max_id * sizeof(int));
    for (int i = 0; i < max_id; i++)
        mat[i * max_id + i] = 0;

    int *ndist = new int[h * w];
    memset(ndist, -1, h * w * sizeof(int));
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            if (i > 0 && j > 0 && i < h - 1 && j < w - 1) {
                int tid[] = {id[(i - 1) * w + j], id[(i + 1) * w + j], id[i * w + j - 1], id[i * w + j + 1], id[i * w + j]};
                std::sort(tid, tid + 5);
                int num = std::unique(tid, tid + 5) - tid;
                if (num >= 2) {
                    for (int k = 0; k < num; k++)
                        for (int l = 0; l < num; l++) if (k != l)
                            check_min(mat[(tid[k] - 1) * max_id + tid[l] - 1], dist[i * w + j]);
                    ndist[i * w + j] = dist[i * w + j];
                }
            }
        }
    memcpy(dist, ndist, h * w * sizeof(int));
}