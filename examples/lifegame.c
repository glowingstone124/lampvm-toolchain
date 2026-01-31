/*
Conway's Game of Life for LampVM framebuffer.
Each pixel is a single uint32 color stored in framebuffer memory.
*/

int GRID_W = 640;
int GRID_H = 480;

char grid[307200];
char next_grid[307200];
int min_x;
int min_y;
int max_x;
int max_y;
int prev_min_x;
int prev_min_y;
int prev_max_x;
int prev_max_y;

int idx(int y, int x) {
    return y * GRID_W + x;
}

int clamp_min(int v, int lo) {
    if (v < lo) {
        return lo;
    }
    return v;
}

int clamp_max(int v, int hi) {
    if (v > hi) {
        return hi;
    }
    return v;
}

int reset_bounds() {
    min_x = GRID_W - 1;
    min_y = GRID_H - 1;
    max_x = 0;
    max_y = 0;
    return 0;
}

int reset_prev_bounds() {
    prev_min_x = GRID_W - 1;
    prev_min_y = GRID_H - 1;
    prev_max_x = 0;
    prev_max_y = 0;
    return 0;
}

int update_bounds_cell(int y, int x) {
    if (x < min_x) {
        min_x = x;
    }
    if (x > max_x) {
        max_x = x;
    }
    if (y < min_y) {
        min_y = y;
    }
    if (y > max_y) {
        max_y = y;
    }
    return 0;
}

int bounds_empty() {
    if (min_x > max_x || min_y > max_y) {
        return 1;
    }
    return 0;
}

int prev_bounds_empty() {
    if (prev_min_x > prev_max_x || prev_min_y > prev_max_y) {
        return 1;
    }
    return 0;
}

int recompute_bounds() {
    reset_bounds();
    int y = 0;
    while (y <= 479) {
        int x = 0;
        while (x <= 639) {
            if (grid[idx(y, x)]) {
                update_bounds_cell(y, x);
            }
            x = x + 1;
        }
        y = y + 1;
    }
    return 0;
}

int count_neighbors(int y, int x) {
    int count = 0;
    int dy = -1;
    while (dy <= 1) {
        int ny = y + dy;
        if (ny >= 0 && ny <= 479) {
            int dx = -1;
            while (dx <= 1) {
                int nx = x + dx;
                if (nx >= 0 && nx <= 639) {
                    if (!(dy == 0 && dx == 0)) {
                        if (grid[idx(ny, nx)]) {
                            count = count + 1;
                        }
                    }
                }
                dx = dx + 1;
            }
        }
        dy = dy + 1;
    }
    return count;
}

int clear_grid() {
    int y = 0;
    while (y <= 479) {
        int x = 0;
        while (x <= 639) {
            grid[idx(y, x)] = 0;
            next_grid[idx(y, x)] = 0;
            x = x + 1;
        }
        y = y + 1;
    }
    reset_bounds();
    reset_prev_bounds();
    return 0;
}

int seed_patterns() {
    reset_bounds();
    // Glider
    grid[idx(10, 10)] = 1;
    update_bounds_cell(10, 10);
    grid[idx(11, 11)] = 1;
    update_bounds_cell(11, 11);
    grid[idx(12, 9)] = 1;
    update_bounds_cell(12, 9);
    grid[idx(12, 10)] = 1;
    update_bounds_cell(12, 10);
    grid[idx(12, 11)] = 1;
    update_bounds_cell(12, 11);

    // Small exploder
    grid[idx(30, 30)] = 1;
    update_bounds_cell(30, 30);
    grid[idx(29, 30)] = 1;
    update_bounds_cell(29, 30);
    grid[idx(31, 30)] = 1;
    update_bounds_cell(31, 30);
    grid[idx(28, 29)] = 1;
    update_bounds_cell(28, 29);
    grid[idx(28, 31)] = 1;
    update_bounds_cell(28, 31);
    grid[idx(29, 28)] = 1;
    update_bounds_cell(29, 28);
    grid[idx(29, 32)] = 1;
    update_bounds_cell(29, 32);
    grid[idx(30, 28)] = 1;
    update_bounds_cell(30, 28);
    grid[idx(30, 32)] = 1;
    update_bounds_cell(30, 32);
    grid[idx(31, 28)] = 1;
    update_bounds_cell(31, 28);
    grid[idx(31, 32)] = 1;
    update_bounds_cell(31, 32);
    grid[idx(32, 29)] = 1;
    update_bounds_cell(32, 29);
    grid[idx(32, 31)] = 1;
    update_bounds_cell(32, 31);
    grid[idx(33, 30)] = 1;
    update_bounds_cell(33, 30);

    // Blinker
    grid[idx(60, 80)] = 1;
    update_bounds_cell(60, 80);
    grid[idx(60, 81)] = 1;
    update_bounds_cell(60, 81);
    grid[idx(60, 82)] = 1;
    update_bounds_cell(60, 82);

    // Toad
    grid[idx(100, 100)] = 1;
    update_bounds_cell(100, 100);
    grid[idx(100, 101)] = 1;
    update_bounds_cell(100, 101);
    grid[idx(100, 102)] = 1;
    update_bounds_cell(100, 102);
    grid[idx(101, 99)] = 1;
    update_bounds_cell(101, 99);
    grid[idx(101, 100)] = 1;
    update_bounds_cell(101, 100);
    grid[idx(101, 101)] = 1;
    update_bounds_cell(101, 101);

    // Beacon
    grid[idx(140, 140)] = 1;
    update_bounds_cell(140, 140);
    grid[idx(140, 141)] = 1;
    update_bounds_cell(140, 141);
    grid[idx(141, 140)] = 1;
    update_bounds_cell(141, 140);
    grid[idx(141, 141)] = 1;
    update_bounds_cell(141, 141);
    grid[idx(142, 142)] = 1;
    update_bounds_cell(142, 142);
    grid[idx(142, 143)] = 1;
    update_bounds_cell(142, 143);
    grid[idx(143, 142)] = 1;
    update_bounds_cell(143, 142);
    grid[idx(143, 143)] = 1;
    update_bounds_cell(143, 143);

    return 0;
}

int step_life() {
    if (bounds_empty()) {
        return 0;
    }

    int y0 = clamp_min(min_y - 1, 0);
    int y1 = clamp_max(max_y + 1, 479);
    int x0 = clamp_min(min_x - 1, 0);
    int x1 = clamp_max(max_x + 1, 639);

    int y = y0;
    while (y <= y1) {
        int x = x0;
        while (x <= x1) {
            int n = count_neighbors(y, x);
            if (grid[idx(y, x)]) {
                if (n == 2 || n == 3) {
                    next_grid[idx(y, x)] = 1;
                } else {
                    next_grid[idx(y, x)] = 0;
                }
            } else {
                if (n == 3) {
                    next_grid[idx(y, x)] = 1;
                } else {
                    next_grid[idx(y, x)] = 0;
                }
            }
            x = x + 1;
        }
        y = y + 1;
    }

    reset_bounds();
    y = y0;
    while (y <= y1) {
        int x = x0;
        while (x <= x1) {
            char v = next_grid[idx(y, x)];
            grid[idx(y, x)] = v;
            if (v) {
                update_bounds_cell(y, x);
            }
            x = x + 1;
        }
        y = y + 1;
    }
    return 0;
}

int draw() {
    int *fb = (int *)0x400000;
    if (!prev_bounds_empty()) {
        int py0 = clamp_min(prev_min_y - 1, 0);
        int py1 = clamp_max(prev_max_y + 1, 479);
        int px0 = clamp_min(prev_min_x - 1, 0);
        int px1 = clamp_max(prev_max_x + 1, 639);
        int py = py0;
        while (py <= py1) {
            int px = px0;
            while (px <= px1) {
                fb[py * 640 + px] = 0x00000000;
                px = px + 1;
            }
            py = py + 1;
        }
    }

    if (bounds_empty()) {
        reset_prev_bounds();
        return 0;
    }
    int y0 = clamp_min(min_y - 1, 0);
    int y1 = clamp_max(max_y + 1, 479);
    int x0 = clamp_min(min_x - 1, 0);
    int x1 = clamp_max(max_x + 1, 639);

    int y = y0;
    while (y <= y1) {
        int x = x0;
        while (x <= x1) {
            int color = 0x00000000;
            if (grid[idx(y, x)]) {
                color = 0x00FFFFFF;
            }
            fb[y * 640 + x] = color;
            x = x + 1;
        }
        y = y + 1;
    }

    prev_min_x = min_x;
    prev_min_y = min_y;
    prev_max_x = max_x;
    prev_max_y = max_y;
    return 0;
}

int delay(int ticks) {
    int i = 0;
    int end = ticks - 1;
    while (i <= end) {
        i = i + 1;
    }
    return 0;
}

int main() {
    clear_grid();
    seed_patterns();

    while (1) {
        draw();
        step_life();
        delay(2000000);
    }

    return 0;
}
