#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <fstream>
#include <cmath>

struct Cell {
    int r;
    int c;
};

struct Pose {
    double x;
    double y;
    double theta; // not really used here, but included for future expansion
};

bool isValid(int r, int c, int rows, int cols) {
    return (r >= 0 && r < rows && c >= 0 && c < cols);
}

int main() {
    // Simple occupancy grid:
    // '.' = free, '#' = obstacle
    // You can tweak this to match your course spec.
    std::vector<std::string> grid = {
        "..........",
        "..####....",
        "..#.......",
        "..#..####.",
        "..#.......",
        "..####..#.",
        "......#.#.",
        ".####.#.#.",
        "......#...",
        ".........."
    };

    const int rows = static_cast<int>(grid.size());
    const int cols = static_cast<int>(grid[0].size());

    // Start and goal in (row, col)
    Cell start{0, 0};
    Cell goal{9, 9};

    if (grid[start.r][start.c] == '#' || grid[goal.r][goal.c] == '#') {
        std::cerr << "Start or goal inside obstacle. Exiting.\n";
        return 1;
    }

    // BFS for shortest path on the grid
    std::queue<Cell> q;
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<Cell>> parent(rows, std::vector<Cell>(cols, {-1, -1}));

    q.push(start);
    visited[start.r][start.c] = true;

    // 4-connected neighbors (up, down, left, right)
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    bool reached = false;
    while (!q.empty() && !reached) {
        Cell cur = q.front();
        q.pop();

        for (int k = 0; k < 4; ++k) {
            int nr = cur.r + dr[k];
            int nc = cur.c + dc[k];

            if (!isValid(nr, nc, rows, cols)) continue;
            if (visited[nr][nc]) continue;
            if (grid[nr][nc] == '#') continue;

            visited[nr][nc] = true;
            parent[nr][nc] = cur;
            q.push({nr, nc});

            if (nr == goal.r && nc == goal.c) {
                reached = true;
                break;
            }
        }
    }

    if (!reached) {
        std::cerr << "No path found from start to goal.\n";
        return 1;
    }

    // Reconstruct path from goal back to start
    std::vector<Cell> path;
    Cell cur = goal;
    while (!(cur.r == start.r && cur.c == start.c)) {
        path.push_back(cur);
        cur = parent[cur.r][cur.c];
    }
    path.push_back(start);

    // Reverse to go from start -> goal
    std::reverse(path.begin(), path.end());

    std::cout << "Path length: " << path.size() << " cells\n";

    // Simulate a simple motion along the path
    // Assume each grid cell is 0.5 m, and each step is 0.5 s
    const double cell_size = 0.5;   // meters
    const double dt = 0.5;          // seconds per step
    double time = 0.0;

    // Open CSV file for Python plotting
    // This path assumes you run the program from cpp/build/
    std::ofstream ofs("../python/data/sim_path.csv");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open ../python/data/sim_path.csv for writing.\n";
        return 1;
    }

    // CSV header
    ofs << "time,x,y,step\n";

    for (size_t i = 0; i < path.size(); ++i) {
        Cell cell = path[i];
        // Convert cell indices to world coordinates (simple mapping)
        double x = cell.c * cell_size;
        double y = cell.r * cell_size;

        ofs << time << "," << x << "," << y << "," << i << "\n";
        time += dt;
    }

    ofs.close();

    std::cout << "Simulation complete. Path saved to ../python/data/sim_path.csv\n";
    std::cout << "Start: (" << start.r << ", " << start.c << "), "
              << "Goal: (" << goal.r << ", " << goal.c << ")\n";

    return 0;
}
