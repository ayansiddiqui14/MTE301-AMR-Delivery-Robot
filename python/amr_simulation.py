"""
MTE301 - Autonomous Mobile Robot (AMR) Package Delivery Simulation
Toronto Metropolitan University - Fall 2025
Group: Raed Jamal, Umair Khan, Lageen Pirabalini, Ayan Siddiqui

Three-Layer Architecture:
1. Path Planning Layer - A* algorithm for route optimization
2. Perception Layer - Obstacle detection and avoidance
3. Control Layer - Motor command execution for navigation
"""

import pygame
import heapq
import random
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

# =============================================================================
# CONFIGURATION
# =============================================================================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
GRID_SIZE = 20
COLS = WINDOW_WIDTH // GRID_SIZE
ROWS = WINDOW_HEIGHT // GRID_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (220, 60, 60)
GREEN = (60, 180, 60)
BLUE = (60, 100, 220)
YELLOW = (240, 200, 60)
ORANGE = (240, 140, 40)
PURPLE = (150, 80, 180)
CYAN = (60, 200, 220)
DARK_GREEN = (40, 120, 40)

# =============================================================================
# DATA STRUCTURES
# =============================================================================
class RobotState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    AVOIDING = "AVOIDING"
    PICKUP = "PICKING UP"
    DELIVERY = "DELIVERING"
    RETURNING = "RETURNING"

@dataclass
class Package:
    id: int
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    picked_up: bool = False
    delivered: bool = False

@dataclass
class Pedestrian:
    x: float
    y: float
    dx: float
    dy: float
    grid_x: int = 0
    grid_y: int = 0
    
    def update(self):
        self.x += self.dx
        self.y += self.dy
        if self.x < 1 or self.x > COLS - 2:
            self.dx *= -1
        if self.y < 1 or self.y > ROWS - 2:
            self.dy *= -1
        self.grid_x = int(self.x)
        self.grid_y = int(self.y)

# =============================================================================
# PATH PLANNING LAYER - A* Algorithm
# =============================================================================
class PathPlanner:
    """Implements A* algorithm for optimal route planning"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS:
                    if self.grid[ny][nx] == 0:
                        neighbors.append((nx, ny))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  dynamic_obstacles: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        if dynamic_obstacles is None:
            dynamic_obstacles = set()
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        open_set_hash = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                if neighbor in dynamic_obstacles:
                    continue
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if dx + dy == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []

# =============================================================================
# PERCEPTION LAYER - Obstacle Detection
# =============================================================================
class PerceptionSystem:
    """Handles sensor simulation and obstacle detection"""
    
    def __init__(self, detection_range: int = 3):
        self.detection_range = detection_range
        self.detected_obstacles = set()
    
    def scan(self, robot_pos: Tuple[int, int], grid, pedestrians: List[Pedestrian]) -> Set[Tuple[int, int]]:
        self.detected_obstacles = set()
        rx, ry = robot_pos
        
        for dx in range(-self.detection_range, self.detection_range + 1):
            for dy in range(-self.detection_range, self.detection_range + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS:
                    if grid[ny][nx] == 1:
                        self.detected_obstacles.add((nx, ny))
                    for ped in pedestrians:
                        if ped.grid_x == nx and ped.grid_y == ny:
                            self.detected_obstacles.add((nx, ny))
        
        return self.detected_obstacles
    
    def check_collision_ahead(self, robot_pos: Tuple[int, int], 
                               next_pos: Tuple[int, int], 
                               pedestrians: List[Pedestrian]) -> bool:
        for ped in pedestrians:
            if (ped.grid_x, ped.grid_y) == next_pos:
                return True
        return False

# =============================================================================
# CONTROL LAYER - Robot Controller
# =============================================================================
class RobotController:
    """Controls robot movement and state management"""
    
    def __init__(self, start_pos: Tuple[int, int]):
        self.x, self.y = start_pos
        self.target_x, self.target_y = start_pos
        self.state = RobotState.IDLE
        self.path = []
        self.path_index = 0
        self.speed = 0.1
        self.current_package: Optional[Package] = None
        self.packages_delivered = 0
        self.total_distance = 0.0
        self.home_pos = start_pos
    
    @property
    def grid_pos(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))
    
    def set_path(self, path: List[Tuple[int, int]]):
        self.path = path
        self.path_index = 0
        if path:
            self.state = RobotState.MOVING
    
    def update(self, perception: PerceptionSystem, pedestrians: List[Pedestrian]) -> bool:
        if not self.path or self.path_index >= len(self.path):
            return True
        
        target = self.path[self.path_index]
        
        if perception.check_collision_ahead(self.grid_pos, target, pedestrians):
            self.state = RobotState.AVOIDING
            return False
        
        self.state = RobotState.MOVING
        
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < self.speed:
            self.total_distance += dist
            self.x, self.y = float(target[0]), float(target[1])
            self.path_index += 1
            if self.path_index >= len(self.path):
                return True
        else:
            self.total_distance += self.speed
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        
        return False

# =============================================================================
# CAMPUS MAP GENERATOR
# =============================================================================
def generate_campus_map() -> List[List[int]]:
    grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    
    for x in range(COLS):
        grid[0][x] = 1
        grid[ROWS-1][x] = 1
    for y in range(ROWS):
        grid[y][0] = 1
        grid[y][COLS-1] = 1
    
    buildings = [
        (5, 5, 8, 6), (20, 3, 10, 8), (35, 5, 8, 6),
        (5, 18, 10, 8), (22, 16, 8, 10), (38, 18, 8, 8),
        (8, 28, 12, 4), (28, 28, 10, 4),
    ]
    
    for bx, by, bw, bh in buildings:
        for y in range(by, min(by + bh, ROWS)):
            for x in range(bx, min(bx + bw, COLS)):
                grid[y][x] = 1
    
    return grid

# =============================================================================
# SIMULATION CLASS
# =============================================================================
class AMRSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("MTE301 - AMR Package Delivery Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        self.grid = generate_campus_map()
        self.path_planner = PathPlanner(self.grid)
        self.perception = PerceptionSystem(detection_range=4)
        self.robot = RobotController((3, ROWS - 5))
        self.packages = self.generate_packages()
        self.current_package_idx = 0
        self.pedestrians = self.generate_pedestrians(5)
        self.show_grid = True
        self.show_sensors = True
        self.planned_path = []
        self.replan_cooldown = 0
        self.start_time = pygame.time.get_ticks()
    
    def generate_packages(self) -> List[Package]:
        return [
            Package(1, (16, 10), (45, 8)),
            Package(2, (45, 15), (15, 25)),
            Package(3, (16, 25), (42, 30)),
        ]
    
    def generate_pedestrians(self, count: int) -> List[Pedestrian]:
        pedestrians = []
        for _ in range(count):
            while True:
                x = random.randint(5, COLS - 5)
                y = random.randint(5, ROWS - 5)
                if self.grid[y][x] == 0:
                    dx = random.choice([-0.03, 0.03])
                    dy = random.choice([-0.03, 0.03])
                    pedestrians.append(Pedestrian(float(x), float(y), dx, dy))
                    break
        return pedestrians
    
    def get_dynamic_obstacles(self) -> Set[Tuple[int, int]]:
        obstacles = set()
        for ped in self.pedestrians:
            obstacles.add((ped.grid_x, ped.grid_y))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    obstacles.add((ped.grid_x + dx, ped.grid_y + dy))
        return obstacles
    
    def update(self):
        for ped in self.pedestrians:
            next_x = int(ped.x + ped.dx * 10)
            next_y = int(ped.y + ped.dy * 10)
            if 0 <= next_x < COLS and 0 <= next_y < ROWS:
                if self.grid[next_y][next_x] == 1:
                    ped.dx *= -1
                    ped.dy *= -1
            ped.update()
        
        self.perception.scan(self.robot.grid_pos, self.grid, self.pedestrians)
        
        if self.robot.state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.robot.state in [RobotState.MOVING, RobotState.AVOIDING]:
            self.handle_moving_state()
        
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1
    
    def handle_idle_state(self):
        if self.current_package_idx < len(self.packages):
            pkg = self.packages[self.current_package_idx]
            
            if not pkg.picked_up:
                self.robot.state = RobotState.PLANNING
                target = pkg.pickup
            elif not pkg.delivered:
                self.robot.state = RobotState.PLANNING
                target = pkg.dropoff
            else:
                self.current_package_idx += 1
                return
            
            dynamic_obs = self.get_dynamic_obstacles()
            path = self.path_planner.find_path(self.robot.grid_pos, target, dynamic_obs)
            
            if path:
                self.planned_path = path
                self.robot.set_path(path)
            else:
                self.robot.state = RobotState.IDLE
        else:
            if self.robot.grid_pos != self.robot.home_pos:
                path = self.path_planner.find_path(
                    self.robot.grid_pos, 
                    self.robot.home_pos,
                    self.get_dynamic_obstacles()
                )
                if path:
                    self.robot.state = RobotState.RETURNING
                    self.planned_path = path
                    self.robot.set_path(path)
    
    def handle_moving_state(self):
        if self.robot.state == RobotState.AVOIDING and self.replan_cooldown == 0:
            self.replan_path()
            self.replan_cooldown = 30
            return
        
        reached = self.robot.update(self.perception, self.pedestrians)
        
        if reached:
            if self.current_package_idx < len(self.packages):
                pkg = self.packages[self.current_package_idx]
                
                if not pkg.picked_up and self.robot.grid_pos == pkg.pickup:
                    pkg.picked_up = True
                    self.robot.current_package = pkg
                    self.robot.state = RobotState.PICKUP
                    pygame.time.delay(500)
                    self.robot.state = RobotState.IDLE
                    
                elif pkg.picked_up and not pkg.delivered and self.robot.grid_pos == pkg.dropoff:
                    pkg.delivered = True
                    self.robot.current_package = None
                    self.robot.packages_delivered += 1
                    self.robot.state = RobotState.DELIVERY
                    pygame.time.delay(500)
                    self.robot.state = RobotState.IDLE
                else:
                    self.robot.state = RobotState.IDLE
            else:
                self.robot.state = RobotState.IDLE
    
    def replan_path(self):
        if self.current_package_idx < len(self.packages):
            pkg = self.packages[self.current_package_idx]
            if not pkg.picked_up:
                target = pkg.pickup
            elif not pkg.delivered:
                target = pkg.dropoff
            else:
                return
        elif self.robot.grid_pos != self.robot.home_pos:
            target = self.robot.home_pos
        else:
            return
        
        dynamic_obs = self.get_dynamic_obstacles()
        path = self.path_planner.find_path(self.robot.grid_pos, target, dynamic_obs)
        
        if path:
            self.planned_path = path
            self.robot.set_path(path)
    
    def draw(self):
        self.screen.fill(DARK_GREEN)
        
        if self.show_grid:
            for x in range(0, WINDOW_WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, (50, 100, 50), (x, 0), (x, WINDOW_HEIGHT), 1)
            for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, (50, 100, 50), (0, y), (WINDOW_WIDTH, y), 1)
        
        for y in range(ROWS):
            for x in range(COLS):
                if self.grid[y][x] == 1:
                    rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                    pygame.draw.rect(self.screen, GRAY, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        if self.show_sensors:
            rx, ry = self.robot.grid_pos
            for dx in range(-self.perception.detection_range, self.perception.detection_range + 1):
                for dy in range(-self.perception.detection_range, self.perception.detection_range + 1):
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS:
                        surf = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                        surf.fill((100, 200, 255, 30))
                        self.screen.blit(surf, (nx * GRID_SIZE, ny * GRID_SIZE))
        
        for obs in self.perception.detected_obstacles:
            rect = pygame.Rect(obs[0] * GRID_SIZE + 2, obs[1] * GRID_SIZE + 2, 
                              GRID_SIZE - 4, GRID_SIZE - 4)
            pygame.draw.rect(self.screen, RED, rect, 2)
        
        if self.planned_path:
            for i, pos in enumerate(self.planned_path):
                color = CYAN if i >= self.robot.path_index else LIGHT_GRAY
                center = (pos[0] * GRID_SIZE + GRID_SIZE // 2,
                         pos[1] * GRID_SIZE + GRID_SIZE // 2)
                pygame.draw.circle(self.screen, color, center, 4)
            
            for i in range(len(self.planned_path) - 1):
                start = (self.planned_path[i][0] * GRID_SIZE + GRID_SIZE // 2,
                        self.planned_path[i][1] * GRID_SIZE + GRID_SIZE // 2)
                end = (self.planned_path[i+1][0] * GRID_SIZE + GRID_SIZE // 2,
                      self.planned_path[i+1][1] * GRID_SIZE + GRID_SIZE // 2)
                color = CYAN if i >= self.robot.path_index else LIGHT_GRAY
                pygame.draw.line(self.screen, color, start, end, 2)
        
        for pkg in self.packages:
            if not pkg.picked_up:
                px, py = pkg.pickup
                rect = pygame.Rect(px * GRID_SIZE + 2, py * GRID_SIZE + 2,
                                  GRID_SIZE - 4, GRID_SIZE - 4)
                pygame.draw.rect(self.screen, YELLOW, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)
                text = self.font.render(str(pkg.id), True, BLACK)
                self.screen.blit(text, (px * GRID_SIZE + 6, py * GRID_SIZE + 2))
            
            if not pkg.delivered:
                dx, dy = pkg.dropoff
                rect = pygame.Rect(dx * GRID_SIZE + 2, dy * GRID_SIZE + 2,
                                  GRID_SIZE - 4, GRID_SIZE - 4)
                pygame.draw.rect(self.screen, ORANGE if pkg.picked_up else PURPLE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)
                text = self.font.render(str(pkg.id), True, WHITE)
                self.screen.blit(text, (dx * GRID_SIZE + 6, dy * GRID_SIZE + 2))
        
        hx, hy = self.robot.home_pos
        rect = pygame.Rect(hx * GRID_SIZE, hy * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, GREEN, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        text = self.font.render("H", True, BLACK)
        self.screen.blit(text, (hx * GRID_SIZE + 5, hy * GRID_SIZE + 2))
        
        for ped in self.pedestrians:
            center = (int(ped.x * GRID_SIZE + GRID_SIZE // 2),
                     int(ped.y * GRID_SIZE + GRID_SIZE // 2))
            pygame.draw.circle(self.screen, RED, center, GRID_SIZE // 3)
            pygame.draw.circle(self.screen, BLACK, center, GRID_SIZE // 3, 2)
        
        robot_center = (int(self.robot.x * GRID_SIZE + GRID_SIZE // 2),
                       int(self.robot.y * GRID_SIZE + GRID_SIZE // 2))
        pygame.draw.circle(self.screen, BLUE, robot_center, GRID_SIZE // 2 - 2)
        pygame.draw.circle(self.screen, BLACK, robot_center, GRID_SIZE // 2 - 2, 2)
        
        if self.robot.current_package:
            pygame.draw.circle(self.screen, YELLOW, robot_center, GRID_SIZE // 4)
        
        self.draw_ui()
    
    def draw_ui(self):
        panel_rect = pygame.Rect(WINDOW_WIDTH - 250, 10, 240, 200)
        pygame.draw.rect(self.screen, (40, 40, 40, 200), panel_rect)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2)
        
        x, y = WINDOW_WIDTH - 240, 20
        title = self.large_font.render("AMR Status", True, WHITE)
        self.screen.blit(title, (x, y))
        y += 35
        
        state_color = {
            RobotState.IDLE: WHITE, RobotState.MOVING: GREEN,
            RobotState.AVOIDING: ORANGE, RobotState.PLANNING: CYAN,
            RobotState.PICKUP: YELLOW, RobotState.DELIVERY: PURPLE,
            RobotState.RETURNING: BLUE
        }
        state_text = self.font.render(f"State: {self.robot.state.value}", True, 
                                      state_color.get(self.robot.state, WHITE))
        self.screen.blit(state_text, (x, y))
        y += 25
        
        delivered_text = self.font.render(
            f"Delivered: {self.robot.packages_delivered}/{len(self.packages)}", True, WHITE)
        self.screen.blit(delivered_text, (x, y))
        y += 25
        
        if self.current_package_idx < len(self.packages):
            pkg = self.packages[self.current_package_idx]
            task = f"Task: Pickup #{pkg.id}" if not pkg.picked_up else f"Task: Deliver #{pkg.id}"
        else:
            task = "Task: Return Home"
        task_text = self.font.render(task, True, WHITE)
        self.screen.blit(task_text, (x, y))
        y += 25
        
        dist_text = self.font.render(f"Distance: {self.robot.total_distance:.1f} units", True, WHITE)
        self.screen.blit(dist_text, (x, y))
        y += 25
        
        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000
        time_text = self.font.render(f"Time: {elapsed:.1f}s", True, WHITE)
        self.screen.blit(time_text, (x, y))
        
        legend_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, (40, 40, 40), (10, legend_y - 10, 300, 95))
        pygame.draw.rect(self.screen, WHITE, (10, legend_y - 10, 300, 95), 1)
        
        legend_items = [
            (BLUE, "Robot"), (YELLOW, "Pickup Point"), (PURPLE, "Dropoff Point"),
            (RED, "Pedestrian"), (GREEN, "Home/Depot"), (CYAN, "Planned Path")
        ]
        
        for i, (color, label) in enumerate(legend_items):
            lx = 20 + (i % 2) * 150
            ly = legend_y + (i // 2) * 25
            pygame.draw.circle(self.screen, color, (lx, ly + 8), 8)
            text = self.font.render(label, True, WHITE)
            self.screen.blit(text, (lx + 15, ly))
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_s:
                        self.show_sensors = not self.show_sensors
                    elif event.key == pygame.K_r:
                        self.__init__()
            
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MTE301 - Autonomous Mobile Robot Package Delivery Simulation")
    print("=" * 60)
    print("\nControls:")
    print("  G - Toggle grid")
    print("  S - Toggle sensor visualization")
    print("  R - Reset simulation")
    print("  ESC - Quit")
    print("\nStarting simulation...")
    print("-" * 60)
    
    sim = AMRSimulation()
    sim.run()