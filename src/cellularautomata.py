import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

GRID_SIZE = 100
NUM_AGENTS = 150
STEPS = 200
EXIT_POS = (GRID_SIZE - 1, GRID_SIZE // 2)

USE_GAUSSIAN_REPULSION = False
USE_BARRIER = True

kO = 6                # obstacle repulsion strength
OBSTACLE_SIGMA = 1   # influence radius

BARRIER_RADIUS = 8
BARRIER_OFFSET = 20

BOTTLENECK_LENGTH = 12
BOTTLENECK_WIDTH = 6

kS = 6
kD = 4
CRUSH_DENSITY = 4

def bottleneck_mask():
    mid = GRID_SIZE // 2
    half_open = BOTTLENECK_WIDTH // 2
    wall_start = mid - BOTTLENECK_LENGTH // 2
    wall_end = mid + BOTTLENECK_LENGTH // 2

    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    mask[mid - half_open:mid + half_open + 1, wall_start:wall_end] = 1
    return mask


BOTTLENECK_MASK = bottleneck_mask()

def build_geometry():
    mid = GRID_SIZE // 2
    half_open = BOTTLENECK_WIDTH // 2
    wall_start = mid - BOTTLENECK_LENGTH // 2
    wall_end = mid + BOTTLENECK_LENGTH // 2

    geometry = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    geometry[:, :wall_start] = 1
    geometry[:, wall_end:] = 1
    geometry[
        mid - half_open:mid + half_open + 1,
        wall_start:wall_end
    ] = 1

    if USE_BARRIER:
        barrier_center_x = wall_start - BARRIER_OFFSET
        barrier_center_y = mid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):

                dx = x - barrier_center_x
                dy = y - barrier_center_y

                distance_squared = dx**2 + dy**2

                if distance_squared <= BARRIER_RADIUS**2:
                    geometry[y, x] = 0

    ex, ey = EXIT_POS
    geometry[ey, ex] = 1
    return geometry

def compute_obstacle_field(geometry):
    field = np.zeros((GRID_SIZE, GRID_SIZE))
    if not USE_GAUSSIAN_REPULSION:
        return field

    obstacle_positions = np.argwhere(geometry == 0)
    if obstacle_positions.size == 0:
        return field

    ys, xs = np.indices((GRID_SIZE, GRID_SIZE))
    dx = xs[:, :, None] - obstacle_positions[:, 1]
    dy = ys[:, :, None] - obstacle_positions[:, 0]
    d2 = dx**2 + dy**2

    min_d2 = np.min(d2, axis=2)
    field = np.exp(-min_d2 / (2 * OBSTACLE_SIGMA**2))
    field[geometry == 0] = 0

    field = field / np.max(field)
    return field


def compute_static_field(geometry):
    S = np.full((GRID_SIZE, GRID_SIZE), -9999.0)
    dist = np.full((GRID_SIZE, GRID_SIZE), np.inf)
    ex, ey = EXIT_POS

    queue = deque([(ex, ey)])
    dist[ey, ex] = 0

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if geometry[ny, nx] == 1 and dist[ny, nx] == np.inf:
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((nx, ny))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if geometry[y, x] == 1:
                S[y, x] = -dist[y, x]
    return S


class Person(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        grid = self.model.grid
        geometry = self.model.geometry
        neighbors = grid.get_neighborhood(self.pos, moore=True, include_center=False)

        valid_cells = []
        scores = []

        for x, y in neighbors:
            if geometry[y, x] == 0:
                continue

            S = self.model.static_field[y, x]
            D = len(grid.get_cell_list_contents([(x, y)]))
            O = self.model.obstacle_field[y, x]

            score = kS * S - kD * D - kO * O

            valid_cells.append((x, y))
            scores.append(score)

        if not valid_cells:
            self.model.next_positions[self.unique_id] = self.pos
            return

        scores = np.array(scores)
        probs = np.exp(scores - np.max(scores))
        probs = probs / (np.sum(probs) + 1e-9)

        chosen = self.model.random.choices(valid_cells, weights=probs, k=1)[0]
        self.model.next_positions[self.unique_id] = chosen


class CrowdModel(Model):
    def __init__(self):
        super().__init__()
        self.grid = MultiGrid(GRID_SIZE, GRID_SIZE, torus=False)
        self.schedule = RandomActivation(self)
        self.geometry = build_geometry()
        self.obstacle_field = compute_obstacle_field(self.geometry)
        self.static_field = compute_static_field(self.geometry)
        self.next_positions = {}
        self.crush_event_history = []
        self.bottleneck_density_history = []
        
        
        mid = GRID_SIZE // 2
        for i in range(NUM_AGENTS):
            agent = Person(i, self)
            while True:
                x = self.random.randrange(0, mid - 6)
                y = self.random.randrange(0, GRID_SIZE)
                if self.geometry[y, x] == 1:
                    break
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

    def resolve_moves(self):
        target_map = {}
        for aid, pos in self.next_positions.items():
            target_map.setdefault(pos, []).append(aid)

        final_moves = {}
        for pos, agents in target_map.items():
            if len(agents) == 1:
                final_moves[agents[0]] = pos
            elif self.random.random() < 0.6:
                chosen = self.random.choice(agents)
                final_moves[chosen] = pos
        return final_moves

    def compute_crush_events(self):
        crush_cells = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.geometry[y, x] == 1:
                    count = len(self.grid.get_cell_list_contents([(x, y)]))
                    if count >= CRUSH_DENSITY:
                        crush_cells += 1
        return crush_cells

    def compute_bottleneck_density(self):
        total = 0
        cells = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if BOTTLENECK_MASK[y, x] == 1:
                    count = len(self.grid.get_cell_list_contents([(x, y)]))
                    total += count
                    cells += 1
        return total / max(1, cells)

    def step(self):
        self.next_positions = {}
        self.schedule.step()

        moves = self.resolve_moves()
        for agent in list(self.schedule.agents):
            if agent.unique_id in moves:
                self.grid.move_agent(agent, moves[agent.unique_id])

        for agent in list(self.schedule.agents):
            if agent.pos == EXIT_POS:
                self.grid.remove_agent(agent)
                self.schedule.remove(agent)

        crush = self.compute_crush_events()
        density = self.compute_bottleneck_density()
        self.crush_event_history.append(crush)
        self.bottleneck_density_history.append(density)


fig, ax = plt.subplots(figsize=(7, 7))


def visualize(model, step):
    ax.clear()
    density = np.zeros((GRID_SIZE, GRID_SIZE))
    crush_cells = []

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            count = len(model.grid.get_cell_list_contents([(x, y)]))
            density[y, x] = count
            if count >= CRUSH_DENSITY:
                crush_cells.append((x, y))

    ax.set_facecolor("white")
    masked = np.where(model.geometry == 1, density, np.nan)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="black")

    ax.imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=CRUSH_DENSITY * 2)
    for x, y in crush_cells:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="deepskyblue", linewidth=2))

    ex, ey = EXIT_POS
    ax.scatter(ex, ey, marker="X", s=200, color="blue")
    ax.set_title(f"Step {step} | Remaining {len(model.schedule.agents)}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.05)


def plot_results(model):
    steps = np.arange(len(model.crush_event_history))
    crush = np.array(model.crush_event_history)
    density = np.array(model.bottleneck_density_history)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(steps, density, color="tab:blue")
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("Density (agents/cell)")
    ax[0].set_title("Bottleneck density over time")
    ax[0].grid(alpha=0.3)

    ax[1].plot(steps, crush, color="red")
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Number of crush cells")
    ax[1].set_title("Crush events over time")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    model = CrowdModel()
    for t in range(STEPS):
        model.step()
        visualize(model, t)
        if len(model.schedule.agents) == 0:
            break
    plot_results(model)


if __name__ == "__main__":
    main()
