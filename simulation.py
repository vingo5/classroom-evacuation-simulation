import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ─────────────────────────────────────────────────────────────
# 1. Simulation Parameters
# ─────────────────────────────────────────────────────────────
N          = 30           # Number of students
L          = 10.0         # Classroom side length (m)
dt         = 0.04         # Time step (s)
T_final    = 60.0         # Max simulation time (s)
steps      = int(T_final / dt)

alpha      = 4.5          # Velocity relaxation rate (s⁻¹)

# Door parameters: single point-sink at centre of right wall
door_pos   = np.array([L, L / 2.0])
door_delay = 0.8          # Seconds between successive exits (single-file throughput)

v_max      = 1.4          # Asymptotic free-flow speed (m/s)
R_rep      = 0.6          # Personal-space radius (m)
k_rep      = 3.0          # Repulsion gain scaler

CONE_COS   = 0.70         # cos(45°) -> defines 90° full forward vision cone

# ─────────────────────────────────────────────────────────────
# 2. Optimal Velocity Function
# ─────────────────────────────────────────────────────────────
def V_optimal(dx):
    if dx <= 0.0:
        return 0.0
    return v_max * (np.tanh(dx - 0.7) + np.tanh(0.7))

# ─────────────────────────────────────────────────────────────
# 3. Initial Conditions
# ─────────────────────────────────────────────────────────────
np.random.seed(12)

# Grid-jittered distribution to prevent unphysical starting overlaps
cols = int(np.ceil(np.sqrt(N)))
rows = int(np.ceil(N / cols))
spacing = min(7.5 / cols, 8.0 / rows)

positions = []
for r in range(rows):
    for c in range(cols):
        if len(positions) >= N:
            break
        px = 1.0 + c * spacing + 0.15 * np.random.randn()
        py = 1.0 + r * spacing + 0.15 * np.random.randn()
        px = np.clip(px, 0.3, L - 0.3)
        py = np.clip(py, 0.3, L - 0.3)
        positions.append([px, py])

x  = np.array([p[0] for p in positions], dtype=float)
y  = np.array([p[1] for p in positions], dtype=float)
vx = np.zeros(N)
vy = np.zeros(N)
active = np.ones(N, dtype=bool)

door_available_time = 0.0   # Queue server availability timestamp

# Historical state arrays for rendering animation frames
X_hist      = np.zeros((steps, N))
Y_hist      = np.zeros((steps, N))
Active_hist = np.zeros((steps, N), dtype=bool)

# ─────────────────────────────────────────────────────────────
# 4. Main Discretized Simulation Loop
# ─────────────────────────────────────────────────────────────
for t in range(steps):
    current_time = t * dt

    X_hist[t, :]      = x
    Y_hist[t, :]      = y
    Active_hist[t, :] = active

    if not np.any(active):
        X_hist[t:, :]      = x
        Y_hist[t:, :]      = y
        Active_hist[t:, :] = active
        break

    # ── 4a. Door Queue Serialization ──
    if current_time >= door_available_time:
        candidates  = []
        cand_dists  = []
        for i in range(N):
            if not active[i]:
                continue
            d = np.sqrt((x[i] - door_pos[0])**2 + (y[i] - door_pos[1])**2)
            if d <= 0.50:
                candidates.append(i)
                cand_dists.append(d)

        if candidates:
            winner = candidates[int(np.argmin(cand_dists))]
            active[winner]   = False
            x[winner]        = L + 2.5    # Shift cleanly off-screen
            y[winner]        = L / 2.0
            vx[winner]       = 0.0
            vy[winner]       = 0.0
            door_available_time = current_time + door_delay

    # ── 4b. Discretized Equations of Motion Updates ──
    for i in range(N):
        if not active[i]:
            continue

        pos_i      = np.array([x[i], y[i]])
        to_door    = door_pos - pos_i
        d_to_door  = np.linalg.norm(to_door)
        heading    = to_door / (d_to_door + 1e-9)

        # ── Anisotropic Forward Look-Ahead ──
        min_clearance = 5.0
        for j in range(N):
            if j == i or not active[j]:
                continue

            pos_j     = np.array([x[j], y[j]])
            vec_to_j  = pos_j - pos_i
            d_to_j    = np.linalg.norm(vec_to_j)
            if d_to_j < 1e-9:
                continue

            # Condition 1: Verify alignment falls inside the forward sight cone
            alignment = np.dot(heading, vec_to_j / d_to_j)
            if alignment <= CONE_COS:
                continue

            # Condition 2: Ensure neighbor is strictly between agent and the door
            d_j_to_door = np.linalg.norm(door_pos - pos_j)
            if d_j_to_door >= d_to_door:
                continue

            clearance = d_to_j - 0.3
            if clearance < min_clearance:
                min_clearance = clearance

        target_speed = V_optimal(min_clearance)

        # Halt target movement if the single-file server door is locked
        if d_to_door <= 0.50 and current_time < door_available_time:
            target_speed = 0.0

        v_des_x = target_speed * heading[0]
        v_des_y = target_speed * heading[1]

        # ── Isotropic Neighbor Repulsion ──
        rep_x, rep_y = 0.0, 0.0
        for j in range(N):
            if j == i or not active[j]:
                continue
            dx_ij = x[i] - x[j]
            dy_ij = y[i] - y[j]
            d_ij  = np.sqrt(dx_ij**2 + dy_ij**2)
            if 0.0 < d_ij < R_rep:
                strength = k_rep * (R_rep - d_ij) / (d_ij + 1e-9)
                rep_x   += strength * dx_ij
                rep_y   += strength * dy_ij

        # ── Euler-Discretized ODE Steps ──
        # dv/dt = alpha * (V_des - v)
        vx[i] += alpha * (v_des_x + rep_x - vx[i]) * dt
        vy[i] += alpha * (v_des_y + rep_y - vy[i]) * dt

        # dx/dt = v
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt

        # ── Continuous Structural Boundaries ──
        if x[i] < 0.05:    x[i] = 0.05;    vx[i] = max(0.0, vx[i])
        if y[i] < 0.05:    y[i] = 0.05;    vy[i] = max(0.0, vy[i])
        if y[i] > L - 0.05: y[i] = L - 0.05; vy[i] = min(0.0, vy[i])

        # Continuous right perimeter wall clamp
        if x[i] >= L - 0.05:
            x[i] = L - 0.05
            vx[i] = min(0.0, vx[i])

# ─────────────────────────────────────────────────────────────
# 5. Animation Compilation Pipeline
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Render uniform continuous boundary walls
ax.plot([0, L], [0, 0], 'k-', linewidth=2.5)
ax.plot([0, 0], [0, L], 'k-', linewidth=2.5)
ax.plot([0, L], [L, L], 'k-', linewidth=2.5)
ax.plot([L, L], [0, L], 'k-', linewidth=2.5)

# Single-file point door indicator placed directly on the perimeter line
ax.plot(door_pos[0], door_pos[1], 'ro', markersize=10, label='Exit Point Sink', zorder=5)
ax.annotate('', xy=(L + 0.6, L / 2), xytext=(L, L / 2), arrowprops=dict(arrowstyle='->', color='green', lw=2))

scat = ax.scatter([], [], c='royalblue', s=55, edgecolor='k', zorder=3)

ax.set_xlim(-0.5, L + 1.8)
ax.set_ylim(-0.5, L + 0.5)
ax.set_xlabel("Room Width (m)")
ax.set_ylabel("Room Length (m)")
ax.legend(loc='upper left', fontsize=9)

def init_anim():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update_anim(frame):
    mask   = Active_hist[frame, :]
    curr_x = X_hist[frame, mask]
    curr_y = Y_hist[frame, mask]
    offsets = np.column_stack((curr_x, curr_y)) if curr_x.size else np.empty((0, 2))
    scat.set_offsets(offsets)

    n_left = int(np.sum(mask))
    ax.set_title(f"Single-File Evacuation  |  t = {frame * dt:.1f} s\nStudents Inside: {n_left}/{N}")
    return scat,

ani = FuncAnimation(fig, update_anim, init_func=init_anim, frames=steps, blit=True, interval=30, repeat=False)
plt.close(fig)
HTML(ani.to_jshtml())

from matplotlib.animation import PillowWriter
import os

# Create an assets directory if it doesn't already exist
os.makedirs("assets", exist_ok=True)

# Save the animation as a GIF
ani.save("assets/evacuation_sim.gif", writer=PillowWriter(fps=30))
print("Animation saved successfully to assets/evacuation_sim.gif!")