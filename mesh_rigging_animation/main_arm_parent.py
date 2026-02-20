import csv
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
IMU_FILE = r"imu_data\Calibration_2_Douglas_20-11.txt"


def load_imu_file(path):
    # find first line containing tabs
    header_line = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                header_line = line
                break

    if header_line is None:
        raise RuntimeError("ERROR: No tab-separated header found in IMU file")

    # Now parse using DictReader starting from the correct header
    with open(path, "r", encoding="utf-8") as f:
        # skip until header
        for line in f:
            if line == header_line:
                break

        reader = csv.DictReader(f, delimiter="\t", fieldnames=header_line.strip().split("\t"))
        next(reader, None)  # skip header duplication
        return list(reader)

FPS = 25


IMU_COLUMNS = {
    "time": "Time(s):",
    "chest": ["Chest :W():", "Chest :X():", "Chest :Y():", "Chest :Z():"],
    "rightarm": ["R.Right.Arm :W():", "R.Right.Arm :X():", "R.Right.Arm :Y():", "R.Right.Arm :Z():"],
    "rightforearm": ["R.Right.Forearm :W():", "R.Right.Forearm :X():", "R.Right.Forearm :Y():",
                     "R.Right.Forearm :Z():"],
    "righthand": ["R.Right.Hand :W():", "R.Right.Hand :X():", "R.Right.Hand :Y():", "R.Right.Hand :Z():"],
}

BONES = {
    "arm": {
        "quat": (
            "R.Right.Arm :W():",
            "R.Right.Arm :X():",
            "R.Right.Arm :Y():",
            "R.Right.Arm :Z():",
        ),
        "length": 0.30,
    },
    "forearm": {
        "quat": (
            "R.Right.Forearm :W():",
            "R.Right.Forearm :X():",
            "R.Right.Forearm :Y():",
            "R.Right.Forearm :Z():",
        ),
        "length": 0.25,
    },
    "hand": {
        "quat": (
            "R.Right.Hand :W():",
            "R.Right.Hand :X():",
            "R.Right.Hand :Y():",
            "R.Right.Hand :Z():",
        ),
        "length": 0.15,
    },
}

imu_frames = load_imu_file(IMU_FILE)
print("Loaded", len(imu_frames), "frames")

print("Original IMU rows:", len(imu_frames))

# Downsample to target FPS
target_dt = 1.0 / FPS
filtered_frames = []
next_time = 0.0

t_RightArm = (0.9763, 0, 0, -0.2164)
t_RightForeArm = (0.9537, -0.3007, 0, 0)
t_RightHand = (1.0, 0, 0, 0)

for row in imu_frames:

    t = float(row[IMU_COLUMNS["time"]])

    if t >= next_time:
        filtered_frames.append(row)
        next_time += target_dt

imu_frames = filtered_frames



def quat_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v

    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)

    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )

def quat_inv(q):
    w,x,y,z = q
    return w, -x, -y, -z

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

arm_pts = []
forearm_pts = []
hand_pts = []


for row in imu_frames:

    # --- ARM ---
    q_arm_world = tuple(float(row[c]) for c in BONES["arm"]["quat"])
    arm_end = quat_rotate(q_arm_world, (0, BONES["arm"]["length"], 0))

    # --- FOREARM ---
    q_fore_world = tuple(float(row[c]) for c in BONES["forearm"]["quat"])

    # convert to local relative to arm
    q_fore_local = quat_mul(quat_inv(q_arm_world), q_fore_world)

    # rebuild world rot properly via hierarchy
    q_fore_world_correct = quat_mul(q_arm_world, q_fore_local)

    fore_local = quat_rotate(q_fore_world_correct, (0, BONES["forearm"]["length"], 0))
    fore_end = (
        arm_end[0] + fore_local[0],
        arm_end[1] + fore_local[1],
        arm_end[2] + fore_local[2],
    )

    # --- HAND ---
    q_hand_world = tuple(float(row[c]) for c in BONES["hand"]["quat"])

    q_hand_local = quat_mul(quat_inv(q_fore_world), q_hand_world)
    q_hand_world_correct = quat_mul(q_fore_world_correct, q_hand_local)

    hand_local = quat_rotate(q_hand_world_correct, (0, BONES["hand"]["length"], 0))
    hand_end = (
        fore_end[0] + hand_local[0],
        fore_end[1] + hand_local[1],
        fore_end[2] + hand_local[2],
    )

    arm_pts.append(arm_end)
    forearm_pts.append(fore_end)
    hand_pts.append(hand_end)





arm_pts = np.array(arm_pts)
forearm_pts = np.array(forearm_pts)
hand_pts = np.array(hand_pts)

t = np.arange(len(arm_pts))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sc1 = ax.scatter(
    arm_pts[:,0], arm_pts[:,1], arm_pts[:,2],
    c=t, cmap="Blues", s=6, label="Arm"
)

sc2 = ax.scatter(
    forearm_pts[:,0], forearm_pts[:,1], forearm_pts[:,2],
    c=t, cmap="Greens", s=6, label="Forearm"
)

sc3 = ax.scatter(
    hand_pts[:,0], hand_pts[:,1], hand_pts[:,2],
    c=t, cmap="Reds", s=6, label="Hand"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Right Arm Motion Path (colored by time)")
ax.set_box_aspect([1,1,1])

plt.show()