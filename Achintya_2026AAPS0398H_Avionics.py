import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Load Data
file_path = "Depth Data.csv"


df = pd.read_csv(file_path)


if "depth" in df.columns:
    depth_raw = df["depth"]
else:

    depth_raw = df.iloc[:, -1]


if "time" in df.columns:
    time_series = df["time"]
else:
    time_series = list(range(len(depth_raw)))

# Clean non-numeric/corrupted values by converting to numeric and interpolating
depth_clean = pd.to_numeric(depth_raw, errors="coerce").interpolate(method="linear")

# Noise reduction: Apply a 3-second centered moving average filter
depth_smoothed = depth_clean.rolling(window=3, min_periods=1, center=True).mean()

# 3. Visualization Setup
fig, ax = plt.subplots(figsize=(10, 6))


(line_raw,) = ax.plot([], [], color="#95a5a6", linestyle="--", alpha=0.6, label="Raw Sensor Data")
(line_smooth,) = ax.plot([], [], color="#2980b9", linewidth=2, label="Filtered Depth (Moving Avg)")

ax.set_xlim(0, len(time_series))
ax.set_ylim(depth_clean.min() - 5, depth_clean.max() + 5)
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("Depth (meters)", fontsize=12)
ax.set_title("Odysseus Vessel - Sea Floor Depth Profile", fontsize=14, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")

# 4. Animation Function
def update(frame):
    # Slice the data up to the current frame/second
    current_time = time_series[: frame + 1]
    current_raw = depth_clean[: frame + 1]
    current_smooth = depth_smoothed[: frame + 1]

    # Update line data
    line_raw.set_data(current_time, current_raw)
    line_smooth.set_data(current_time, current_smooth)

    return line_raw, line_smooth

# Animate 1 frame per second (1000 ms)
ani = FuncAnimation(
    fig,
    update,
    frames=len(time_series),
    interval=1000,
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()