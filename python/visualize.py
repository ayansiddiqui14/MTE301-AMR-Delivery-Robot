import csv
import os
import time

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sim_path.csv")

def load_path(csv_path: str):
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return xs, ys

def main():
    if not os.path.exists(DATA_PATH):
        print(f"CSV file not found at: {DATA_PATH}")
        print("Run the C++ sim first.")
        return

    xs, ys = load_path(DATA_PATH)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("AMR Delivery Robot Animation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)

    path_line, = ax.plot([], [], marker="o")
    robot_point, = ax.plot([], [], marker="s", markersize=8)

    for i in range(len(xs)):
        path_line.set_data(xs[: i + 1], ys[: i + 1])
        robot_point.set_data(xs[i], ys[i])
        plt.pause(0.2)  # simple, not real time
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
