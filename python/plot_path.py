import csv
import os

import matplotlib.pyplot as plt

# Path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sim_path.csv")

def load_path(csv_path: str):
    xs = []
    ys = []
    times = []
    steps = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            times.append(float(row["time"]))
            steps.append(int(row["step"]))
    return xs, ys, times, steps

def main():
    if not os.path.exists(DATA_PATH):
        print(f"CSV file not found at: {DATA_PATH}")
        print("Did you run the C++ simulation first?")
        return

    xs, ys, times, steps = load_path(DATA_PATH)

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title("AMR Delivery Robot Path")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
