import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sim_path.csv")

def test_csv_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} does not exist. Run the C++ sim."

def test_csv_structure():
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        assert first is not None, "CSV is empty."
        for key in ["time", "x", "y", "step"]:
            assert key in first, f"Missing column: {key}"

if __name__ == "__main__":
    try:
        test_csv_exists()
        test_csv_structure()
        print("Simulation CSV looks OK ✅")
    except AssertionError as e:
        print("Test failed:", e)
