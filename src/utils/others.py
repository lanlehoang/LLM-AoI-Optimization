import pandas as pd
import numpy as np
from typing import List


def state_to_arrays(state: np.ndarray) -> List[np.ndarray]:
    """
    Convert state to 4 feature arrays
    """
    dis = state[0::4].astype(int)
    arc = state[1::4].astype(int)
    process_rate = state[2::4].astype(int)
    queue = state[3::4].astype(int)
    return [dis, arc, process_rate, queue]


def generate_prompt_data(csv_path, num_arrived=10, num_dropped=10, num_none=10):
    """
    Read data samples from CSV and format them for prompt inclusion.
    """
    df = pd.read_csv(csv_path)
    df["state_hash"] = df["state"].apply(lambda s: hash(str(s)))
    df = df.drop_duplicates(subset=["state_hash"])  # Prevent duplicate states
    df["info_type"] = df["info_type"].fillna("None").astype(str)

    def parse_float_array(s):
        # Parse arrays stored as strings in the CSV
        s = str(s)
        s = s.replace("[", " ").replace("]", " ")
        s = s.replace('"', " ").replace("'", " ")
        s = s.replace(",", " ")
        s = s.replace("\n", " ")
        parts = [p for p in s.split() if p]
        return np.array([float(x) for x in parts], dtype=float)

    target = {
        "dropped": num_dropped,
        "arrived": num_arrived,
        "None": num_none,
    }

    sampled = []
    for key, n in target.items():
        sub = df[df["info_type"] == key]
        if len(sub) > 0:
            sampled.append(sub.sample(n=min(n, len(sub)), random_state=42))

    if not sampled:
        return "Error: no samples found."

    out = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    lines = []

    def format_int(arr):
        return "[" + ",".join(str(int(x)) for x in arr) + "]"

    def format_q(arr):
        return "[" + ",".join(f"{x:.3f}" for x in arr) + "]"

    for i, row in out.iterrows():
        state = parse_float_array(row["state"])
        qvals = parse_float_array(row["q_values"])
        dis, arc, process, queue = state_to_arrays(state)
        line = (
            f"#{i+1} | "
            f"D:{format_int(dis)} A:{format_int(arc)} P:{format_int(process)} Q:{format_int(queue)} | "
            f"QV:{format_q(qvals)} | "
            f"Act:{int(row['action'])} | "
            f"R:{row['reward']:.3f} | "
            f"Info:{row['info_type']}"
        )
        lines.append(line)

    return "\n".join(lines)
