import pandas as pd
import numpy as np


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

    for i, row in out.iterrows():
        state = parse_float_array(row["state"])
        qvals = parse_float_array(row["q_values"])

        D = state[0::4].astype(int)
        A = state[1::4].astype(int)
        P = state[2::4].astype(int)
        Q = state[3::4].astype(int)

        def format_int(arr):
            return "[" + ",".join(str(int(x)) for x in arr) + "]"

        def format_q(arr):
            return "[" + ",".join(f"{x:.4f}" for x in arr) + "]"

        line = (
            f"#{i+1} | "
            f"D:{format_int(D)} A:{format_int(A)} P:{format_int(P)} Q:{format_int(Q)} | "
            f"QV:{format_q(qvals)} | "
            f"Act:{int(row['action'])} | "
            f"R:{row['reward']:.4f} | "
            f"Info:{row['info_type']}"
        )
        lines.append(line)

    return "\n".join(lines)
