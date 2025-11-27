import numpy as np
from src.utils.get_config import get_system_config

config = get_system_config()

R_SATELLITE = config["satellite"]["height"] + config["physics"]["r_earth"]
D_MAX = config["satellite"]["d_max"]
N_NEIGHBOURS = config["satellite"]["n_neighbours"]


def convert_polar_to_cartesian(theta: np.ndarray, phi: np.ndarray, radius: float):
    """
    Convert polar coordinates (theta, phi) to Cartesian coordinates (x, y, z).
    theta: azimuth (0..2π), phi: colatitude (0..π)
    """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def convert_cartesian_to_polar(cartesian_coords: np.ndarray):
    """
    Convert Cartesian coordinates (x, y, z) to polar coordinates (theta, phi).
    theta: azimuth, phi: colatitude
    """
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    r = np.linalg.norm(cartesian_coords, axis=1)
    z_norm = np.clip(z / r, -1.0, 1.0)
    phi = np.arccos(z_norm)
    theta = np.arctan2(y, x)
    return np.column_stack([theta, phi])


def find_neighbours(cur_idx, dst_pos, satellite_positions):
    """
    Find neighbours of current satellite cur_idx that:
      - are within D_MAX
      - form a forward path toward dst_pos (angle condition)
    Return a list of original indices (in 0..N-1) of up to N_NEIGHBOURS nearest neighbours.
    """
    cur_pos = satellite_positions[cur_idx]
    neighbour_vectors = satellite_positions - cur_pos
    distances = np.linalg.norm(neighbour_vectors, axis=1)

    mask_not_self = np.ones(len(distances), dtype=bool)
    mask_not_self[cur_idx] = False

    within_range = (distances < D_MAX) & mask_not_self

    dst_vector = dst_pos - cur_pos
    angle_limit = config["satellite"]["angle_limit"] * np.pi / 180.0
    dots = np.dot(neighbour_vectors, dst_vector)
    forward = dots > (np.linalg.norm(neighbour_vectors, axis=1) * np.linalg.norm(dst_vector) * np.cos(angle_limit))
    forward = forward & mask_not_self

    candidate_mask = within_range & forward
    candidate_indices = np.where(candidate_mask)[0]

    if candidate_indices.size == 0:
        raise ValueError("No valid neighbours found. Increase environment configurations.")

    sorted_idx = candidate_indices[np.argsort(distances[candidate_indices])]
    return sorted_idx[:N_NEIGHBOURS].tolist()


def compute_euclidean_distance(cur_pos, dst_pos):
    """
    Compute the Euclidean distance between two points in 3D space.
    """
    return np.linalg.norm(dst_pos - cur_pos).item()


def compute_arc_length(cur_pos, dst_pos):
    """
    Compute the exact great-circle distance (arc length) between two points on a sphere.
    Pure geometry using Cartesian vectors.
    """
    cur_pos = np.asarray(cur_pos)
    dst_pos = np.asarray(dst_pos)

    if np.allclose(cur_pos, dst_pos, atol=1e-9):
        return 0.0

    cos_gamma = np.dot(cur_pos, dst_pos) / (np.linalg.norm(cur_pos) * np.linalg.norm(dst_pos))
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    return R_SATELLITE * np.arccos(cos_gamma).item()
