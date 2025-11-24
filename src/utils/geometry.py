import numpy as np
from src.utils.get_config import get_system_config

config = get_system_config()

R_SATELLITE = config["satellite"]["height"] + config["physics"]["r_earth"]
D_MAX = config["satellite"]["d_max"]
N_NEIGHBOURS = config["satellite"]["n_neighbours"]


def convert_polar_to_cartesian(theta: np.ndarray, phi: np.ndarray, radius: float):
    """
    Convert polar coordinates (theta, phi) to Cartesian coordinates (x, y, z).
    """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z]).T


def convert_cartesian_to_polar(cartesian_coords: np.ndarray, radius: float):
    """
    Convert Cartesian coordinates (x, y, z) to polar coordinates (theta, phi).
    """
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(z / radius)
    return np.array([theta, phi]).T


def find_neighbours(cur_idx, dst_pos, satellite_positions):
    """
    Find neighbours of current satellite cur_idx that:
      - are within D_MAX
      - form a forward path toward dst_pos (angle condition)
    Return a list of original indices (in 0..N-1) of up to N_NEIGHBOURS nearest neighbours.
    """
    # All positions
    cur_pos = satellite_positions[cur_idx]  # (3,)

    # Compute vectors from current satellite to all others
    neighbour_vectors = satellite_positions - cur_pos  # (N, 3)
    distances = np.linalg.norm(neighbour_vectors, axis=1)  # (N,)

    # Exclude self
    mask_not_self = np.ones(len(distances), dtype=bool)
    mask_not_self[cur_idx] = False

    # Communication range filter (exclude self)
    within_range = distances < D_MAX
    within_range = within_range & mask_not_self

    # Forward path condition
    dst_vector = dst_pos - cur_pos
    angle_limit = config["satellite"]["angle_limit"] * np.pi / 180.0
    # compute dot product for all (including self, which will be masked)
    dots = np.dot(neighbour_vectors, dst_vector)  # (N,)
    forward = dots > (np.linalg.norm(neighbour_vectors, axis=1) * np.linalg.norm(dst_vector) * np.cos(angle_limit))
    forward = forward & mask_not_self

    # Combined filter
    candidate_mask = within_range & forward
    candidate_indices = np.where(candidate_mask)[0]  # original indices

    if candidate_indices.size == 0:
        return []

    # sort the candidates by distance (using original distances)
    sorted_idx = candidate_indices[np.argsort(distances[candidate_indices])]

    # return up to N_NEIGHBOURS
    return sorted_idx[:N_NEIGHBOURS].tolist()


def compute_euclidean_distance(cur_pos, dst_pos):
    """
    Compute the Euclidean distance between two points in 3D space.
    """
    return np.linalg.norm(dst_pos - cur_pos).item()


def compute_arc_length(cur_pos, dst_pos):
    """
    Compute the exact great-circle distance (arc length) between two points on a sphere.
    Uses the spherical law of cosines:
        arc = R * arccos( sinφ1*sinφ2 + cosφ1*cosφ2*cos(Δλ) )
    where φ = colatitude (0 at north pole), λ = longitude.
    """
    # Convert Cartesian to polar (θ = longitude, φ = colatitude)
    positions = np.array([cur_pos, dst_pos])
    polar_coords = convert_cartesian_to_polar(positions, R_SATELLITE)
    theta1, phi1 = polar_coords[0]
    theta2, phi2 = polar_coords[1]

    # Spherical law of cosines
    cos_gamma = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(theta2 - theta1)
    # Numerical stability
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)

    arc_length = R_SATELLITE * np.arccos(cos_gamma)
    return arc_length.item()
