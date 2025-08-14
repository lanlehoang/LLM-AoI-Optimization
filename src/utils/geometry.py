import numpy as np
from src.utils.get_config import get_system_config

config = get_system_config()

R_EARTH = config['physics']['r_earth']
R_SATELLITE = config['satellite']['height'] + R_EARTH  # Radius of the satellites' orbit


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


def find_neighbours(cur_idx, cur_pos, dst_pos, satellite_positions):
    """
    Find all neighbours of the current satellite that satisfy the following conditions:
    - Line-of-sight exists: The paths between the current and neighbour satellites are not
    obstructed by the Earth.
    - Forward path: The angles made by the neighbours satellite, the current satellite,
    and the destination satellite are acute.
    """
    neighbour_vectors = satellite_positions - cur_pos
    # LoS condition
    los = np.linalg.norm(neighbour_vectors, axis=1) < R_SATELLITE

    # Forward path condition
    dst_vector = dst_pos - cur_pos
    forward = np.dot(neighbour_vectors, dst_vector) > 0

    # Combine conditions
    neighbours = np.where(los & forward)[0]
    
    # Exclude the current satellite itself
    neighbours = neighbours[neighbours != cur_idx]

    return neighbours.tolist() if len(neighbours) > 0 else None


def compute_arc_length(cur_pos, dst_pos):
    """
    Compute the arc length between two points on a sphere with radius R_SATELLITE and centre (0, 0, 0).
    cur_pos and dst_pos are 3D Cartesian coordinates.
    """
    positions = np.array([cur_pos, dst_pos])
    polar_coords = convert_cartesian_to_polar(positions, R_SATELLITE)
    theta_diff = polar_coords[1, 0] - polar_coords[0, 0]
    phi_diff = polar_coords[1, 1] - polar_coords[0, 1]

    # Ensure the angles are in the range [0, 2*pi]
    theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi
    phi_diff = (phi_diff + np.pi) % (2 * np.pi) - np.pi

    # Calculate the arc length
    arc_length = R_SATELLITE * np.sqrt(theta_diff**2 + phi_diff**2)
    return arc_length
    