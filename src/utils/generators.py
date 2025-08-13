import numpy as np

RANDOM_SEED = 42


def generate_satallite_positions(num_satellites, radius):
    """
    Generate random positions over polar coordinates for a given number of satellites.
    Convert polar coordinates to Cartesian coordinates.
    Ouput shape: (num_satellites, 3)
    """
    theta = np.random.uniform(0, 2*np.pi, num_satellites)
    phi = np.random.uniform(0, np.pi/2, num_satellites)     # Only consider the Northern hemisphere
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z]).T


def generate_satellite_processing_rates(num_satellites, lower, upper):
    """
    Generate random processing rates (mu in Poisson distribution) for a given number of satellites.
    Output shape: (num_satellites,)
    """
    return np.random.uniform(lower, upper, num_satellites)


def generate_processing_time(mu):
    """
    Generate processing times for a given mean (mu) using an exponential distribution.
    """
    return np.random.exponential(1/mu)


def choose_start_end(satellite_positions):
    """
    Choose start and end satellites for a transfer operation.
    Ensure that the angle between the two satellites is greater than 135 degrees.
    Return (start_index, end_index).
    """
    n_satellites = satellite_positions.shape[0]
    for i in range(n_satellites):
        for j in range(i + 1, satellite_positions.shape[0]):
            angle = np.arccos(
                np.dot(satellite_positions[i], satellite_positions[j]) / \
                (np.linalg.norm(satellite_positions[i]) * np.linalg.norm(satellite_positions[j]))
            )
            if angle > 3*np.pi/4:  # 135 degrees in radians
                return i, j