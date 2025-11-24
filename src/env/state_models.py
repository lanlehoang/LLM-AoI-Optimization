"""
Defines Pydantic models to validate and represent the state of neighbour satellites
"""

from typing import List, ClassVar
from pydantic import BaseModel, Field
import numpy as np
from src.utils.get_config import get_system_config

N_NEIGHBOURS = get_system_config()["satellite"]["n_neighbours"]


class NeighbourState(BaseModel):
    """
    Represents the state of a neighbour satellite, including:
    - Its Euclidean distance to the current satellite
    - Its arc length to the destination satellite
    - Its average processing rate \mu
    - Its current queue length
    """

    STATE_DIM: ClassVar[int] = 4  # distance, arc_length, processing_rate, queue_length

    distance: float = Field(..., description="Euclidean distance to the current satellite")
    arc_length: float = Field(..., description="Arc length to the destination satellite")
    processing_rate: float = Field(..., description="Average processing rate (mu)")
    queue_length: int = Field(..., description="Current queue length of the neighbour satellite")

    def to_numpy(self) -> np.ndarray:
        """
        Converts the neighbour state to a NumPy array.
        """
        return np.array(
            [
                self.distance,
                self.arc_length,
                self.processing_rate,
                self.queue_length,
            ],
            dtype=np.float32,
        )


class EnvironmentState(BaseModel):
    """
    Concatenation of exactly N_NEIGHBOURS NeighbourState instances.
    """

    STATE_DIM: ClassVar[int] = N_NEIGHBOURS * NeighbourState.STATE_DIM

    neighbours: List[NeighbourState] = Field(
        ...,
        description=f"List of exactly {N_NEIGHBOURS} neighbour states",
        min_length=N_NEIGHBOURS,
        max_length=N_NEIGHBOURS,
    )

    def to_numpy(self) -> np.ndarray:
        """
        Converts the full environment state to a 1D NumPy array by concatenating neighbour states.
        """
        state_arrays = [neighbour.to_numpy() for neighbour in self.neighbours]
        return np.concatenate(state_arrays, axis=0)
