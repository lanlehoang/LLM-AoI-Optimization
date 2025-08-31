"""
Define classes to be used in the env.py file
"""

import heapq  # Used for event priority queue
from typing import List, Optional, TypedDict
import numpy as np


class Event:
    def __init__(self, package_id, event_time, src):
        self.package_id = package_id
        self.event_time = event_time
        self.src = src

    def __lt__(self, other):
        # For priority queue, events with earlier time come first
        return self.event_time < other.event_time


class EventQueue:
    def __init__(self, events: List[Event]):
        self.events = events
        heapq.heapify(self.events)

    def push(self, new_event):
        heapq.heappush(self.events, new_event)

    def pop(self):
        return heapq.heappop(self.events)


class Satellite:
    def __init__(self, position, processing_rate, queue_length=0):
        self.position = position
        self.processing_rate = processing_rate
        self.queue_length = queue_length
        self.busy_time = 0  # This satellite will be busy until this time

    def enqueue_package(self, processing_time):
        """
        Increment the queue length of the satellite.
        Add processing_time to busy time
        """
        self.queue_length += 1
        self.busy_time += processing_time

    def dequeue_package(self):
        """
        Decrement the queue length of the satellite.
        """
        if self.queue_length > 0:
            self.queue_length -= 1
        else:
            raise ValueError("Queue is empty, cannot dequeue package.")

    def reset_queue(self):
        """
        Reset the satellite's queue length and busy time to zero.
        """
        self.queue_length = 0
        self.busy_time = 0

    def set_queue_length(self, length):
        """
        Set the queue length of the satellite to a specific value.
        Only use for the first satellite when initializing the environment.
        """
        if length < 0:
            raise ValueError("Queue length cannot be negative.")
        self.queue_length = length


class Package:
    def __init__(self, generation_time):
        self.generation_time = generation_time
        self.end_time = None  # To be set when processing is complete

    def record_end_time(self, end_time):
        """
        Record the end time of the package processing.
        """
        self.end_time = end_time


class Experience(TypedDict):
    """
    A single experience tuple (s, a, r, s', done).
    Fields can be None if not yet available.
    """

    state: Optional[np.ndarray]
    action: Optional[int]
    reward: Optional[float]
    next_state: Optional[np.ndarray]
    done: Optional[bool]


class ExperienceBuffer:
    """
    Store incomplete experiences e = (s, a, r, s').
    Necessary because s' IS NOT IMMEDIATELY AVAILABLE after taking action a.
    The next time the same package is processed, we can update s' and push it out of the buffer.
    """

    def __init__(self):
        self.buffer = {}

    def add_experience(self, package_id, experience: Experience):
        """
        Add an experience to the buffer.
        """
        self.buffer[package_id] = experience

    def update_experience(self, package_id, new_experience: Experience):
        """
        Update the state of an experience in the buffer based on package_id.
        """
        if package_id in self.buffer:
            experience = self.buffer[package_id]
            for key in new_experience:
                if new_experience[key] is not None:
                    experience[key] = new_experience[key]
        else:
            raise KeyError(f"Package ID {package_id} not found in buffer.")

    def pop_experience(self, package_id):
        """
        Remove an experience from the buffer based on package_id.
        """
        if package_id in self.buffer:
            return self.buffer.pop(package_id)
        else:
            raise KeyError(f"Package ID {package_id} not found in buffer.")
