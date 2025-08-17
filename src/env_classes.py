"""
Define classes to be used in the env.py file
"""

from enum import Enum
import heapq  # Used for event priority queue
from typing import List


class EventType(Enum):
    """
    Enum for different types of events in the satellite environment.
    Types of events:
    - PROCESS: A package is processed by satellite i at time t
    - TRANSFER: A package is transferred from satellite i to j at time t
    """

    PROCESS = "process"
    TRANSFER = "transfer"


class Event:
    def __init__(self, package_id, event_time, event_type, src, dst=None):
        self.package_id = package_id
        self.event_time = event_time
        self.event_type = event_type
        self.src = src
        self.dst = dst  # Destination is None for PROCESS events

    def __lt__(self, other):
        # For priority queue, events with earlier time come first
        return self.time < other.time

    def __repr__(self):
        return (
            f"Event(time={self.time}, type={self.type}, src={self.src}, dst={self.dst})"
        )


class EventQueue:
    def __init__(self, events: List[Event]):
        heapq.heapify(events)
        self.events = events

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
        Reset the satellite's queue length to zero.
        """
        self.queue_length = 0

    def set_queue_length(self, length):
        """
        Set the queue length of the satellite to a specific value.
        Only use for the first satellite when initializing the environment.
        """
        if length < 0:
            raise ValueError("Queue length cannot be negative.")
        self.queue_length = length


class Package:
    def __init__(self, package_id, generation_time):
        self.package_id = package_id
        self.generation_time = generation_time
        self.end_time = None  # To be set when processing is complete

    def record_end_time(self, end_time):
        """
        Record the end time of the package processing.
        """
        self.end_time = end_time


class ExperienceBuffer:
    """
    Store incomplete experiences e = (s, a, r, s').
    Necessary because s' IS NOT IMMEDIATELY AVAILABLE after taking action a.
    The next time the same package is processed, we can update s' and push it out of the buffer.
    """

    def __init__(self):
        self.buffer = {}

    def add_experience(self, package_id, experience):
        """
        Add an experience to the buffer.
        """
        self.buffer[package_id] = experience

    def update_experience(self, package_id, new_state):
        """
        Update the state of an experience in the buffer based on package_id.
        """
        if package_id in self.buffer:
            experience = self.buffer[package_id]
            experience[-1] = new_state
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
