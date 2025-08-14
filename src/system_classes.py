from enum import Enum


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
    def __init__(self, package_id, time, type, src, dst=None):
        self.package_id = package_id
        self.time = time
        self.type = type
        self.src = src
        self.dst = dst  # Destination is None for PROCESS events

    def __lt__(self, other):
        # For priority queue, events with earlier time come first
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time}, type={self.type}, src={self.src}, dst={self.dst})"
    

class Satellite:
    def __init__(self, position, processing_rate, queue_length=0):
        self.position = position
        self.processing_rate = processing_rate
        self.queue_length = queue_length

    def enqueue_package(self):
        """
        Increment the queue length of the satellite.
        """
        self.queue_length += 1

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
