from enum import IntEnum
from typing import List, Tuple


class Direction(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Segment:
    def __init__(self, length: int, direction: Direction):
        self.length = length
        self.direction = direction


class Path:
    def __init__(self, starting_point: Point, segments: List[Segment]):
        self.starting_point = starting_point
        self.segments = segments


class Specimen:
    def __init__(self, paths: List[Path]):
        self.paths = paths


class Board:
    def __init__(self, x_size: int, y_size: int, point_pairs: List[Tuple[Point, Point]]):
        self.x_size = x_size
        self.y_size = y_size
        self.point_pairs = point_pairs
