import math
import random
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from board import *


class Problem:
    def __init__(self, board: Board, intersection_weight: float, paths_length_weight: float,
                 segmnets_number_weight: float, paths_outside_board_number_weight: float,
                 backwards_prob: float = 0.5, noise_prob: float = 0.1):
        self.board = board
        self.intersection_weight = intersection_weight
        self.paths_length_weight = paths_length_weight
        self.segmnets_number_weight = segmnets_number_weight
        self.paths_outside_board_number_weight = paths_outside_board_number_weight
        self.backwards_prob = backwards_prob
        self.noise_prob = noise_prob

    def random_method(self) -> Specimen:
        while True:
            specimen = self.__random_specimen()
            if Problem.__count_intersections(specimen.paths)[0] == 0:
                break
        return specimen

    def random_population(self, population_size: int) -> List[Specimen]:
        return [self.__random_specimen() for _ in range(population_size)]

    def tournament_selection(self, population: List[Specimen], size: int) -> Specimen:
        chosen_specimens = Problem.__choose_random_specimens(population, size)
        best_specimen = self.choose_best_specimen(chosen_specimens)
        return best_specimen

    def rate_specimen(self, specimen: Specimen) -> float:
        intersections_count, points = Problem.__count_intersections(specimen.paths)
        intersections = intersections_count * self.intersection_weight
        paths_length = Problem.__sum_paths_length(specimen.paths) * self.paths_length_weight
        segmets_number = Problem.__count_segments(specimen.paths) * self.segmnets_number_weight
        paths_outside_board_number = Problem.__count_paths_outside_board(self.board,
                                                                         points) * self.paths_outside_board_number_weight

        return intersections + paths_length + segmets_number + paths_outside_board_number

    def roulette_selection(self, population: List[Specimen]) -> Specimen:
        rates_sum = self.__sum_rates(population)
        population_with_intervals = self.__get_population_intervals(population, rates_sum[1], rates_sum[0])
        random_number = random.uniform(0, 1)
        for specimen_with_intervals in population_with_intervals:
            if specimen_with_intervals[1] <= random_number <= specimen_with_intervals[2]:
                return specimen_with_intervals[0]

    @staticmethod
    def crossover_operator(parents: Tuple[Specimen, Specimen], crossover_prob: float) -> Specimen:
        if crossover_prob > random.random():
            paths = []
            paths_number = len(parents[0].paths)
            separation_point = int(random.uniform(0, paths_number))
            for i in range(paths_number):
                if i + 1 <= separation_point:
                    paths.append(Problem.__copy_path(parents[0].paths[i]))
                else:
                    paths.append(Problem.__copy_path(parents[1].paths[i]))

            return Specimen(paths)
        return Problem.__copy_specimen(parents[0])

    @staticmethod
    def mutation_operator(specimen: Specimen, mutation_prob: float) -> Specimen:
        for path in specimen.paths:
            if mutation_prob > random.random():
                segment_index = int(random.uniform(0, len(path.segments)))
                shift = random.choice([-1, 1])

                # check segment behind
                if segment_index > 0:
                    if shift == 1 and (path.segments[segment_index - 1].direction == Direction.UP or
                                       path.segments[segment_index - 1].direction == Direction.RIGHT):
                        Problem.__lengthen_segment(path, segment_index, segment_index - 1, 1)
                    elif shift == 1:
                        segment_index = Problem.__lengthen_segment(path, segment_index, segment_index - 1, -1)
                    elif shift == -1 and (path.segments[segment_index - 1].direction == Direction.UP or
                                          path.segments[segment_index - 1].direction == Direction.RIGHT):
                        segment_index = Problem.__lengthen_segment(path, segment_index, segment_index - 1, -1)
                    else:
                        Problem.__lengthen_segment(path, segment_index, segment_index - 1, 1)
                else:
                    if shift == 1 and (path.segments[segment_index].direction == Direction.RIGHT or
                                       path.segments[segment_index].direction == Direction.LEFT):
                        Problem.__insert_segment(path, 0, Direction.UP)
                    elif shift == -1 and (path.segments[segment_index].direction == Direction.RIGHT or
                                          path.segments[segment_index].direction == Direction.LEFT):
                        Problem.__insert_segment(path, 0, Direction.DOWN)
                    elif shift == 1:
                        Problem.__insert_segment(path, 0, Direction.RIGHT)
                    else:
                        Problem.__insert_segment(path, 0, Direction.LEFT)
                    segment_index += 1

                # check segment after
                if segment_index < len(path.segments) - 1:
                    if shift == 1 and (path.segments[segment_index + 1].direction == Direction.UP or
                                       path.segments[segment_index + 1].direction == Direction.RIGHT):
                        Problem.__lengthen_segment(path, segment_index, segment_index + 1, -1)
                    elif shift == 1:
                        Problem.__lengthen_segment(path, segment_index, segment_index + 1, 1)
                    elif shift == -1 and (path.segments[segment_index + 1].direction == Direction.UP or
                                          path.segments[segment_index + 1].direction == Direction.RIGHT):
                        Problem.__lengthen_segment(path, segment_index, segment_index + 1, 1)
                    else:
                        Problem.__lengthen_segment(path, segment_index, segment_index + 1, -1)
                else:
                    if shift == 1 and (path.segments[segment_index].direction == Direction.RIGHT or
                                       path.segments[segment_index].direction == Direction.LEFT):
                        Problem.__insert_segment(path, len(path.segments), Direction.DOWN)
                    elif shift == -1 and (path.segments[segment_index].direction == Direction.RIGHT or
                                          path.segments[segment_index].direction == Direction.LEFT):
                        Problem.__insert_segment(path, len(path.segments), Direction.UP)
                    elif shift == 1:
                        Problem.__insert_segment(path, len(path.segments), Direction.LEFT)
                    else:
                        Problem.__insert_segment(path, len(path.segments), Direction.RIGHT)

                Problem.__delete_empty(path)
                Problem.__merge_directions(path)
                Problem.__merge_directions(path)

        return specimen

    @staticmethod
    def __delete_empty(path):
        segments_to_delete = []
        for segment in path.segments:
            if segment.length == 0:
                segments_to_delete.append(segment)
        for segment in segments_to_delete:
            path.segments.remove(segment)

    @staticmethod
    def __merge_directions(path):
        for i in range(len(path.segments)):
            if i + 1 != len(path.segments) and Problem.__is_same_direction(path.segments[i], path.segments[i + 1]):
                path.segments[i].length += path.segments[i + 1].length
                path.segments[i + 1].length = 0
            elif i + 1 != len(path.segments) and Problem.__is_opposite_direction(path.segments[i], path.segments[i + 1]):
                path.segments[i].length -= path.segments[i + 1].length
                Problem.__fix_negative_length(path.segments[i])
                path.segments[i + 1].length = 0
        Problem.__delete_empty(path)

    @staticmethod
    def __is_opposite_direction(segment1: Segment, segment2: Segment):
        if Problem.__is_same_direction(segment1, segment2):
            return False
        return (Problem.__is_vertical(segment1.direction) and (Problem.__is_vertical(segment2.direction)) or
                (not Problem.__is_vertical(segment1.direction) and not Problem.__is_vertical(segment2.direction)))

    @staticmethod
    def __is_same_direction(segment1: Segment, segment2: Segment):
        return segment1.direction == segment2.direction

    @staticmethod
    def __fix_negative_length(segment):
        if segment.length < 0:
            direction = (segment.direction + 2) % 4
            if direction == 0:
                direction = 4
            segment.direction = Direction(direction)
            segment.length = -segment.length

    @staticmethod
    def __insert_segment(path: Path, position: int, direction: Direction):
        path.segments.insert(position, Segment(1, direction))

    @staticmethod
    def __lengthen_segment(path: Path, original_index: int, index: int, length: int) -> int:
        path.segments[index].length += length
        return original_index

    @staticmethod
    def __copy_specimen(specimen: Specimen) -> Specimen:
        paths_copy = [Problem.__copy_path(path) for path in specimen.paths]
        return Specimen(paths_copy)

    @staticmethod
    def __copy_path(path: Path) -> Path:
        segments_copy = [Segment(segment.length, segment.direction) for segment in path.segments]
        return Path(path.starting_point, segments_copy)

    def __sum_rates(self, population: List[Specimen]):
        rates_sum, specimens_rates = 0, []
        for specimen in population:
            specimen_rate = self.rate_specimen(specimen)
            specimens_rates.append(specimen_rate)
            rates_sum += 1 / specimen_rate
        return rates_sum, specimens_rates

    @staticmethod
    def __get_population_intervals(population: List[Specimen], specimens_rates: List[float], rates_sum: float):
        population_with_intervals = []
        curr_interval = 0
        for i in range(len(population)):
            next_interval = curr_interval + (1 / specimens_rates[i]) / rates_sum
            population_with_intervals.append((population[i], curr_interval, next_interval))
            curr_interval = next_interval
        return population_with_intervals

    @staticmethod
    def __choose_random_specimens(population: List[Specimen], size: int) -> List[Specimen]:
        copy = list(population)
        random.shuffle(copy)
        chosen_specimens = []
        for i in range(size):
            chosen_specimens.append(copy[i])
        return chosen_specimens

    def choose_best_specimen(self, chosen_specimens: List[Specimen]) -> Specimen:
        best_specimen, best_specimen_rate = None, math.inf
        for specimen in chosen_specimens:
            curr_specimen_rate = self.rate_specimen(specimen)
            if curr_specimen_rate < best_specimen_rate:
                best_specimen = specimen
                best_specimen_rate = curr_specimen_rate
        return best_specimen

    def __random_specimen(self) -> Specimen:
        paths = []
        for point_pair in self.board.point_pairs:
            path = self.__random_path(self.board, point_pair)
            while Problem.__count_intersections([path])[0] != 0:
                path = self.__random_path(self.board, point_pair)
            paths.append(path)
        return Specimen(paths)

    def __random_path(self, board: Board, point_pair: Tuple[Point, Point]) -> Path:
        segments = []
        curr_pos_x, curr_pos_y = point_pair[0].x, point_pair[0].y
        dest_pos_x, dest_pos_y = point_pair[1].x, point_pair[1].y
        prev_direction = Problem.__random_direction()

        prev_max_dist = Problem.__get_dist_to_dest(Direction.UP, curr_pos_x, curr_pos_y, dest_pos_x, dest_pos_y)
        if prev_max_dist == 0:
            prev_max_dist = Problem.__get_dist_to_dest(Direction.RIGHT, curr_pos_x, curr_pos_y, dest_pos_x, dest_pos_y)

        while not (curr_pos_x == dest_pos_x and curr_pos_y == dest_pos_y):
            backwards = Problem.__random_truth(self.backwards_prob)
            direction = Problem.__get_direction(backwards, prev_direction, curr_pos_x, curr_pos_y, dest_pos_x,
                                                dest_pos_y)
            max_dist = Problem.__get_dist_to_dest(direction, curr_pos_x, curr_pos_y, dest_pos_x, dest_pos_y)
            max_dist = Problem.__add_noise(max_dist, prev_max_dist, self.noise_prob)
            dist = Problem.__random_dist(max_dist)
            prev_pos_x, prev_pos_y = curr_pos_x, curr_pos_y
            curr_pos_x, curr_pos_y = Problem.__calc_new_pos(dist, direction, curr_pos_x, curr_pos_y)
            curr_pos_x, curr_pos_y, dist = Problem.__check_and_fix_position(prev_pos_x, prev_pos_y, curr_pos_x,
                                                                            curr_pos_y, board, dist)
            prev_direction = direction
            prev_max_dist = max_dist
            Problem.__save_segment(segments, dist, direction)
        return Path(point_pair[0], segments)

    @staticmethod
    def __random_direction() -> Direction:
        return random.choice([Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT])

    @staticmethod
    def __random_truth(probability: float) -> bool:
        return True if (random.random() * 100 < probability * 100) else False

    @staticmethod
    def __get_direction(backwards: bool, last_direction: Direction, curr_pos_x: int, curr_pos_y: int, dest_pos_x: int,
                        dest_pos_y: int) -> Direction:
        if Problem.__is_vertical(last_direction):
            if curr_pos_x < dest_pos_x:
                return Direction.LEFT if backwards else Direction.RIGHT
            else:
                return Direction.RIGHT if backwards else Direction.LEFT
        else:
            if curr_pos_y < dest_pos_y:
                return Direction.DOWN if backwards else Direction.UP
            else:
                return Direction.UP if backwards else Direction.DOWN

    @staticmethod
    def __is_vertical(direction: Direction) -> bool:
        return direction == Direction.DOWN or direction == Direction.UP

    @staticmethod
    def __get_dist_to_dest(direction: Direction, curr_pos_x: int, curr_pos_y: int, dest_pos_x: int,
                           dest_pos_y: int) -> int:
        return abs(curr_pos_y - dest_pos_y) if Problem.__is_vertical(direction) else abs(curr_pos_x - dest_pos_x)

    @staticmethod
    def __add_noise(max_dist: int, prev_max_dist: int, noise_prob: float) -> int:
        max_dist = prev_max_dist if max_dist == 0 else max_dist
        return max_dist * 2 if Problem.__random_truth(noise_prob) else max_dist

    @staticmethod
    def __random_dist(max_dist: int) -> int:
        return math.floor(random.random() * max_dist) + 1

    @staticmethod
    def __calc_new_pos(dist: int, direction: Direction, curr_pos_x: int, curr_pos_y: int) -> Tuple[int, int]:
        if direction == Direction.UP:
            curr_pos_y += dist
        elif direction == Direction.DOWN:
            curr_pos_y -= dist
        elif direction == Direction.RIGHT:
            curr_pos_x += dist
        else:
            curr_pos_x -= dist

        return curr_pos_x, curr_pos_y

    @staticmethod
    def __check_and_fix_position(prev_pos_x: int, prev_pos_y: int, curr_pos_x: int, curr_pos_y: int, board: Board,
                                 dist: int) -> Tuple[int, int, int]:
        if curr_pos_x > board.x_size:
            curr_pos_x = board.x_size
            dist = abs(prev_pos_x - board.x_size)
        elif curr_pos_x < 0:
            curr_pos_x = 0
            dist = prev_pos_x
        elif curr_pos_y > board.y_size:
            curr_pos_y = board.y_size
            dist = abs(prev_pos_y - board.y_size)
        elif curr_pos_y < 0:
            curr_pos_y = 0
            dist = prev_pos_y
        return curr_pos_x, curr_pos_y, dist

    @staticmethod
    def __save_segment(segments: List[Segment], dist: int, direction: Direction):
        if dist > 0:
            if segments and segments[-1].direction == direction:
                segments[-1].length += dist
            elif segments and ((Problem.__is_vertical(segments[-1].direction) and Problem.__is_vertical(direction)) or (
                    not Problem.__is_vertical(segments[-1].direction) and not Problem.__is_vertical(direction))):
                segments[-1].length -= dist
            else:
                segments.append(Segment(dist, direction))

    @staticmethod
    def __count_intersections(paths: List[Path]) -> Tuple[int, Dict[Tuple[int, int], int]]:
        points = {}
        intersections = 0
        for path in paths:
            last_point = path.starting_point
            if points.get((last_point.x, last_point.y)):
                intersections += 1
            else:
                points[(last_point.x, last_point.y)] = 1
            for segment in path.segments:
                for i in range(segment.length):
                    if segment.direction == Direction.RIGHT:
                        x = last_point.x + 1
                    elif segment.direction == Direction.LEFT:
                        x = last_point.x - 1
                    else:
                        x = last_point.x
                    if segment.direction == Direction.UP:
                        y = last_point.y + 1
                    elif segment.direction == Direction.DOWN:
                        y = last_point.y - 1
                    else:
                        y = last_point.y
                    last_point = Point(x, y)

                    if points.get((last_point.x, last_point.y)):
                        intersections += 1
                    else:
                        points[(last_point.x, last_point.y)] = 1
        return intersections, points

    @staticmethod
    def __sum_paths_length(paths: List[Path]) -> int:
        paths_length = 0

        for path in paths:
            for segment in path.segments:
                paths_length += segment.length

        return paths_length

    @staticmethod
    def __count_segments(paths: List[Path]) -> int:
        segments_number = 0

        for path in paths:
            segments_number += len(path.segments)

        return segments_number

    @staticmethod
    def __count_paths_outside_board(board: Board, points: Dict[Tuple[int, int], int]) -> int:
        paths_outside_board_number = 0

        for point in points.keys():
            if point[0] < 0 or point[0] > board.x_size or point[1] < 0 or point[1] > board.y_size:
                paths_outside_board_number += 1

        return paths_outside_board_number


def draw_specimen(specimen: Specimen, board: Board):
    colors = ['#800080', '#FF00FF', '#000080', '#0000FF', '#008080', '#00FFFF', '#008000', '#00FF00',
              '#FFFF00', '#800000', '#FF0000']
    color_iter = 0

    x_values = [0, board.x_size]
    y_values = [0, 0]
    plt.plot(x_values, y_values, 'k')
    x_values = [0, board.x_size]
    y_values = [board.y_size, board.y_size]
    plt.plot(x_values, y_values, 'k')
    x_values = [0, 0]
    y_values = [0, board.y_size]
    plt.plot(x_values, y_values, 'k')
    x_values = [board.x_size, board.x_size]
    y_values = [0, board.y_size]
    plt.plot(x_values, y_values, 'k')

    for path in specimen.paths:
        x2, y2 = path.starting_point.x, path.starting_point.y
        plt.scatter(x2, y2, color=colors[color_iter])

        for segment in path.segments:
            x1 = x2
            y1 = y2

            if segment.direction == Direction.UP:
                y2 = y1 + segment.length
                x2 = x1
            elif segment.direction == Direction.DOWN:
                y2 = y1 - segment.length
                x2 = x1
            elif segment.direction == Direction.RIGHT:
                y2 = y1
                x2 = x1 + segment.length
            else:
                y2 = y1
                x2 = x1 - segment.length

            x_values = [x1, x2]
            y_values = [y1, y2]
            plt.plot(x_values, y_values, colors[color_iter])
        plt.scatter(x2, y2, color=colors[color_iter])
        color_iter += 1
        if color_iter == len(colors):
            color_iter = 0

    plt.xticks(np.arange(0, board.x_size + 1, 1.0))
    plt.yticks(np.arange(0, board.y_size + 1, 1.0))
    plt.grid()
    plt.show()
