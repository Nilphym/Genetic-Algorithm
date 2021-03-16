from board import Point, Board
from problem import Problem


def read_from_file(file_path: str, intersection_weight: float, paths_length_weight: float,
                   segmnets_number_weight: float, paths_outside_board_number_weight: float,
                   backwards_prob: float = 0.5, noise_prob: float = 0.1) -> Problem:
    x_size = 0
    y_size = 0
    point_pairs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].split(';')
            if i == 0:
                x_size = int(line[0]) - 1
                y_size = int(line[1]) - 1
            else:
                fst_point = Point(int(line[0]), int(line[1]))
                snd_point = Point(int(line[2]), int(line[3]))
                point_pairs.append((fst_point, snd_point))
    return Problem(Board(x_size, y_size, point_pairs), intersection_weight, paths_length_weight, segmnets_number_weight,
                   paths_outside_board_number_weight, backwards_prob, noise_prob)
