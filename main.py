import timeit
from file_operations import read_from_file
from problem import *
from constants import *


def ga_iterations(problem: Problem, population_size: int, iterations: int, crossover_prob: float, mutation_prob: float) -> Specimen:
    population = problem.random_population(population_size)

    for i in range(iterations):
        new_population = []
        print("Progress:", str(i) + "/" + str(iterations))
        for _ in range(len(population)):
            parents = problem.tournament_selection(population, TOURNAMENT_SIZE), problem.tournament_selection(population, TOURNAMENT_SIZE)
            # parents = problem.roulette_selection(population), problem.roulette_selection(population)
            child = problem.crossover_operator(parents, crossover_prob)
            new_population.append(problem.mutation_operator(child, mutation_prob))
        population = new_population
        print("Current best rate: ", problem.rate_specimen(problem.choose_best_specimen(population)), '\n')

    return problem.choose_best_specimen(population)


if __name__ == "__main__":
    start = timeit.default_timer()

    problem = read_from_file(FILE, INTERSECTION_WEIGHT, PATHS_LENGTH_WEIGHT, SEGMENTS_NUMBER_WEIGHT, NUMBER_OF_PATHS_OUTSIDE_BOARD_WEIGHT)
    best_specimen = ga_iterations(problem, POPULATION_SIZE, ITERATIONS, CROSSOVER_PROB, MUTATION_PROB)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    print(problem.rate_specimen(best_specimen))
    draw_specimen(best_specimen, problem.board)
