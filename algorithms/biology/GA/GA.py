import numpy as np
from matplotlib import pyplot

# Binary Tournament Selection
# Two-point crossover
# Bit flip mutation

def evaluate(pop, fitness_func):
    return np.array([fitness_func(i) for i in pop])

def track_best(pop, fitness, best_sol, best_fit, history):
    """Track the best solution: individual having the most fitness_func value"""
    idx = np.argmax(fitness)
    if fitness[idx] > best_fit:
        best_fit = fitness[idx]
        best_sol = pop[idx].copy()
    history.append(best_fit)
    return best_sol, best_fit, history

def selection(pop, fitness, pop_size, k):
    """Binary Tournament Selection"""
    parents = []
    for _ in range(pop_size):
        idx = np.random.choice(pop_size, k, replace=False)
        parents.append(pop[np.argmax(fitness[idx])])
    return np.array(parents)

def crossover(parents, pop_size, chrom_len, CR):
    """Two-point crossover"""
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = parents[i], parents[min(i+1, pop_size-1)]
        if np.random.rand() < CR:
            cut1, cut2 = sorted(np.random.choice(chrom_len, 2, replace=False))
            offspring.extend([np.concatenate([p1[:cut1], p2[cut1:cut2], p1[cut2:]]),
                            np.concatenate([p2[:cut1], p1[cut1:cut2], p2[cut2:]])])
        else:
            offspring.extend([p1.copy(), p2.copy()])
    return np.array(offspring[:pop_size])

def mutation(pop, pop_size, chrom_len, mutation_rate):
    """Bit flip mutation"""
    mask = np.random.random((pop_size, chrom_len)) < mutation_rate
    return np.where(mask, 1 - pop, pop)

def apply_elitism(pop, fitness, elitism_count):
    """Select top individuals based on fitness"""
    elite_indices = np.argsort(fitness)[-elitism_count:]
    return pop[elite_indices].copy(), fitness[elite_indices].copy()

def GA(fitness_func, pop_size=200, chrom_len = 5, iter=100, mutation_rate=0.01, CR=0.9, tournament_size = 3, elitism_rate = 0.1):
    """Binary Genetic Algorithm for discrete optimization"""
    # Initialize binary population
    pop = np.random.randint(0, 2, size = (pop_size, chrom_len))
    best_sol, best_fit = None, -np.inf
    history = []

    # elitism (tinh hoa)
    elitism_count = max(1, int(pop_size * elitism_rate))
    offspring_size = pop_size - elitism_count

    for i in range(iter):
        fitness = evaluate(pop, fitness_func)
        best_sol, best_fit, history = track_best(pop, fitness, best_sol, best_fit, history)
        elite_pop, elite_fitness = apply_elitism(pop, fitness, elitism_count)


        parents = selection(pop, fitness, offspring_size, tournament_size)

        offspring = crossover(parents, offspring_size, chrom_len, CR)
        offspring = mutation(offspring, offspring_size, chrom_len, mutation_rate)

        pop = np.vstack([elite_pop, offspring])
        print('Iteration: %d f([%s]) = %.5f' % (i,best_sol,np.around(best_fit, decimals=5)))
    return best_fit, best_sol, history 

def fitness_func(x: np.array): 
    return np.sum(x)

if __name__ == "__main__": 

    # algorithm parameters configuration
    pop_size = 150
    iter = 150
    cr = 0.8
    mutation_rate = 0.01
    chrom_len = 100
    elitism_rate = 0.2
    tournament_size = 3

    solution = GA(fitness_func, pop_size, chrom_len, iter, mutation_rate, cr, tournament_size, elitism_rate)
    print('\nSolution: f([%s]) = %.5f' % (solution[1], np.around(solution[0], decimals=5)))
    pyplot.plot(solution[2], '.-')
    pyplot.xlabel('Generation')
    pyplot.ylabel('Best fitness')
    pyplot.show()

# Future features:
# history_best.append(best_fit)
# history_mean.append(np.mean(fitness))
