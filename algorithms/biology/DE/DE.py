from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
from numpy import sum
from matplotlib import pyplot
from numpy import cos
from numpy import pi
import os
import sys


# first element of bounds: bound of the first column of pop 
# or we can know that: len(bounds) = pop.columns

def crossover(mutated, target, dims, cr):  
    p = rand(dims)
    
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]

    return trial
    
def mutation(x, F): 
    return x[0] + F * (x[1] - x[2])

def check_bounds(mutated, bounds): 
    return [clip(mutated[i], bounds[i,0], bounds[i,1]) for i in range(len(bounds))]

def DE(pop_size, bounds, iter, F, cr, obj): 
    # create population of candidate solutions
    # pop = lower_pound + rand(0,1) * (upper_bound - lower_bound)
    # Cách sinh này để phủ đều không gian
    pop = bounds[:,0] + rand(pop_size, len(bounds))*(bounds[:,1] - bounds[:,0])
    obj_all = [obj(i) for i in pop] 
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # for plotting 
    obj_iter = list()
    for i in range(iter):
        for j in range(pop_size):
            # pick 3 candidates from the population except for the current target
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a,b,c = pop[choice(candidates, 3, replace = False)]
            # mutation
            mutated = mutation([a,b,c], F)
            mutated = check_bounds(mutated, bounds)
            # crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # selection
            obj_trial = obj(trial)
            obj_target= obj(pop[j])
            if (obj_trial < obj_target):
                pop[j] = trial 
                obj_all[j] = obj_trial
        best_obj = min(obj_all)
        if (best_obj < prev_obj):
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj 
            obj_iter.append(best_obj)
            print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter]
            
if __name__ == "__main__": 

    # algorithm parameters configuration
    pop_size = 10
    iter = 100
    F = 0.5
    cr = 0.7
    bounds = asarray([(-5.0,5.0), (-5.0,5.0)])

    # configure filepath
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from testing.continous.function import sphere, rastrigin, rosenbrock, griewank, ackley
    
    # choosing "obj" function to test the performance of the problem
    solution = DE(pop_size, bounds, iter, F, cr, ackley)
    print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))
    pyplot.plot(solution[2], '.-')
    pyplot.xlabel('Improvement Number')
    pyplot.ylabel('Evaluation X')
    pyplot.show()
