from deap import base, algorithms
from deap import creator
from deap import tools
from tqdm import tqdm,tnrange
from pulp import *
from scipy.linalg import expm
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import elitism

import os
import time
import pickle , json

# set the random seed:
RANDOM_SEED = 10
random.seed(RANDOM_SEED)

import ortools


def varAnd(population, toolbox, cxpb, mutpb):
    
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook



def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    global gen
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)
        
        budget_max_min(halloffame)
        print(budget_hof)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)
    
        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('PavementDF.csv')

# Show the DataFrame
print(df.head())

import numpy as np

def method_0(x):   # do nothing / natural deterioration
    return x * np.exp(-0.05)

def method_1(current_pqi, improvement = 0.5, max_value=10):    # preservation
    return min(current_pqi + improvement, max_value)

def method_2(current_pqi, improvement = 3, max_value=10):   # rehabilitation
    return min(current_pqi + improvement, max_value)

def method_3(current_pqi, improvement = 7, max_value=10):   # reconstruction
    return min(current_pqi + improvement, max_value)



def deterioration(current_los, scale = 10, k = 0.1 ):
    return np.maximum(current_los - scale * k * (np.power(current_los,k-1)), 1e-9)

def preservation(current_pqi, improvement=0.8, max_value=10):
    return np.minimum(current_pqi + improvement, max_value)

def rehabilitation(current_pqi, improvement=3, max_value=10):
    return np.minimum(current_pqi + improvement, max_value)

def reconstruction(current_pqi, improvement=10, max_value=10):
    return np.minimum(current_pqi + improvement, max_value)



import numpy as np

action_functions = [deterioration, preservation, rehabilitation, reconstruction]

method_costs = df[['method_1_cost', 'method_2_cost', 'method_3_cost']].values
method_costs = np.hstack([np.zeros((method_costs.shape[0], 1)), method_costs])
action_functions = [deterioration, preservation, rehabilitation, reconstruction]
current_los = df['current_los'].to_numpy().copy()
weights = df['weight'].to_numpy()

n = len(df)  # Number of assets
h = 5  # Number of years
m = len(action_functions)
annual_budget = 30_000_000  # The annual budget
Tol = 0.2  # Tolerance level for the budget
annual_budget_penalty = 1.0
total_budget_penalty = 5.0


def evaluate(individual):
    
    individual = np.array(individual)
    # Initialize parameters
    current_los = df['current_los'].to_numpy().copy()

    # Initialize variables for tracking LOS and budget
    annual_los = np.zeros(h)
    annual_spend = np.zeros(h)
    total_spend = 0
    penalty = 0

    # Loop through each year to calculate the LOS and budget
    for year in range(h):
        actions_this_year = individual[year * n: (year + 1) * n]

        annual_spend[year] = np.sum(method_costs[np.arange(n), actions_this_year])
        # Check annual budget constraints
        if annual_spend[year] < annual_budget * (1 - Tol) or annual_spend[year] > annual_budget * (1 + Tol):
            penalty += annual_budget_penalty
        
        # Update total spend
        total_spend += annual_spend[year]
        for i, action_func in enumerate(action_functions):
            mask = actions_this_year == i
            current_los[mask] = action_func(current_los[mask])
        
        # Calculate the weighted average LOS for this year and store it
        annual_los[year] = np.sum(weights * current_los)/np.sum(weights)
        
    # Check total budget constraint
    if total_spend > annual_budget * h:
        penalty += total_budget_penalty

    # Calculate the average LOS across all years
    avg_los = np.mean(annual_los)
    
    # Apply penalty if any
    avg_los -= penalty
    
    return avg_los,


from ortools.linear_solver import pywraplp
import numpy as np  # Assuming you're using NumPy for numerical operations

def optimize_annual_plan(current_los = current_los, lbudget = annual_budget*(1-Tol) , ubudget = annual_budget*(1+Tol) , pert = 0):
    # Initialize solver with SCIP
    solver = pywraplp.Solver.CreateSolver('SAT')

    # Number of segments
    n = len(df)

    # Decision variables
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # Objective function: Maximize sum of PQI (or your specific objective)
    objective = solver.Objective()
    for i in range(n):
        for j in range(m): 
            objective.SetCoefficient(x[i, j], action_functions[j](current_los[i]) * weights[i] * np.random.uniform(1-pert, 1+pert))
    objective.SetMaximization()

    # Budget constraint
    budget_constraint = solver.Constraint(lbudget, ubudget)
    for i in range(n):
        for j in range(m):  # Generalized to cover all 'm' methods
            budget_constraint.SetCoefficient(x[i, j], method_costs[i, j])

    # Action constraint
    for i in range(n):
        action_constraint = solver.Constraint(0, 1)
        for j in range(m):
            action_constraint.SetCoefficient(x[i, j], 1)

    # Solve
    status = solver.Solve()

    # Extract solution
    if status == pywraplp.Solver.OPTIMAL:
        annual_plan = [next((j for j in range(m) if x[i, j].solution_value() > 0.5), 0) for i in range(n)]
        return annual_plan
    else:
        return None


def performance(individual):
    
    individual = np.array(individual)
    # Initialize parameters
    current_los = df['current_los'].to_numpy().copy()

    # Initialize variables for tracking LOS and budget
    annual_los = np.zeros(h)
    annual_spend = np.zeros(h)
    total_spend = 0
    penalty = 0
    
    print('Initial LOS', np.sum(weights * current_los)/np.sum(weights))
    # Loop through each year to calculate the LOS and budget
    for year in range(h):
        actions_this_year = individual[year * n: (year + 1) * n]
        annual_spend[year] = np.sum(method_costs[np.arange(n), actions_this_year])
        
        # Check annual budget constraints
        if annual_spend[year] < annual_budget * (1 - Tol) or annual_spend[year] > annual_budget * (1 + Tol):
            penalty += annual_budget_penalty

        # Update total spend
        total_spend += annual_spend[year]
        
        for i, action_func in enumerate(action_functions):
            mask = actions_this_year == i
            current_los[mask] = action_func(current_los[mask])
        
        # Calculate the weighted average LOS for this year and store it
        annual_los[year] = np.sum(weights * current_los)/np.sum(weights)

    # Check total budget constraint
    if total_spend > annual_budget * h:
        penalty += total_budget_penalty

    # Calculate the average LOS across all years
    avg_los = np.mean(annual_los)
    
    # Apply penalty if any
    avg_los -= penalty
    
    print(annual_spend)
    print(total_spend)
    print(annual_los)
    
    
    return avg_los,


def budget_max_min(hof):
    
    global budget_hof
    budget_hof = np.zeros((2,h))
    
    budget_matrix = np.zeros((len(hof),h))
                             
    for i, individual in enumerate(hof):
        individual = np.array(individual)
        
        # Initialize parameters
        annual_spend = np.zeros(h)
                        
        # Loop through each year to calculate the LOS and budget
        for year in range(h):
            actions_this_year = individual[year * n: (year + 1) * n]
            annual_spend[year] = np.sum(method_costs[np.arange(n), actions_this_year])
        
        budget_matrix[i,:] = annual_spend
        budget_hof[0,:] = budget_matrix.max(axis=0)
        budget_hof[1,:] = budget_matrix.min(axis=0)
        
        


import random

def annual_plan_swap_crossover(ind1, ind2):
    """Execute an Annual-Plan-Swap crossover on the input individuals.
    
    The individuals are modified in place and both are returned.
    
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple containing the two individuals with their annual plans
              for a randomly selected year swapped.
    """
    
    # Select a random year
    random_year = np.random.randint(0, h-1)  
    
    # Calculate the start and end indices for the annual plan for that year
    start_index = random_year * n
    end_index = (random_year + 1) * n
    
    # Perform the swap
    ind1[start_index:end_index], ind2[start_index:end_index] = ind2[start_index:end_index], ind1[start_index:end_index]
    
    return ind1, ind2



def annual_plan_lp_mutation(individual):
    """
    Perform Annual-Plan-LP Mutation on a given individual.
    
    :param individual: The individual to be mutated.
    :returns: A tuple containing the mutated individual.
    """
    
    # Define the feasible range for the budget
    lbudget = annual_budget * (1 - Tol)
    ubudget = annual_budget * (1 + Tol)
    
    infeasibleYear = None
    
    # Step 1: Examine each year's annual plan
    budget_list = []
    for year in range(h):
        actions_this_year = individual[year * n: (year + 1) * n]
        annual_spend = np.sum(method_costs[np.arange(n), actions_this_year])
        budget_list.append(annual_spend)
        if annual_spend < lbudget or annual_spend > ubudget:
            infeasibleYear = year
            break
    
    # Step 2: Identify which year to optimize
    if infeasibleYear is not None:
        year_to_optimize = infeasibleYear
        remained_budget = ubudget
    else:
        year_to_optimize = random.randint(0, h - 1)
        remained_budget = (annual_budget*h) - sum(budget_list) +  budget_list[year_to_optimize]
    
    # Step 3: Evolve the system states up to that year
    current_los = df['current_los'].to_numpy().copy()
    for year in range(year_to_optimize):
        actions_this_year = individual[year * n: (year + 1) * n]
        for i, action_func in enumerate(action_functions):
            mask = actions_this_year == i
            current_los[mask] = action_func(current_los[mask])
    
    # Step 4: Randomly select a new budget value for the year in optimize
    new_budget = random.uniform(lbudget , ubudget)
    if gen > h+1:
        new_budget = random.uniform(budget_hof[1,year_to_optimize]*(1-Tol/gen), budget_hof[0,year_to_optimize]*(1+Tol/gen))
    

    new_budget = min(max(new_budget, lbudget*1.01), remained_budget)
    
    # Step 5: Use LP to optimize the plan for the year in focus
    optimal_plan = greedy_annual_plan(current_los = current_los, lbudget = lbudget , ubudget = new_budget, pert = 0.05)
    
    # Replace the part of the individual corresponding to the year in optimize
    if optimal_plan:
        start_index = year_to_optimize * n
        end_index = (year_to_optimize + 1) * n
        individual[start_index:end_index] = optimal_plan
    
    return individual,


import elitism
from tqdm import tqdm , trange

POPULATION_SIZE = 100
P_CROSSOVER = 0.5  # probability for crossover
P_MUTATION = 1.0   # probability for mutating an individual
MAX_GENERATIONS = 40
HALL_OF_FAME_SIZE = 10

n = len(df)
h = 10

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 3)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, n * h)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


toolbox.register("evaluate", evaluate)



# RANDOM_SEED = 10
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.1)

toolbox.register("mate", annual_plan_swap_crossover)
toolbox.register("mutate", annual_plan_lp_mutation)

def main():
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    

    
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
#     population, logbook = algorithms.eaSimple(population, toolbox,  cxpb=P_CROSSOVER, mutpb=P_MUTATION,
#                                               ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
    # print best solution found:
    best = hof.items[0]
#     print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
#     print()
#     print("-- Schedule = ")

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("best", "avg")

    # plot statistics:
    
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

    return best, hof , maxFitnessValues, meanFitnessValues

for i in range(1):
    hof_list = []
    train_list = []
    if __name__ == "__main__":
        start_time = time.time()
        sol, hof , maxFitnessValues, meanFitnessValues  =main()
        hof_list.extend(hof.items)
        train_list.append((maxFitnessValues, meanFitnessValues))
#         with open(r'C:/Users/Riiiuser/solutions/Hybrid_20t_5y.json', 'w') as f:
#             json.dump(hof_list, f)

    print("--- %s seconds ---" % (time.time() - start_time)) 
