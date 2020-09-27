import random
from deap import base, creator, tools, benchmarks

# Create the toolbox with the right parameters
def create_toolbox(num_bits):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize the toolbox
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=num_bits)

    # Define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation operator & evaluate with the rastrigin function
    toolbox.register("evaluate", benchmarks.rastrigin)

    # Register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # Register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # Operator for selecting individuals for breeding
    toolbox.register("select", tools.selRoulette)

    return toolbox

if __name__ == "__main__":
    # Define the number of bits
    num_bits = 2

    # Create a toolbox using the above parameter
    toolbox = create_toolbox(num_bits)

    # Seed the random number generator
    random.seed()

    # Create an initial population of 20 individuals
    population = toolbox.population(n=20)

    # Define probabilities of crossing and mutating
    probab_crossing, probab_mutating  = 0.75, 0.1

    # Define the number of generations
    num_generations = 100

    print('\nStarting the evolution process')

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    # fitnesses = (toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print('\nEvaluated', len(population), 'individuals')

    # Iterate through generations

    for g in range(num_generations):
        print("\n===== Generation", g)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cross two individuals
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

                # "Forget" the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            # Mutate an individual
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('Evaluated', len(invalid_ind), 'individuals')

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2), ', Standard deviation =',
                round(std, 2))


    print("\n==== End of evolution")

    # Determine best individual and its place in the population
    best_ind = tools.selBest(population, -1)[0]
    best_n_ind = 0
    for y in range(len(population)):
        if(population[y] == best_ind):
            best_n_ind = y+1

    print('\nBest individual is',best_n_ind,'\n',best_ind)
    print('\nThe fitness of the best individual is', min(fits))
    # print('\nNumber of ones:', sum(best_ind))
