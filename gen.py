import pygad
import numpy

inputs = [0.4,1,0,7,8]
desired_output = 32

def fitness_func(instance, solution, solution_idx):
    output = numpy.sum(solution*inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    
    return fitness

ga_instance = pygad.GA(num_generations=100,
                       sol_per_pop=10,
                       num_genes=len(inputs),
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       mutation_type="random",
                       mutation_probability=0.6)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("\nLa mejor solución encontrada es:")
print("Genes (pesos):", solution)
print("Aptitud de la solución:", solution_fitness)

# Calcular el resultado final usando la mejor solución
final_output = numpy.sum(solution * inputs)
print("Resultado obtenido con la mejor solución:", final_output)
print("Diferencia con la salida deseada:", abs(final_output - desired_output))

ga_instance.plot_fitness()