import sys
import os
import gymnasium as gym
import numpy as np 
import keras
import pygad.kerasga
import pygad

env = gym.make('MountainCar-v0')

def get_best_model_filename(folder):
    filenames = next(os.walk(folder), (None, None, []))[2]
    if filenames:
        best = filenames[0]
        for file in filenames:
            score = float(file[:-4])
            if score > float(best[:-4]):
                best = file
        return best
    else:
        return None

def getAction(model, state):
    state = state.reshape(-1, state.shape[0])
    prediction = model.predict(state)
    action = np.argmax(prediction[0])
    return action

def testModel(gaInstance, solution, sol_idx):
    # Reset environment, getting initial state
    state, info = env.reset()

    prediction = pygad.kerasga.predict(model=model, solution=solution, data=state.reshape(-1,state.shape[0]))
    action = np.argmax(prediction[0])
    
    total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table temporal difference learning algorithm
    while (not epoch_done) and (not epoch_truncated):
        prediction = pygad.kerasga.predict(model=model, solution=solution, data=state.reshape(-1,state.shape[0]))
        action = np.argmax(prediction[0])

        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        env.render()

        reward += abs(next_state[1])*0.05

        total_reward += reward
        state = next_state

    return total_reward
    
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    ga_instance.save("saves/"+str(ga_instance.best_solution()[1]))
    sys.exit(0)


model = keras.models.Sequential()

model.add(keras.layers.Input(shape=env.observation_space.shape))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dense(4, activation="relu"))
model.add(keras.layers.Dense(env.action_space.n, activation="linear"))

#print(testModel(model))

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=3)

num_generations = 250 # Number of generations.
num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights

bestFile = get_best_model_filename("saves/")
if bestFile:
    ga_instance = pygad.load("saves/"+bestFile[:-4])
else:
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=testModel,
                           on_generation=callback_generation,
                           parallel_processing = None)

ga_instance.run()


filename = 'solution'
ga_instance.save(filename=filename)
env.close()
