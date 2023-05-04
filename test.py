import sys
import os
import time
import gymnasium as gym
import numpy as np 
import keras
import pygad.kerasga
import pygad

env = gym.make('MountainCar-v0')

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
        prediction = pygad.kerasga.predict(model=model, solution=solution, data=state.reshape(-1,state.shape[0]), verbose=0)
        action = np.argmax(prediction[0])

        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        env.render()

        reward += abs(next_state[1])*0.05

        total_reward += reward
        state = next_state

    return total_reward
    

model = keras.models.Sequential()

model.add(keras.layers.Input(shape=env.observation_space.shape))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dense(4, activation="relu"))
model.add(keras.layers.Dense(env.action_space.n, activation="linear"))

#print(testModel(model))

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=3)

bestFile = sys.argv[1]
ga_instance = pygad.load(bestFile[:-4])

print(pygad.kerasga.model_weights_as_vector(ga_instance))
