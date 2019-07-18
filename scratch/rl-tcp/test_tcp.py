#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

# Modification start
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#Modification End

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10 # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
#env = ns3env.Ns3Env()
#env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            # time-based = 1
            tcpAgent = TcpTimeBased()
            print("time-base")
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent

# initialize variable
# get_agent.tcpAgents = {}
# get_agent.ob_space = ob_space
# get_agent.ac_space = ac_space
 
# try:
#    while True:
#        print("Start iteration: ", currIt)
#        obs = env.reset()
#        reward = 0
#        done = False
#        info = None
#        print("Step: ", stepIdx)
#        print("---obs: ", obs)

        # get existing agent of create new TCP agent if needed
#        tcpAgent = get_agent(obs)

#        while True:
#            stepIdx += 1
#            action = tcpAgent.get_action(obs, reward, done, info)
#            print("---action: ", action)

#            print("Step: ", stepIdx)
#            obs, reward, done, info = env.step(action)
#            print("---obs, reward, done, info: ", obs, reward, done, info)

            # get existing agent of create new TCP agent if needed
#            tcpAgent = get_agent(obs)

#            if done:
#                stepIdx = 0
#                if currIt + 1 < iterationNum:
#                    env.reset()
#                break

#        currIt += 1
#        if currIt == iterationNum:
#            break

# Modification Start

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape = (observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
       # if np.random.rand() < self.exploration_rate:
        #    return random.randrange(self.action_space)
        print(state.shape)
        q_values = self.model.predict(state)
        print("Q_Value: ",q_values)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

# try:
#    while True:
#        print("Start iteration: ", currIt)
#        obs = env.reset()
#        reward = 0
#        done = False
#        info = None
#        print("Step: ", stepIdx)
#        print("---obs: ", obs)

        # get existing agent of create new TCP agent if needed
#        tcpAgent = get_agent(obs)

#        while True:
#            stepIdx += 1
#            action = tcpAgent.get_action(obs, reward, done, info)
#            print("---action: ", action)

#            print("Step: ", stepIdx)
#            obs, reward, done, info = env.step(action)
#            print("---obs, reward, done, info: ", obs, reward, done, info)

            # get existing agent of create new TCP agent if needed
#            tcpAgent = get_agent(obs)

#            if done:
#                stepIdx = 0
#                if currIt + 1 < iterationNum:
#                    env.reset()
#                break

#        currIt += 1
#        if currIt == iterationNum:
#            break

# Modification Start
obs = env.reset()
print(obs)
#observation_space = [obs[11], obs[15]]
#observation_space = env.observation_space.shape[0]
print("size of obs space: ",env.observation_space.shape[0])
action_space = env.action_space.shape[0]

#print("Observation_space : ",observation_space)

dqn_solver = DQNSolver(2, action_space)
#state = observation_space
reward = 0
done = False
info = None
try:
    while True:
        print("Start iteration: ", currIt)
        print("Step: ", stepIdx)
        #print("---obs: ", observation_space)
        obs = env.reset()
        print("obs: ",obs)
        observation_space = [obs[11], obs[15]]
        
        state = np.reshape(observation_space, [1,2])
        print(state)
        print(state.shape)
        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs)

        while True:
            stepIdx += 1
            # action = tcpAgent.get_action(obs, reward, done, info)
            action = dqn_solver.act(state)
            exec_action = [100, 100 * action]
            print("---action: ", action)
            print("Step: ", stepIdx)
            #state_next, reward, terminal, info = env.step(action)
            obs, reward, done, info = env.step(exec_action)
            print("---obs, reward, done, info: ", obs, reward, done, info)
            reward = reward if not done else -reward
            observation_space = np.reshape([obs[11], obs[15]],[1,2])
            state_next = observation_space;
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next
            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break
            dqn_solver.experience_replay()
        currIt += 1
        if currIt == iterationNum:
            break

# Modification End


except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")
