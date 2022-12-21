import numpy as np
import random
from keras.layers import LSTM, Dense
from keras import Sequential
from keras.optimizers import Adam
from collections import deque

import matplotlib.pyplot as plt

names = ['random', 'fixed', 'repeat', 'follow', 'Nfollow', 'NN', 'DQN', 'LSTM']

class LSTMAgent:
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name = 'LSTM'
        
        self.maxNum = maxNum
        self.agentNum = agentNum
        self.agentIndex = agentIndex 
        
        self.LSTMlearningRate = 1e-2
        self.NNlearningRate = 5e-3
        
        self.sequenceSize = 8

        self.epsilon = 1
        self.epsilonDecay = 0.999
        self.epsilonMin = 0.001
        
        self.LSTMmodel = self._buildModel()
        self.NNmodel = self._buildOutputModel()
        
        self.memory = deque([], self.sequenceSize)
        self.RHIS = []
        
    def _buildModel(self):
        model = Sequential()
        model.add(LSTM(10, input_shape=(self.sequenceSize, self.agentNum+1), return_sequences=True, 
                       kernel_initializer='he_uniform'))
        model.add(LSTM(10, kernel_initializer='he_uniform'))
        model.add(Dense(self.agentNum-1, activation='linear', 
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.LSTMlearningRate))
        return model

    def _buildOutputModel(self):
        model = Sequential()
        model.add(Dense(10, input_dim=2*self.agentNum, activation='relu', 
                        kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='relu', 
                        kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.NNlearningRate))
        return model
    
    def predict(self):
        Pred = np.round(self.maxNum * self.LSTMmodel.predict(
            np.divide([np.append(self.memory, np.vstack(np.array(self.memory)[:,self.agentIndex]), axis=1)], self.maxNum)))
        
        arr = np.concatenate((Pred, self.memory[-1], self.memory[-1][self.agentIndex]), axis=None)
        X = np.divide([arr], self.maxNum)
        action = self.NNmodel.predict(X)
        return int(round(self.maxNum * action[0,0]))
        
    def trainLSTM(self, Answers):
        X = np.divide([np.append(self.memory, np.vstack(np.array(self.memory)[:,self.agentIndex]), axis=1)], self.maxNum)
        Y = np.divide([np.delete(Answers, self.agentIndex)], self.maxNum)
        self.LSTMmodel.fit(X, Y, epochs=1, verbose=0)
    
    def trainNN(self, Answers, Rewards):
        Pred = np.round(self.maxNum * self.LSTMmodel.predict(
            np.divide([np.append(self.memory, np.vstack(np.array(self.memory)[:,self.agentIndex]), axis=1)], self.maxNum)))
        
        self.RHIS.append(np.subtract(Pred, np.delete(Answers, self.agentIndex)))
        
        arr = np.concatenate((Pred, self.memory[-1], self.memory[-1][self.agentIndex]), axis=None)
        X = np.divide([arr], self.maxNum)
        ans = Answers[np.argmax(Rewards)]
        Y = np.divide([[ans]], self.maxNum)
        self.NNmodel.fit(X,Y, epochs=1, verbose=0)

    def data(self, ContestNum):
        
        plt.figure(figsize=(15,7.5))
        for i in range(self.agentNum-1):
            tempList = []
            for j in range(ContestNum):
                tempList.append(np.mean(np.absolute(self.RHIS)[j-100:j+1, 0, i]))
            plt.plot(range(ContestNum), tempList, label=names[i])
        plt.legend()
        plt.savefig('graph.png')
            
        
    def getAns(self, Answers, Rewards):
        
        if len(self.memory) == self.sequenceSize:
            self.trainLSTM(Answers)
            self.trainNN(Answers, Rewards)
        
        self.memory.append(Answers)
        
        if random.random() < self.epsilon or len(self.memory) != self.sequenceSize:
            action = random.choice(range(self.maxNum+1))
        else:
            action = self.predict()

        if self.epsilon > self.epsilonMin:
            self.epsilon*=self.epsilonDecay

        """ loss graph
        ContestNum = 5000
        if len(self.RHIS) == ContestNum:
            self.data(ContestNum)
        """
        
        return action
