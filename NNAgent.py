import numpy as np
import keras
import random
from keras.layers import Dense
from keras.optimizers import Adam

class NNAgent:
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='NN'
        self.agentNum = agentNum
        self.agentIndex = agentIndex
        
        self.maxNum = maxNum
        
        self.learning_rate = 5e-3
        self.model = self._buildModel()
        
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        
        self.last_answers = random.sample(range(maxNum+1), agentNum)
        
        
    def _buildModel(self):
        model = keras.Sequential()
        model.add(Dense(10, input_dim=self.agentNum+1, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
        
    def getAns(self, last_answers:list, last_rewards:list):
        
        X = np.divide([np.append(self.last_answers, self.last_answers[self.agentIndex])], self.maxNum)
        Y = np.divide([last_answers[np.argmax(last_rewards)]], self.maxNum)
        self.model.fit(X, Y, epochs=1, verbose=0)
                    
        self.last_answers = last_answers
        
        if random.random() < self.epsilon:
            action = random.choice(range(self.maxNum+1))
        else:
            action = int(round(self.maxNum * self.model.predict(np.divide([np.append(last_answers, last_answers[self.agentIndex])], self.maxNum))[0,0]))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if action<0:
            return 0
        elif action>self.maxNum:
            return self.maxNum
        else:
            return action
