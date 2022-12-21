import numpy as np
import keras
import random
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='DQN'
        self.agentIndex = agentIndex
        self.agentNum = agentNum
        
        self.maxNum = maxNum
        
        self.dis = 0.95
        self.learning_rate = 5e-3
        self.model = self._buildModel()
        
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        
        self.last_answers = random.sample(range(self.maxNum+1), agentNum)
        self.last_action = np.random.randint(self.maxNum+1)
        
    def _buildModel(self):
        model = keras.Sequential()
        model.add(Dense(10, input_dim=self.agentNum+1, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.maxNum+1, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def train(self, answers, rewards):
        Q = self.model.predict(np.divide([np.append(self.last_answers, self.last_answers[self.agentIndex])], self.maxNum))
        Q[0, self.last_action] = rewards[self.agentIndex] + self.dis * self.model.predict(np.divide([np.append(answers, answers[self.agentIndex])], self.maxNum))[0, self.last_action]
        
        self.model.fit(np.divide([np.append(self.last_answers, self.last_answers[self.agentIndex])], self.maxNum), Q, epochs=1, verbose=0)
        
    def getAns(self, last_answers:list, last_rewards:list):
        
        self.train(last_answers, last_rewards)
        self.last_answers = last_answers
        
        if random.random() < self.epsilon:
            self.last_action = int(round(random.random() * self.maxNum))
        else:
            self.last_action = np.argmax(self.model.predict(np.divide([np.append(last_answers, last_answers[self.agentIndex])], self.maxNum)))
        
        if self.last_action < 0:
            self.last_action = 0
        elif self.last_action > self.maxNum:
            self.last_action = self.maxNum

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return self.last_action
