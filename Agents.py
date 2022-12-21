import random as rd
import numpy as np

class Agent:
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Parent'
    def getAns(self, Answers:list, Rewards:list):
        return 0

class randomAgent(Agent):
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Random'
        self.maxNum = maxNum
    def getAns(self, Answers:list, Rewards:list):
        return rd.randint(0, self.maxNum)

class zeroAgent(Agent):
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Zero'
    def getAns(self, Answers:list, Rewards:list):
        return 0

class FixedAgent(Agent):
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Fixed'
        self.num = rd.randrange(0, maxNum+1)
    def getAns(self, Answers:list, Rewards:list):
        return self.num

class RepeatAgent(Agent):
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Repeat'
        self.num = 2 #+ rd.randint(0, 99)%2
        self.arr = rd.sample(range(maxNum+1), self.num)
        self.cnt=0
    def getAns(self, Answers:list, Rewards:list):
        self.cnt+=1
        return self.arr[self.cnt%self.num]

class noisedFollowAgent(Agent):
    def __init__(self, maxNum, agentNum, agentIndex):
        self.name='Follow'
        self.maxNum = maxNum
        self.StdDev = rd.randint(1,10)
    def getAns(self, Answers:list, Rewards:list):
        ans = Answers[np.argmax(Rewards)]
        ans = int(round(np.random.normal(loc=Answers[np.argmax(Rewards)],
                               scale = self.StdDev)))
        if (ans>self.maxNum):
            return self.maxNum
        elif ans<0:
            return 0
        else:
            return ans
