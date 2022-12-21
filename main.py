import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

import env
import Agents
import NNAgent
import DQNAgent
import LSTMAgent

maxNum = 100
ContestNum = 5000 #한 세대의 대회 회수
agentNum = 20
selectionRate = 25
genNum = 10
minimum = 2
maximum = 4
stdList = []

agentDict = {0: Agents.randomAgent,
             1: Agents.FixedAgent,
             2: Agents.RepeatAgent,
             3: Agents.noisedFollowAgent,
             4: NNAgent.NNAgent,
             5: DQNAgent.DQNAgent,
             6: LSTMAgent.LSTMAgent}

colorDict = {'Random':'dodgerblue',
             'Fixed':'orange', 
             'Repeat':'limegreen',
             'Follow':'purple',
             'NN':'brown',
             'DQN':'violet',
             'LSTM': 'gray'}

numDict = {  'Random':0,
             'Fixed':1, 
             'Repeat':2,
             'Follow':3,
             'NN':4,
             'DQN':5,
             'LSTM':6}

def makeInitialGroup():
    fn=random.randint(2,4)
    numList = [0]*6
    while True:
        numList = [0]*6
        while sum(numList)<(agentNum-fn):
            numList[random.randint(0,5)]+=1
        if max(numList)<=maximum and min(numList)>=minimum:
            break
    numList.insert(3,fn)
    #numList = [1, 1, 1, 1, 1, 1, 1, 1]
    print("agentList: {}".format(numList))
    return numList

def makeGroup(numList:list):
    Group=[]
    cnt=0
    for i in range(len(numList)):
        for j in range(numList[i]):
            Group.append(agentDict[i](maxNum, sum(numList), cnt))
            cnt+=1
        print("Successfully made Group of Agents. [{}/{}]".format(i+1,len(numList)))
    return Group


def main(filename, mainNum, genNum):
    GroupList = [makeInitialGroup()]
    #GroupList = [[0,0,0,0,0,0,20]]
    '''
    stdList.append([0])
    for i in range(1, GroupList[0][3]):
        stdList[-1].append(random.randint(0, 10))
    '''
    for i in range(1, genNum+1):
        GroupList.append(expGen(filename, mainNum, i, GroupList[-1]))
    names = ['Random', 'Fixed', 'Repeat', 'Follow', 'NN', 'DQN', 'LSTM']
    data = [[] for i in range(7)]
    print(GroupList)
    for i in range(7):
        for j in range(len(GroupList)):
            data[i].append(GroupList[j][i])
    dataframe = pd.DataFrame(data, index=names)
    dataframe.to_csv("data/{}/population{}.csv".format(filename, str(mainNum)), header=False, index=True)
    '''
    dataframe = pd.DataFrame(stdList)
    dataframe.to_csv("data/{}/std{}.csv".format(filename, str(mainNum)), header=False)
    '''
    print('Done.')

def expGen(filename, mainNum, genNum, Group):    
    # initial setting
    Agent = makeGroup(Group)
    '''
    print(Group[3])
    print(stdList[-1])
    for i in range(Group[3]):
        Agent[sum(Group[:3])+i].StdDev = stdList[-1][i]
    '''
    # Contest
    lastRewards = np.zeros(agentNum)
    lastAnswers = random.sample(range(maxNum+1), agentNum)
    winHistory=[ [0] for i in range(agentNum) ]
    ansHistory=[ [0] for i in range(agentNum) ]
    
    for Contest in range(ContestNum):
        startTime = time.time()
        
        rewards=[]
        answers = []
        
        for i in range(agentNum):
            answers.append(int(round(Agent[i].getAns(lastAnswers, lastRewards))))
        rewards=env.env.play(answers)

        for i in range(agentNum):
            winHistory[i].append(1 if rewards[i]==1 else 0)
            ansHistory[i].append(answers[i])
        
        lastRewards=rewards
        lastAnswers=answers
        
        endTime = time.time()
        print("[Episode:{:>5}][time:{:<5}]".format(Contest,round(endTime - startTime, 3)))
        if Contest % 100 == 0:
            for i in range(agentNum):
                print("[Agent {:>2} {:>6}:{:>3}]".format(i,Agent[i].name,answers[i]))

    #Saving data
    names = []
    for agt in Agent:
        names.append(agt.name)
    dataframe = pd.DataFrame(winHistory, index=names)
    dataframe.to_csv("data/{}/rewards{}_{}.csv".format(filename, str(mainNum), str(genNum)), header=False, index=True)
    dataframe = pd.DataFrame(ansHistory, index=names)
    dataframe.to_csv("data/{}/answers{}_{}.csv".format(filename, str(mainNum), str(genNum)), header=False, index=True)

    # Learning Ability Graph
    plt.figure(figsize=(15,7.5))
    plt.subplot(211)
    for i in range(agentNum):
        tempList = []
        for j in range(ContestNum):
            tempList.append(sum(winHistory[i][:j]))
        if i==0 or Agent[i].name!=Agent[i-1].name:
            plt.plot(range(ContestNum), np.divide(tempList, range(1,ContestNum+1)), label=Agent[i].name, color=colorDict[Agent[i].name])
        else:
            plt.plot(range(ContestNum), np.divide(tempList, range(1,ContestNum+1)), color=colorDict[Agent[i].name])
    plt.legend()

    plt.subplot(212)
    for i in range(agentNum):
        tempList = []
        for j in range(ContestNum):
            tempList.append(np.mean(winHistory[i][j-100:j+1]))
        plt.plot(range(ContestNum), tempList, label=Agent[i].name)
    plt.savefig('data/{}/graph{}_{}.png'.format(filename, str(mainNum), str(genNum)))

    #Survival of Fittest Task
    tempList=[]
    idxList = [j for j in range(agentNum)]
    for j in range(agentNum):
                tempList.append(sum(winHistory[j][-1000:]))

    for k in range(agentNum):
        for j in range(agentNum-1):
            if tempList[j]>tempList[j+1]:
                tempList[j], tempList[j+1] = tempList[j+1], tempList[j]
                idxList[j], idxList[j+1] = idxList[j+1], idxList[j]

    newPopulation = []
    for j in range(7):
        newPopulation.append(Group[j])
    '''
    stdList.append([])
    for j in range(Group[3]):
        stdList[-1].append(Agent[sum(Group[:3])+j].StdDev)
    '''

    print(newPopulation)
    for j in range(int(selectionRate/100*agentNum)): #하위 25%제거.
        newPopulation[numDict[Agent[idxList[j]].name]]-=1
        #if Agent[idxList[j]].name == "Follow":
            #stdList[-1].remove(Agent[idxList[j]].StdDev)
             
    for j in range(int(selectionRate/100*agentNum)): #상위 25%복제.
        newPopulation[numDict[Agent[idxList[-1-j]].name]]+=1
        #if Agent[idxList[-1-j]].name == "Follow":
            #stdList[-1].append(Agent[idxList[-1-j]].StdDev)

    return newPopulation

if __name__=='__main__':
    filename = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    if not(os.path.isdir("data/{}".format(filename))):
        os.makedirs(os.path.join("data/{}".format(filename)))
    for i in range(9, 10):
        main(filename, i, genNum)

