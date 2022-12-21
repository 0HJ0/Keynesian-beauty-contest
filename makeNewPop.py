import csv
import numpy as np
import matplotlib.pyplot as plt

lineNum = 20
fileNum = 10
selectionRate = 25
agentNum = 7
agent = [[] for i in range(lineNum)]
agentName = ["Random", "Fixed", "Repeat", "Follow", "NN", "DQN", "LSTM"]

agentDict = {'Random':0,
             'Fixed':1, 
             'Repeat':2,
             'Follow':3,
             'NN':4,
             'DQN':5,
             'LSTM':6}

colorList = ['dodgerblue',
             'orange', 
             'limegreen',
             'purple',
             'brown',
             'violet',
             'gray']


cnt = 0
GroupList = []
newPop = []

for i in range(fileNum):
    f = open('rewards{}_10.csv'.format(str(i)), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    lineN = 0
    member = []
    agent = [[] for j in range(lineNum)]
    cnt = 0
    for line in rdr:
        member.append(line[0])
        agent[cnt].append(list(map(int, line[1:])))
        cnt += 1
        print("Loading data {}%".format(str(cnt*100/(lineNum))))

    win = []
    for j in range(lineNum):
        #print(agent[j][0][-1000:])
        win.append(sum(agent[j][0][-1000:]))

    idxList = [j for j in range(lineNum)]

    for k in range(lineNum):
        for j in range(lineNum-1):
            if win[j]>win[j+1]:
                win[j], win[j+1] = win[j+1], win[j]
                idxList[j], idxList[j+1] = idxList[j+1], idxList[j]

    newPopulation = [0]*7
    for j in range(lineNum):
        newPopulation[agentDict[member[j]]]+=1

    GroupList.append([])
    for j in range(agentNum):
        GroupList[-1].append(newPopulation[j])
    
    for j in range(int(selectionRate/100*lineNum)): #하위 25%제거.
        newPopulation[agentDict[member[idxList[j]]]]-=1

    for j in range(int(selectionRate/100*lineNum)): #상위 25%복제.
        newPopulation[agentDict[member[idxList[-1-j]]]]+=1

    
    newPop.append(newPopulation)


print(GroupList)
print(newPop)
