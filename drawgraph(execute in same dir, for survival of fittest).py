import csv
import numpy as np
import matplotlib.pyplot as plt

lineNum = 20
fileNum = 20

agentNum = 7
agent = [[] for i in range(agentNum)]
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
for i in range(fileNum):
    f = open('population{}.csv'.format(str(i)), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    lineN = 0
    for line in rdr:
        agent[agentDict[line[0]]].append(list(map(int, line[1:])))
        cnt += 1
        print("Loading data {}%".format(str(cnt*100/(lineNum*fileNum))))

agent = np.array(agent)
avg = [[] for i in range(agentNum)]
std = [[] for i in range(agentNum)]

for i in range(agentNum):
    print(np.shape(agent[i]))

for agentN in range(agentNum):
    arr = np.array(agent[agentN])
    print(np.shape(arr))
    for i in range(len(arr[0])):
        avg[agentN].append(np.mean(arr[:,i]))
        std[agentN].append(np.std(arr[:,i]))

for agentN in range(agentNum):
    plt.fill_between(range(len(arr[0])), np.array(avg[agentN])-np.array(std[agentN]), np.array(avg[agentN])+np.array(std[agentN]),alpha=0.3, color = colorList[agentN])
for agentN in range(agentNum):
    plt.plot(avg[agentN], label = agentName[agentN], color = colorList[agentN])
    
axes = plt.gca()
axes.set_ylim([-0.025,20])
plt.legend()
plt.show()
