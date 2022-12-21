import csv
import numpy as np
import matplotlib.pyplot as plt

lineNum = 30
fileNum = 3

agentNum = 8
agent = [[] for i in range(agentNum)]
agentName = ["Random", "Fixed", "Repeat", "Follow", "NFollow", "NN", "DQN", "LSTM"]

agentDict = {'Random':0,
             'Fixed':1, 
             'Repeat':2,
             'Follow':3,
             'NFollow':4,
             'NN':5,
             'DQN':6,
             'LSTM':7}

colorList = ['dodgerblue',
             'orange', 
             'limegreen',
             'red',
             'purple',
             'brown',
             'violet',
             'gray']

cnt = 0
for i in range(fileNum):
    f = open('rewards{}.csv'.format(str(i)), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    lineN = 0
    for line in rdr:
        agent[agentDict[line[0]]].append(list(map(int, line[1:])))
        cnt += 1
        print("Loading data {}%".format(str(cnt*100/(lineNum*fileNum))))

for agentN in range(agentNum):
    for lineN in range(len(agent[agentN])):
        for i in range(len(agent[agentN][lineN])):
            if i!=0:
                agent[agentN][lineN][i] = ((agent[agentN][lineN][i-1]*i + agent[agentN][lineN][i]) / (i+1))

agent = np.array(agent)
avg = [[] for i in range(agentNum)]
std = [[] for i in range(agentNum)]

for i in range(agentNum):
    print(np.shape(agent[i]))

for agentN in range(agentNum):
    arr = np.array(agent[agentN])
    print(np.shape(arr))
    for i in range(10000):
        avg[agentN].append(np.mean(arr[:,i]))
        std[agentN].append(np.std(arr[:,i]))

for agentN in range(agentNum):
    plt.fill_between(range(10000), np.array(avg[agentN])-np.array(std[agentN])/2, np.array(avg[agentN])+np.array(std[agentN])/2,alpha=0.3)
for agentN in range(agentNum):
    plt.plot(avg[agentN], label = agentName[agentN], color = colorList[agentN])
    
axes = plt.gca()
axes.set_ylim([-0.025,0.5])
plt.legend()
plt.show()
