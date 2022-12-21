import numpy as np

class env:
    def play(inputData: list):
        
        data = np.array(inputData) #배열 원소 연산을 위해 array로 변환
        N = len(data) #input 길이
        rewards = np.full(N,-1) # 
        mean = sum(data)/N #평균 mean
        ans = mean * (2/3) #정답값 ans
        #print(ans)
        
        errorList = abs(data - ans) #각 뭔소와 정답값 과의 차 배열
        m = min(errorList) #최소 오차값
        
        for i in range(N):
            if errorList[i] == m: #최소 오차값을 가지면
                rewards[i] = 1 #해당 에이전트에 리워드는 1
                
        #print(type(rewards))
        return rewards
