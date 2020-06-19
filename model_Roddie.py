#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import unravel_index
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Dense, Flatten, Conv1D, MaxPooling1D
import pandas as pd

class DACON():
    def __init__(self):
        self.sample_submission = pd.read_csv('data/sample_submission.csv')
        self.max_count = pd.read_csv('data/max_count.csv')
        self.stock = pd.read_csv('data/stock.csv')
        self.order = pd.read_csv('data/order.csv')
        self.change_time = pd.read_csv('data/change_time.csv')
        self.production = np.array([0,0,0,0])
        self.checker2 = np.array([0,0,0,0]) # 성형공정변경시간 합계 , 변경 이벤트, 멈춤시간합계, 멈춤이벤트
        f = function_set()
        
        
        
        #self.action_dict = {0:'PROCESS', 1:'STOP',2:'CHECK1', 3 : 'CHECK2', 4 : 'CHECK3' , 5 : 'CHECK4', 6 : 'CHANGE12', 7 : 'CHANGE13'
         #     ,8 : 'CHANGE14' , 9: 'CHANGE21', 10 : 'CHANGE23', 11 : 'CHANGE24', 12:'CHANGE31', 13: 'CHANGE32',14:'CHANGE34',
          #    15 : 'CHANGE41', 16: 'CHANGE42', 17:'CHANGE43'}
        
        self.state = [0, self.max_count.loc[0,'count'], self.max_count.loc[0,'count'],  # t, A라인 max_count, B라인 max_count
                      self.order.loc[0,'BLK_1'], self.order.loc[0,'BLK_2'], self.order.loc[0,'BLK_3'], self.order.loc[0,'BLK_4'],
                      # 수요 BLK_1, 2, 3, 4
                     0, 0, 0, 
                      0, 0 ,0,
                      0,0] # process_time_checker, check_time_checker, change_time_checker, 
                        # process_time checker2 , check_time_checker2 change_time_checker2 state1 , 2 총 15개 
                    #state 에서 0이면 chagne, check, stop이고 process 면 1 ,2, 3,4
        self.action1 = 'CHECK'  # A 라인 액션
        self.action2 = 'CHECK' # B 라인 액션

    def reset(self):
        state = np.array([0, int(self.max_count.loc[0,'count']), int(self.max_count.loc[0,'count']),  # t, A라인 max_count, B라인 max_count
                      int(self.order.loc[0,'BLK_1']), int(self.order.loc[0,'BLK_2']), int(self.order.loc[0,'BLK_3']), int(self.order.loc[0,'BLK_4']),
                     0, 0, 0, 
                      0, 0 ,0,
                      0,0])
        return state
    
            
        
    def step(self, action1, action2, observation): 
        f = function_set()
        state = observation
        state[0] = state[0] +1 ## 시간을 업데이트
        hour = state[0]%24 ## 하루 중 몇 시간 째야?? 0 - 23
        if hour < (state[0] -1) % 24 : # 날이 바뀌었으면
            state[1] = self.max_count.loc[state[0]//24, 'count']
            state[2] = self.max_count.loc[state[0]//24, 'count'] ## A,B 라인의 max_count 새로운 날로 초기화
        else : 
            state[1] = state[1] - f.isprocess(action1) * 5.5
            state[2] = state[2] - f.isprocess(action2) * 5.5
        
        state[3] = state[3] - (506 * f.isprocess(action1) * (state[13] == 1) + 506 * f.isprocess(action2) * (state[14] == 1))
        state[4] = state[4] - (506 * f.isprocess(action1) * (state[13] == 2) + 506 * f.isprocess(action2) * (state[14] == 2))
        state[5] = state[5] - (400 * f.isprocess(action1) * (state[13] == 3) + 400 * f.isprocess(action2) * (state[14] == 3))
        state[6] = state[6] - (400 * f.isprocess(action1) * (state[13] == 4) + 400 * f.isprocess(action2) * (state[14] == 4))
        
        if hour == 18 :
            if state[0] // 24 < 90:
                state[3] = state[3] + self.order.loc[state[0]//24+ 1,'BLK_1']
                state[4] = state[4] + self.order.loc[state[0]//24+ 1,'BLK_2']
                state[5] = state[5] + self.order.loc[state[0]//24+ 1,'BLK_3']
                state[6] = state[6] + self.order.loc[state[0]//24+ 1,'BLK_4']

            
        state[7] = state[7] + f.isprocess(action1) 
        state[8] = state[8] + f.ischeck(action1)
        state[9] = state[9] + f.ischange(action1)
        
        state[10] = state[10] + f.isprocess(action2) 
        state[11] = state[11] + f.ischeck(action2)
        state[12] = state[12] + f.ischange(action2)
        
        if state[8] == 28:
            state[13] = int(action1[6])
            state[8] = 0
            
        if state[11] == 28:
            state[14] = int(action2[6])
            state[11] = 0
            
        if f.ischange(action1) ==1:
            if state[9] == int(self.change_time[(self.change_time['from'] == 'MOL_'+ action1[7]) & (self.change_time['to'] == 'MOL_' + action1[8])]['time']):
                state[13] = int(action1[8])
                state[9] = 0
                
        if f.ischange(action2) ==1:
            if state[12] == int(self.change_time[(self.change_time['from'] == 'MOL_'+ action2[7]) & (self.change_time['to'] == 'MOL_' +action2[8])]['time']):
                state[14] = int(action2[8])
                state[12] = 0
        
        if state[7] == 140 : 
            state[13] = 0
            state[7] = 0
            
        if state[10] == 140 :
            state[14] = 0
            state[10] = 0
        
        
        next_state = state
        if state[0] == 2183 :
            check_done = True
        else :
            check_done = False
        
        p = max(self.order.loc[(state[0] // 24):, 'BLK_1'].sum() - self.production[0], 0)+ max(self.order.loc[min((state[0] // 24),90):, 'BLK_2'].sum() - self.production[1], 0)+ max(self.order.loc[min((state[0] // 24),90):, 'BLK_3'].sum() - self.production[2], 0)+  max(self.order.loc[min((state[0] // 24),90):, 'BLK_4'].sum() - self.production[3], 0)
         # 역누적 수요 -역누적 생산량 = 부족분
            
        q = max(self.production[0] - self.order.loc[min((self.state[0] // 24),90):, 'BLK_1'].sum(), 0)+         max(self.production[1] - self.order.loc[min((self.state[0] // 24),90):, 'BLK_2'].sum(), 0)+         max(self.production[2] - self.order.loc[min((self.state[0] // 24),90):, 'BLK_3'].sum(), 0)+         max(self.production[3] - self.order.loc[min((self.state[0] // 24),90):, 'BLK_4'].sum(), 0) 

        self.checker2[0] = self.checker2[0] + f.ischange(action1) + f.ischange(action2)
        c = self.checker2[0]
        self.checker2[2] = self.checker2[2] + f.isstop(action1) + f.isstop(action2)
        s = self.checker2[2]
        
        N = self.order.loc[(state[0]//24):, ['BLK_1','BLK_2','BLK_3','BLK_4']].sum().sum()
        M = state[0]
        
        reward = 50 * f.F(p, 10*N) + 20 * f.F(q, 10*N) + 20 * f.F(c, M)  #+ 10 * f.F(s, M) 
        

        return np.array(next_state) ,reward, check_done
    
class function_set():
    def isprocess(self,action) :
        if action == 'PROCESS':
            return 1
        else :
            return 0
        
    def ischange(self,action) : 
        if action in ['CHANGE_12', 'CHANGE_13', 'CHANGE_14' ,'CHANGE_21','CHANGE_23','CHANGE_24','CHANGE_31','CHANGE_32','CHANGE_34','CHANGE_41','CHANGE_42','CHANGE_43']:
            return 1
        else: return 0
    
    def ischeck(self,action) :
        if action in ['CHECK_1','CHECK_2','CHECK_3','CHECK_4'] :
            return 1
        else : return 0
    
    def isstop(self, action) :
        if action == 'STOP':
            return 1
        else : return 0
                      
    def F(self, x, a):
        if x < a:
            return 1 - x/a
        else :
            return 0


# In[ ]:




