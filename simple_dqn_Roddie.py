#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from numpy import unravel_index
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Dense, Flatten, Conv1D, MaxPooling1D ,InputLayer
from tensorflow import dtypes


# In[4]:


class ReplayBuffer():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr =0    ## memory counter
        self.state_memory = np.zeros(shape= (self.mem_size,input_dims), dtype= np.float32)
        self.new_state_memory = np.zeros(shape=(self.mem_size , input_dims), dtype = np.float32)
        self.action_memory1 = np.zeros(self.mem_size, dtype = np.int32)
        #self.action_memory2 = np.zeros(self.mem_size, dtype =np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype =np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype =np.int32)
        
    def store_transition(self, state, action1, reward, state_, done):  #new state
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory1[index] = action1
        #self.action_memory2[index] = action2
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1- done
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace =False).reshape(batch_size, -1)
        
        states = self.state_memory[batch].reshape(64,15,1)
        states_ = self.new_state_memory[batch].reshape(64,15,1)
        rewards = self.reward_memory[batch]
        actions1 = self.action_memory1[batch]
        #actions2 = self.action_memory2[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions1, rewards, states_,terminal
    
    
    def build_dqn(lr, n_actions, input_dims):
        model =models.Sequential()
        model.add(Conv1D(64,8, input_shape = (15,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size = 4))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(324, activation = 'relu'))
        model.compile(optimizer = 'rmsprop',loss= 'mean_squared_error')
        
        return model
    


# In[6]:


class Agent():
    def __init__(self,lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec= 1e-3, epsilon_end =0.001,
                mem_size = 10000, fname = 'dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size,input_dims)
        self.q_eval = self.memory.build_dqn(lr, n_actions)
        
    def store_transition(self, state, action1, reward, new_state, done):
        self.memory.store_transition(state, action1, reward, new_state, done)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action1 = np.random.choice(self.action_space)
            
            
        else : 
            state = np.array([observation]).reshape(1,15,1)
            actions = self.q_eval.predict(state)
            print(state)
            action1 = np.argmax(actions)
           

        
        return action1
    
    def learn(self) :
        if self.memory.mem_cntr < self.batch_size :
            return
        states, actions1, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        
        q_target[batch_index, actions1] = rewards + self.gamma * np.max(q_next, axis=1)*dones
        
        
        
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon =self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
       
    def save_model(self) :
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)
        


# In[ ]:




