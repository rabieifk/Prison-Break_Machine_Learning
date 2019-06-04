import numpy as np
import pandas as pd
import random
from array import *

class PrisonBuilder:

    def __init__(self, key=0):
        self.key = key

    def next_State(self, state):
        if state == 21:
            print('This state is terminal state and has no successor!')

        if state == 7:
            return [2, 6, 12]
        elif state == 8:
            return [9, 3]
        elif state == 9:
            return [4, 8]
        elif state == 13:
            return [12, 18, 14]
        elif state == 14:
            return [13, 19]
        elif state == 17:
            return [16, 18, 12]
        elif state == 18:
            return [17, 19, 13]
        elif state == 22:
            return [21, 23, 27]
        elif state == 23:
            return [22, 24, 28]
        x = state/5
        y = state % 5

        if y - 1 < 0 and x - 1 < 0:
            return [x * 5 + y + 1, (x + 1) * 5 + y]
        elif y - 1 < 0:
            return [x * 5 + y + 1, (x - 1) * 5 + y, (x + 1) * 5 + y]
        elif x - 1 < 0:
            if y % 4 == 0:
                return [x * 5 + y - 1,(x + 1) * 5 + y]
            else:
                return [x * 5 + y + 1, x * 5 + y - 1, (x + 1) * 5 + y]
        else:
            if y % 4 == 0:
                return [x * 5 + y - 1, (x - 1) * 5 + y, (x + 1) * 5 + y]
            else:
                return [x * 5 + y + 1, x * 5 + y - 1, (x - 1) * 5 + y, (x + 1) * 5 + y]


    def get_Reward(self, end):
        jailer = -1
        x = end / 5
        y = end % 5
        p = random.randint(1, 4)
        #print p
        if p == 1:
            jailer = 3
        elif p == 2:
            jailer = 4
        elif p == 3:
            jailer = 8
        elif p == 4:
            jailer = 9
        if end == jailer:
            print "You have been failed."
            return -100
        else:
            if end == 3 or end == 23:
                self.key = 1
                return 100
            elif end == 6 or end == 11 or end == 19 or end == 22 or end == 20 or end == 26:
                return -10
            elif self.key == 1 and end == 21:
                return 500
            else:
                return 0



prison = PrisonBuilder()

class TD_Lambda:
    def __init__(self, lamb=0.95, alpha=0.2, gamma=0.9):
        self.State_value = np.zeros(30)
        self.eligibility = np.zeros(30)
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb


    def getValue(self, state):
        return self.State_value[state]

    def chooseState_softmax(self, state):
        nextState = prison.next_State(state)
        #print nextState
        V_table = np.zeros(len(nextState))
        for t in range(len(nextState)):
            V_table[t] = self.State_value[nextState[t]]
            V_table[t] = np.exp(V_table[t])
        sumofQ = sum(V_table)
        for r in range(len(V_table)):
            V_table[r] = V_table[r] / sumofQ
        #print V_table
        probability = random.uniform(0, 1)
        #print probability
        i = 0
        sumation = 0
        for l in range(len(V_table)):
            sumation = sumation + V_table[l]
            #print ("Sum:", sumation)
            if l == 0:
                if probability <= V_table[l]:
                    i = 0
                    #print (l)
                    #print (i)
                    #print ("1:",i)
            elif l == len(V_table) - 1:
                if sumation - V_table[l] < probability <= 1:
                    i = len(V_table) - 1
                    #print (l)
                    #print (i)
                    #print ("2:")
            else:
                if V_table[l-1] < probability <= sumation:
                    i = l
                    #print l
                    #print ("3:")
                    #break

        next_State = prison.next_State(state)[i]
        #print next_State
        return next_State

    def learnV(self, state, nextState, reward):
        if state != 21:
            td_error = reward + self.gamma * self.State_value[nextState] - self.State_value[state]
            #G = reward + self.gamma * self.State_value[nextState]
            #self.eligibility[state] *= self.lamb * self.gamma
            self.State_value[state] = self.State_value[state] + self.alpha * td_error
            print (self.State_value)
        elif state == 22:
            td_error = reward - self.State_value[state]
            self.eligibility[state] *= self.lamb * self.gamma
            self.State_value[state] = self.State_value[state] + self.alpha * td_error




tdlambda = TD_Lambda(0.95, 0.2, 0.9)
#tdlambda.State_value = 0
tdlambda.chooseState_softmax(16)
#a = PrisonBuilder()
#b = a.next_State(18)
#m = a.get_Reward(23)
#print m

#a = get_reward(0,4, prison)
#print a

def update(current_state):
    if current_state == 21:
        return current_state
    elif current_state != 21:
        #print ("current_state:", current_state)
        nextState = tdlambda.chooseState_softmax(current_state)
        #print nextState
        if nextState != 21:
            #print ("nextState:", nextState)
            instant_reward = prison.get_Reward(nextState)
            #print ("instant_reward:", instant_reward)
            tdlambda.learnV(current_state, nextState, instant_reward)
            #print (tdlambda.State_value)
            print nextState
            return nextState
        else:
            print "The of Episode."
            return nextState

def episod_gen(state):
    print (state)
    if state != 21:
        # update(state)
        nextState = prison.next_State(state)
        print nextState
        for i in range(0, len(nextState)):
            if nextState[i] != 21:
                tdlambda.State_value[nextState[i]] = prison.get_Reward(nextState[i])
                episod_gen(nextState[i])
                print (nextState[i], tdlambda.State_value)
            else:
                tdlambda.State_value[nextState[i]] = 5000
                print (nextState[i], tdlambda.State_value)
    else:
        tdlambda.State_value[state] = 5000
        print ("The end of episode", tdlambda.State_value)
        return 0

def init():
    for i in range(30):
        tdlambda.State_value[i] = prison.get_Reward(i)
        #print i, tdlambda.State_value[i]
init()
#print tdlambda.State_value

#update(19)

training = 1
sum16811 = 0
sum12311 = 0
sum_Time = {}
path = []
for i in range(training):
    currentState = 1
    path = []
    print ("**One Episode**")
    while currentState != 21:
        nextState = update(currentState)
        path = np.append(path, [nextState])
        print (path)
        #print (currentState, nextState)
        #sum_Time[(currentState, nextState)] = sum_Time[(currentState, nextState)] + sarsa.Q[(currentState, nextState)]
        currentState = nextState
    #time = [getT(currentState, a, sum_Time) for a in mapping.next_State(currentState)]
    #max_sumT = max(time)