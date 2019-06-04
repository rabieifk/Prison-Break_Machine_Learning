import numpy as np
import random


class PrisonBuilder:

    def __init__(self, key=0, jailer=-1):
        self.key = key
        self.jailer = jailer

    def next_State(self, state):
        # if state == 21 and self.key == 1:
        #   print('This state is terminal state and has no successor!')
        if state == 21 and self.key == 0:
            return [22, 20, 16, 26]

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
        elif state == 22 and self.key == 0:
            return [23, 27]
        elif state == 22 and self.key == 1:
            return [23, 21, 27]
        elif state == 23:
            return [22, 24, 28]
        elif state == 16 and self.key == 0:
            return [11, 15, 17]
        elif state == 16 and self.key == 1:
            return [21, 11, 15, 17]
        elif state == 20 and self.key == 0:
            return [15, 25]
        elif state == 20 and self.key == 1:
            return [21, 15, 25]
        elif state == 26 and self.key == 0:
            return [25, 27]
        elif state == 26 and self.key == 1:
            return [21, 25, 27]
        elif state == 22 and self.key == 0:
            return [23, 27]
        elif state == 22 and self.key == 1:
            return [21, 23, 27]

        x = state / 5
        y = state % 5
        # print x,y
        if y - 1 < 0 and x - 1 < 0:
            if x + 1 < 6:
                return [x * 5 + y + 1, (x + 1) * 5 + y]
            else:
                return [x * 5 + y + 1]
        elif y - 1 < 0:
            if x + 1 < 6:
                return [x * 5 + y + 1, (x - 1) * 5 + y, (x + 1) * 5 + y]
            else:
                return [x * 5 + y + 1, (x - 1) * 5 + y]
        elif x - 1 < 0:
            if y % 4 == 0:
                if x + 1 < 6:
                    return [x * 5 + y - 1, (x + 1) * 5 + y]
                else:
                    return [x * 5 + y - 1]
            else:
                if x + 1 < 6:
                    return [x * 5 + y + 1, x * 5 + y - 1, (x + 1) * 5 + y]
                else:
                    return [x * 5 + y + 1, x * 5 + y - 1]
        else:
            if y % 4 == 0:
                if x + 1 < 6:
                    return [x * 5 + y - 1, (x - 1) * 5 + y, (x + 1) * 5 + y]
                else:
                    return [x * 5 + y - 1, (x - 1) * 5 + y]
            else:
                if x + 1 < 6:
                    return [x * 5 + y + 1, x * 5 + y - 1, (x - 1) * 5 + y, (x + 1) * 5 + y]
                else:
                    return [x * 5 + y + 1, x * 5 + y - 1, (x - 1) * 5 + y]

    def get_Reward(self, end):
        tmp = 0
        x = end / 5
        y = end % 5
        p = random.randint(1, 4)
        # print p
        if p == 1:
            self.jailer = 3
            tmp = 1
        elif p == 2:
            self.jailer = 4
        elif p == 3:
            self.jailer = 8
        elif p == 4:
            self.jailer = 9

        if end == self.jailer:
            # print "You have been failed."
            return -30
        else:
            # if end == 3 or end == 23:
            if end == 23 and self.key == 0:
                self.key = 1
                return 5
            elif end == 23 and self.key == 1:
                return -1
            elif end == 3 and tmp == 0 and self.key == 1:
                return -1
            elif end == 3 and tmp == 0 and self.key == 0:
                self.key = 1
                return 5
            elif end == 6 or end == 11 or end == 19 or end == 22 or end == 20 or end == 26:
                return -10
            elif self.key == 1 and end == 21:
                return 50
            elif self.key == 0 and end == 21:
                return -1
            else:
                return -1


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
        # print nextState
        V_table = np.zeros(len(nextState))
        for t in range(len(nextState)):
            V_table[t] = self.State_value[nextState[t]] / 1000
            V_table[t] = np.exp(V_table[t])
        sumofQ = sum(V_table)
        for r in range(len(V_table)):
            V_table[r] = V_table[r] / sumofQ
        # print V_table
        probability = random.uniform(0, 1)
        # print probability
        i = 0
        sumation = 0
        for l in range(len(V_table)):
            sumation = sumation + V_table[l]
            # print ("Sum:", sumation)
            if l == 0:
                if probability <= V_table[l]:
                    i = 0
                    # print (l)
                    # print (i)
                    # print ("1:",i)
            elif l == len(V_table) - 1:
                if sumation - V_table[l] < probability <= 1:
                    i = len(V_table) - 1
                    # print (l)
                    # print (i)
                    # print ("2:")
            else:
                if V_table[l - 1] < probability <= sumation:
                    i = l
                    # print l
                    # print ("3:")
                    # break

        next_State = prison.next_State(state)[i]
        # print next_State
        return next_State

    def learnV(self, state, nextState, reward):
        # if state != 21:
        td_error = reward + self.gamma * self.State_value[nextState] - self.State_value[state]
        self.eligibility[state] = self.eligibility[state] + 1
        for x in range(30):
            self.eligibility[x] *= self.lamb * self.gamma * self.eligibility[x]
            self.State_value[x] = self.State_value[x] + self.alpha * td_error * self.eligibility[x]
        # print (self.State_value)
        # elif state == 21:
        #    td_error = reward - self.State_value[state]
        #    self.eligibility[state] *= self.lamb * self.gamma
        #    self.State_value[state] = self.State_value[state] + self.alpha * td_error


tdlambda = TD_Lambda(0.5, 0.2, 0.9)
# tdlambda.State_value = 0
tdlambda.chooseState_softmax(16)


# b = a.next_State(26)
#m = prison.get_Reward(1)
#print m

# a = get_reward(0,4, prison)
# print a

def update(current_state):
    if current_state == 23:
        return current_state
    elif currentState == 3 and prison.jailer != 3:
        return currentState
    elif currentState == 3 and prison.jailer == 3:
        return 1
    elif current_state != 23 or currentState != 3:
        # print ("current_state:", current_state)
        nextState = tdlambda.chooseState_softmax(current_state)
        # print nextState
        if nextState != 23:
            # print ("nextState:", nextState)
            instant_reward = prison.get_Reward(nextState)
            # print ("instant_reward:", instant_reward)
            tdlambda.learnV(current_state, nextState, instant_reward)
            # print (tdlambda.State_value)
            # print nextState
            return nextState
        else:
            # print "The of Episode."
            return nextState

def update2(current_state):
    if current_state == 3 and prison.jailer != 3:
        return current_state
    elif currentState == 3 and prison.jailer == 3:
        return 2
    elif current_state != 3:
        # print ("current_state:", current_state)
        nextState = tdlambda.chooseState_softmax(current_state)
        # print nextState
        if nextState != 3:
            # print ("nextState:", nextState)
            instant_reward = prison.get_Reward(nextState)
            # print ("instant_reward:", instant_reward)
            tdlambda.learnV(current_state, nextState, instant_reward)
            # print (tdlambda.State_value)
            # print nextState
            return nextState
        else:
            return nextState

def update1(current_state):
    if current_state == 21:
        return current_state
    elif current_state != 21:
        # print ("current_state:", current_state)
        nextState = tdlambda.chooseState_softmax(current_state)
        # print nextState
        if nextState != 21:
            # print ("nextState:", nextState)
            instant_reward = prison.get_Reward(nextState)
            # print ("instant_reward:", instant_reward)
            tdlambda.learnV(current_state, nextState, instant_reward)
            # print (tdlambda.State_value)
            # print nextState
            return nextState
        else:
            # print "The of Episode."
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
    temp = 0
    for i in range(30):
        tdlambda.State_value[i] = prison.get_Reward(i)
        if tdlambda.State_value[i] == -30:
            #print i
            if i == 3:
                tdlambda.State_value[4] = -1
                tdlambda.State_value[8] = -1
                tdlambda.State_value[9] = -1
            elif i == 4:
                tdlambda.State_value[3] = -1
                tdlambda.State_value[8] = -1
                tdlambda.State_value[9] = -1
            elif i == 8:
                tdlambda.State_value[4] = -1
                tdlambda.State_value[3] = -1
                tdlambda.State_value[9] = -1
            elif i == 9:
                tdlambda.State_value[4] = -1
                tdlambda.State_value[8] = -1
                tdlambda.State_value[3] = -1
    if tdlambda.State_value[3] + tdlambda.State_value[4] + tdlambda.State_value[8] + tdlambda.State_value[9] == -4:
        p = random.randint(1, 4)
        if p == 1:
            tdlambda.State_value[3] = -30
        elif p == 2:
            tdlambda.State_value[4] = -30
        elif p == 3:
            tdlambda.State_value[8] = -30
        elif p == 4:
            tdlambda.State_value[9] = -30
    for j in range(30):
        tdlambda.eligibility[j] = 0
    # print i, tdlambda.State_value[i]
    # print tdlambda.State_value


init()
# print tdlambda.State_value

# update(0)
training = 100
sum16811 = 0
sum12311 = 0
sum_Time = {}
path = []
value = []
path1 = []
average_Reward = 0
minV = 0
for i in range(training):
    currentState = 1
    path = []
    #print (i)
    #while currentState != 3:
    #    nextState = update2(currentState)
    #    path = np.append(path, [nextState])
    #    average_Reward = (prison.get_Reward(nextState) + average_Reward)
    #    currentState = nextState
    #    print path
    #if currentState == 3:
    #    if prison.jailer != 3:
    #        prison.key = 1
    #        tdlambda.State_value[3] = 5
    while currentState != 23 or currentState != 3:
        nextState = update(currentState)
        path = np.append(path, [nextState])
        average_Reward = (prison.get_Reward(nextState) + average_Reward)
        #print (path)
        # print (currentState, nextState)
        # sum_Time[(currentState, nextState)] = sum_Time[(currentState, nextState)] + sarsa.Q[(currentState, nextState)]
        currentState = nextState
        # print path
    # print path
        if currentState == 23 or currentState == 3:
            tdlambda.State_value[23] = 5
            tdlambda.State_value[3] = 5
            prison.key = 1
            break
    while currentState != 21:
        nextState = update1(currentState)
        path = np.append(path, [nextState])
        #print ("IR:", prison.get_Reward(nextState))
        average_Reward = (prison.get_Reward(nextState) + average_Reward)

        #print ("AR:", average_Reward)
        # print (path)
        # print (currentState, nextState)
        # sum_Time[(currentState, nextState)] = sum_Time[(currentState, nextState)] + sarsa.Q[(currentState, nextState)]
        currentState = nextState
        # print path
    if currentState == 21:
        tdlambda.State_value[21] = 50
    #print (path)
    #print average_Reward, tdlambda.State_value
    prison.key = 0
    value = np.append(value, [average_Reward])
    tmp = minV
    minV = np.max(value)
    if minV != tmp:
        path1 = np.append(path, minV)
    average_Reward = 0
    # time = [getT(currentState, a, sum_Time) for a in mapping.next_State(currentState)]
    # max_sumT = max(time)
print value
print path1
