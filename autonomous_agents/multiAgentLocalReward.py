import re
import numpy as np
import math
import random
import plotly.graph_objects as go
import operator

GRID_ROWS = 5
GRID_COLS = 10
WIN_STATE = [(1, 3),(2,2)]
CONST_WIN_STATE = WIN_STATE

def genRandomState(grid_rows, grid_cols, win_state):
    x = random.randint(0,grid_rows-1)
    y = random.randint(0,grid_cols-1)
    
    while ((x,y) in win_state):
        x = random.randint(0,grid_rows-1)
        y = random.randint(0,grid_cols-1)
    return (x,y)

class State:
    def __init__(self, state):
        self.grid = np.zeros([GRID_ROWS, GRID_COLS])  ### Change this to average of all rewards 
        self.state = state
        self.isEnd = False
        self.wins = []

    def generateReward(self):
        global WIN_STATE
        if self.state in WIN_STATE:
            WIN_STATE = [s for s in WIN_STATE if not (s==self.state)]
            # self.wins = self.wins + 1
            return 20
        else:
            return -1

    def getReward(self, state):
        if state in CONST_WIN_STATE:
            return 20
        else:
            return -1

    def totalReward(self, states):
        reward = 0
        for state in states:
            reward = reward + self.getReward((state[0],state[1]))
        return reward

    def isEndFunc(self):
        global WIN_STATE
        if self.state in WIN_STATE:
            self.isEnd = True
        return self.isEnd
    
    def isEndFuncEval(self):
        win_states = CONST_WIN_STATE
        if self.state in win_states:
            self.isEnd = True
        return self.isEnd
    ##### Action: up, down, left, right, stay
    def nextPos(self, action):
        # print("state in nxtpos:", self.state)
        acts = ["up", "down", "left", "right"]
        
        flag = 0
        while flag==0:
            acts = [act for act in acts if act!=action]
            # print(acts)
            if action == "up":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nextState = (self.state[0], self.state[1] - 1)
            elif action == "right":
                nextState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nextState[0] >= 0) and (nextState[0] <= (GRID_ROWS -1)):
                if (nextState[1] >= 0) and (nextState[1] <= (GRID_COLS -1)):
                        flag = 1
            if flag ==0:
                action = random.choice(acts)
                        
        return nextState, action ###### stay on same position

    def showBoard(self):
        for i in range(0, GRID_ROWS):
            out = '| '
            for j in range(0, GRID_COLS):
                if self.grid[i, j] == 0:
                    token = '0'
                if  np.isnan(self.grid[i, j])==1:
                    token = 'NA'
                if (i,j) == self.state:
                    token ='*'
                if (i,j) in WIN_STATE:
                    token ='W'
                out += token + ' | '
            print(out)


# Agent of player

class qLearning:

    def __init__(self, start_state):
        self.actions = ["up", "down", "left", "right"]
        self.actions_lookup = {
            "up":0,
            "down":1,
            "left":2,
            "right":3
        }
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 0.2

        ## Initialize Agent 1
        self.StateA1 = State(start_state)
        # initial state reward
        self.state_values_a1 = {}
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                for k in range(0,4):
                    self.state_values_a1[(i, j, k)] = 0  # set initial value to 0
        # print("i m in init")

        self.action_a1 = self.chooseActionA1()
        self.states_a1 = [(self.StateA1.state[0],self.StateA1.state[1],self.actions_lookup[self.action_a1])]
        self.isEndA1 = False
        
        ## Initialize Agent 2
        self.StateA2 = State(start_state)
        # initial state reward
        self.state_values_a2 = {}
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                for k in range(0,4):
                    self.state_values_a2[(i, j, k)] = 0  # set initial value to 0
        # print("i m in init")

        self.action_a2 = self.chooseActionA1()
        self.states_a2 = [(self.StateA2.state[0],self.StateA2.state[1],self.actions_lookup[self.action_a2])]
        self.isEndA2 = False

    def chooseActionA1(self):
        # choose action with most expected value
        action = ""

        if np.random.uniform(0, 1) <= self.epsilon:
            pot_action = np.random.choice(self.actions)
            nxt_pos, action = self.StateA1.nextPos(pot_action)
        else:
            # greedy action
            reward_acts = {}
            for a in self.actions:
                # if the action is deterministic
                # print('curr_state: ', self.State.state)
                nxt_pos, chosen_action = self.StateA1.nextPos(a)
                # print("pos action and state: ", nxt_pos, chosen_action)
            
                if chosen_action not in list(reward_acts.keys()):
                    nxt_rewards = [self.state_values_a1[(nxt_pos[0], nxt_pos[1], 0)],
                            self.state_values_a1[(nxt_pos[0], nxt_pos[1], 1)],
                            self.state_values_a1[(nxt_pos[0], nxt_pos[1], 2)],
                            self.state_values_a1[(nxt_pos[0], nxt_pos[1], 3)]]
                    nxt_reward = max(nxt_rewards)
                    reward_acts[chosen_action] = nxt_reward   
            # print(reward_acts, nxt_pos)
            max_act_val = reward_acts[max(reward_acts, key=reward_acts.get)]
            max_acts = [key for key,val in reward_acts.items() if val==max_act_val]
            
            action = random.choice(max_acts) 
            # print("R,np,act: ",reward_acts, nxt_pos, action)
        # print("returned action:", action)
        return action
    
    def chooseActionA2(self):
        # choose action with most expected value
        action = ""

        if np.random.uniform(0, 1) <= self.epsilon:
            pot_action = np.random.choice(self.actions)
            nxt_pos, action = self.StateA2.nextPos(pot_action)
        else:
            # greedy action
            reward_acts = {}
            for a in self.actions:
                # if the action is deterministic
                # print('curr_state: ', self.State.state)
                nxt_pos, chosen_action = self.StateA2.nextPos(a)
                # print("pos action and state: ", nxt_pos, chosen_action)
            
                if chosen_action not in list(reward_acts.keys()):
                    nxt_rewards = [self.state_values_a2[(nxt_pos[0], nxt_pos[1], 0)],
                            self.state_values_a2[(nxt_pos[0], nxt_pos[1], 1)],
                            self.state_values_a2[(nxt_pos[0], nxt_pos[1], 2)],
                            self.state_values_a2[(nxt_pos[0], nxt_pos[1], 3)]]
                    nxt_reward = max(nxt_rewards)
                    reward_acts[chosen_action] = nxt_reward   
            # print(reward_acts, nxt_pos)
            max_act_val = reward_acts[max(reward_acts, key=reward_acts.get)]
            max_acts = [key for key,val in reward_acts.items() if val==max_act_val]
            
            action = random.choice(max_acts) 
            # print("R,np,act: ",reward_acts, nxt_pos, action)
        # print("returned action:", action)
        return action

    def takeActionA1(self, action):
        position, _ = self.StateA1.nextPos(action)
        return State(state=position)
    
    def takeActionA2(self, action):
        position, _ = self.StateA2.nextPos(action)
        return State(state=position)

    def reset(self):
        start_state = genRandomState(GRID_ROWS, GRID_COLS , WIN_STATE)
        # print("new start state: ", start_state)
        self.StateA1 = State(start_state)
        self.action_a1 = self.chooseActionA1()
        self.states_a1 = [(self.StateA1.state[0],self.StateA1.state[1],self.actions_lookup[self.action_a1])]

        self.StateA2 = State(start_state)
        self.action_a2 = self.chooseActionA2()
        self.states_a2 = [(self.StateA2.state[0],self.StateA2.state[1],self.actions_lookup[self.action_a2])]


    def start_a1(self, state): #### Used in Evaluation
        self.StateA1 = State(state)
        self.action_a1 = self.chooseActionA1()
        self.states_a1 = [(self.StateA1.state[0],self.StateA1.state[1],self.actions_lookup[self.action_a1])]

    def start_a2(self, state): #### Used in Evaluation
        self.StateA2 = State(state)
        self.action_a2 = self.chooseActionA1()
        self.states_a2 = [(self.StateA2.state[0],self.StateA2.state[1],self.actions_lookup[self.action_a2])]

    def play(self, num_episodes=1, num_steps = 20):
        i = 0
        curr_steps = 0
        rewards_a1 = []
        rewards_a2 = []
        flag_a1 = False
        flag_a2 = False
        while i < num_episodes:
            # to the end of game back propagate reward
            if (self.isEndA1) or (curr_steps == num_steps):
                # back propagate
                if flag_a1 == False:
                    reward_a1 = self.StateA1.generateReward()
                    # explicitly assign end state to reward values
                    self.state_values_a1[(self.StateA1.state[0], self.StateA1.state[1], self.actions_lookup[self.action_a1])] = reward_a1
                    
                    rewards_a1.append(self.evaluate_a1((0,0)))
                    flag_a1 = True
            if (self.isEndA2) or (curr_steps == num_steps):
                if flag_a2 == False:
                    # back propagate
                    reward_a2 = self.StateA2.generateReward()
                    # explicitly assign end state to reward values
                    self.state_values_a2[(self.StateA2.state[0], self.StateA2.state[1], self.actions_lookup[self.action_a2])] = reward_a2
                    
                    rewards_a2.append(self.evaluate_a2((0,0)))
                    flag_a2 = True

            if (self.isEndA1 and self.isEndA2) or (curr_steps == num_steps):
                self.reset()
                i += 1
                curr_steps = 0
                self.isEndA1 = False
                self.isEndA2 = False
                flag_a1 = False
                flag_a2 = False
                global WIN_STATE
                WIN_STATE = [(1, 3),(2,2)]
            else:
                # print("in else of play")

                # Agent 1
                # append trace
                if self.isEndA1 == False:
                    curr_state = self.StateA1.state
                    curr_action = self.action_a1
                    self.isEndA1 = self.StateA1.isEndFunc()
                    curr_reward = self.StateA1.generateReward()
                    self.StateA1 = self.takeActionA1(curr_action) ## Update to next state
                    nxt_state = self.StateA1.state
                    nxt_action = self.chooseActionA1()
                    self.action_a1 = nxt_action
                    self.states_a1.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
                    # print("current position {} action {}".format(self.State.state, action))

                    # by taking the action, it reaches the next state
                    self.state_values_a1[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] = \
                                self.state_values_a1[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] + \
                                self.alpha*(curr_reward + \
                                (self.gamma*self.state_values_a1[(nxt_state[0], nxt_state[1], self.actions_lookup[nxt_action])]) - \
                                self.state_values_a1[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])])
                    # mark is end

                # Agent 2
                # append trace
                if self.isEndA2 == False:
                    curr_state = self.StateA2.state
                    curr_action = self.action_a2
                    self.isEndA2 = self.StateA2.isEndFunc()
                    curr_reward = self.StateA2.generateReward()
                    self.StateA2 = self.takeActionA2(curr_action) ## Update to next state
                    nxt_state = self.StateA2.state
                    nxt_action = self.chooseActionA2()
                    self.action_a2 = nxt_action
                    self.states_a2.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
                    # print("current position {} action {}".format(self.State.state, action))

                    # by taking the action, it reaches the next state
                    self.state_values_a2[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] = \
                                self.state_values_a2[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] + \
                                self.alpha*(curr_reward + \
                                (self.gamma*self.state_values_a2[(nxt_state[0], nxt_state[1], self.actions_lookup[nxt_action])]) - \
                                self.state_values_a2[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])])
                    # mark is end
                curr_steps = curr_steps + 1
        return rewards_a1, rewards_a2

    def evaluate_a1(self,state, num_steps = 20):
        self.start_a1(state)
        curr_steps = 0
        isEnd = False
        while not ((isEnd) or (curr_steps == num_steps)):
            # print(self.State.isEnd, curr_steps)
            curr_action = self.action_a1
            self.StateA1 = self.takeActionA1(curr_action) ## Update to next state
            nxt_state = self.StateA1.state
            nxt_action = self.chooseActionA1()
            self.action_a1 = nxt_action
            self.states_a1.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
            # print("nxt state: ", nxt_state, nxt_action)
            # print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            # mark is end
            isEnd = self.StateA1.isEndFuncEval()
            curr_steps = curr_steps + 1
        reward = self.StateA1.totalReward(self.states_a1)
        print("Eval: all states A1: ", self.states_a1)
        print("Eval Rewards A1: ", reward)
        return reward
    
    def evaluate_a2(self,state, num_steps = 20):
        self.start_a2(state)
        curr_steps = 0
        isEnd = False
        while not ((isEnd) or (curr_steps == num_steps)):
            # print(self.State.isEnd, curr_steps)
            curr_action = self.action_a2
            self.StateA2 = self.takeActionA2(curr_action) ## Update to next state
            nxt_state = self.StateA2.state
            nxt_action = self.chooseActionA2()
            self.action_a2 = nxt_action
            self.states_a2.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
            # print("nxt state: ", nxt_state, nxt_action)
            # print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            # mark is end
            isEnd = self.StateA2.isEndFuncEval()
            curr_steps = curr_steps + 1
        reward = self.StateA2.totalReward(self.states_a2)
        print("Eval: all states A2: ", self.states_a2)
        print("Eval Rewards A2: ", reward)
        return reward

if __name__ == "__main__":
    # s1 = State(state=(3,3))
    # s1.showBoard()
    # s1.blockPos()

    #### Sample Random valid Start State
    start_state = (0,0)
    num_episodes = 5000
    ch_ag_1 = qLearning(start_state)
    rewards_agent1, rewards_agent2 = ch_ag_1.play(num_episodes=num_episodes)

    x = np.linspace(1,num_episodes,num_episodes)
    fig_sarsa = go.Figure()
    fig_sarsa.add_trace(go.Scatter(x=x, y=rewards_agent1, mode='markers', name="Q-Learning"))
    fig_sarsa.update_layout(
    title={'text':"Reward for 1000 episodes Q-learning Changing Win State",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="Episode --------->",
    yaxis_title="Reward -------->",
    legend_title="N")
    
    fig_sarsa.show()

    x = np.linspace(1,num_episodes,num_episodes)
    fig_sarsa = go.Figure()
    fig_sarsa.add_trace(go.Scatter(x=x, y=rewards_agent2, mode='markers', name="Q-Learning"))
    fig_sarsa.update_layout(
    title={'text':"Reward for 1000 episodes Q-learning Changing Win State",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="Episode --------->",
    yaxis_title="Reward -------->",
    legend_title="N")
    
    fig_sarsa.show()
    # fig_sarsa.write_image("ChangingWinStateQLearning.png")

    eval_on = [(1,7),(3,8),(2,5),(1,6),(2,8),(1,8),(4,8),(3,5),(2,6),(2,9),(0,7),(1,5),(3,3),(2,4),(1,5),(2,7),(1,0),(4,2),(3,7),(2,3)]
    rewards_a1_eval = []
    rewards_a2_eval =[]

    for st in eval_on:
        rewards_a1_eval.append(ch_ag_1.evaluate_a1(st))
        rewards_a2_eval.append(ch_ag_1.evaluate_a2(st))
    x = np.linspace(1, len(eval_on), len(eval_on))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rewards_a1_eval, mode='markers', name="Agent 1 Rewards"))
    fig.add_trace(go.Scatter(x=x, y=rewards_a2_eval, mode='markers', name="Agent 2 Rewards"))
    fig.update_layout(
    title={'text':"Reward over 20 trials",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="Trial ------->",
    yaxis_title="Reward ------->",
    legend_title="N")
    
    fig.show()
    # fig.write_image("reward_over_5_trials_2_3.png")