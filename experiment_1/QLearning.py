import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

ALPHA = 0.1 #learning rate
GAMMA = 0.95 #discount factor
EPSILION = 0.9 #greedy police
N_STATE = 20 #the length of the 1 dimensional world
ACTIONS = ['left', 'right'] #available actions
MAX_EPISODES = 200 #maximum episodes
FRESH_TIME = 0.1 #fresh time for one move


#############  1. Define Q table  ##############
'''
所构造的q表的大小是:行数是财产的探索者总共所处的位置数,列数就是动作的数量
'''
def build_q_table(n_state, actions):
    q_table = pd.DataFrame(
    np.zeros((n_state, len(actions))),
    np.arange(n_state),
    actions
    )
    return q_table

#############  2. Define action  ##############

def choose_action(state, q_table):
    #epslion - greedy policy
    state_action = q_table.loc[state,:] #将现在agent观测到的状态所对应的q值取出来
    if np.random.uniform()>EPSILION or (state_action==0).all(): #当生撑随机数大于EPSILON或者状态state所对应的q值全部为0时，就随机选择状态state所对应的动作
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_action.idxmax() #选择状态state所对应的使q值最大的动作
    return action_name

#############  3. Environment feedback  ##############

def get_env_feedback(state, action):
    #选择动作之后，还要根据现在的状态和动作获得下一个状态，并且返回奖励，这个奖励是环境给出的，用来评价当前动作的好坏。
    #这里设置的是，只有在获得宝藏是才给奖励，没有获得奖励时，无论是向左移动还是向右移动，给出的即时奖励都是0.
    if action=='right': #move right
        if state == N_STATE-2: #terminate
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state+1
            reward = -0.5
    else: #move left
        if state == 0:
            next_state = 0 #reach the wall
            
        else:
            next_state = state-1
        reward = -0.5
    return next_state, reward


#############  4. Update environment   ##############

def update_env(state,episode, step_counter):
    #更新环境的函数，比如向右移动之后，o表示的agent就距离宝藏进了一步，将agent随处的位置实时打印出来    
    env = ['-'] *(N_STATE-1)+['T'] #'---------T' our environment
    if state =='terminal':
        print("Episode {}, the total step is {}".format(episode+1, step_counter))
        final_env = ['-'] *(N_STATE-1)+['T']
        return True, step_counter
    else:
        env[state]='*'
        env = ''.join(env)
        print(env)
        time.sleep(FRESH_TIME)
        return False, step_counter
        
#############  5. Agent   #################
'''
Please complete the code for this section.
Return value:
-- q_table : Refer to the function 'build_q_table'
-- step_counter_times : List: the number of total steps for every episode.
'''
def q_learning():
    # 开始更新q表
    q_table = build_q_table(N_STATE, ACTIONS) #随机初始化一下q表
    step_counter_times = [0]
    '''
    main loop: 
    '''
    for episode in range(MAX_EPISODES):
        state = 0 #每个episode开始时都将agent初始化在最开始的地方                                  
        is_terminal = False #Judgment variable: whether to end episode.
        step_counter = 0 #记录走了多少步
        update_env(state, episode, step_counter) #打印的就是o-----T
        while not is_terminal:
            '''
            Please complete the content after "while not is_terminal:" 
            according to the algorithm idea and known functions.

            '''
            step_counter += 1
            action = choose_action(state, q_table) #agent根据当前的状态选择动作
            next_state, reward = get_env_feedback(state, action) #上一步已经获得了state对应的动作action，接着我们要获得下一个时间步的状态
            q_predict = q_table.loc[state, action]
            if next_state != 'terminal': #要判断一下，下一个时间步的是不是已经取得宝藏了，如果不是，可以按照公式进行更新
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max() #next state is not terminal
            else: #如果已经得到了宝藏，得到的下一个状态不在q_table中，q_target的计算也不同。
                q_target = reward     #next state is terminal
                is_terminal = True    #terminate this episode
                step_counter_times.append(step_counter)

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  #update
            state = next_state  #move to next state

            update_env(state, episode, step_counter)
            
                
 

    return q_table, step_counter_times



def main():
    q_table, step_counter_times= q_learning()
    print("Q table\n{}\n".format(q_table))
    print('end')

    print(step_counter_times)
    plt.plot(step_counter_times,'b-')
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.show()

main() 