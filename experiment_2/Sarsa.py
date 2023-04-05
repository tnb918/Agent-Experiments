import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

ALPHA = 0.1
GAMMA = 0.95
EPSILION = 0.9
N_STATE = 20
ACTIONS = ['left', 'right']
MAX_EPISODES = 200
FRESH_TIME = 0.1

#############  1. Define Q table  ##############

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
    state_action = q_table.loc[state,:]
    if np.random.uniform()>EPSILION or (state_action==0).all():
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_action.idxmax()
    return action_name

#############  3. Environment feedback  ##############

def get_env_feedback(state, action):
    if action=='right':
        if state == N_STATE-2:
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state+1
            reward = -0.5
    else:
        if state == 0:
            next_state = 0
            
        else:
            next_state = state-1
        reward = -0.5
    return next_state, reward

#############  4. Update environment   ##############

def update_env(state,episode, step_counter):
    env = ['-'] *(N_STATE-1)+['T']
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
def sarsa_learning():
    q_table = build_q_table(N_STATE, ACTIONS)
    step_counter_times = [0]

    '''
    main loop: 
    '''
    for episode in range(MAX_EPISODES):
        state = 0
        is_terminal = False
        step_counter = 0
        action = choose_action(state, q_table) #agent根据当前的状态选择动作
        update_env(state, episode, step_counter)
        while not is_terminal:
            '''
            Please complete the content after "while not is_terminal:" 
            according to the algorithm idea and known functions.
            '''
            step_counter += 1
            next_state, reward = get_env_feedback(state, action) #上一步已经获得了state对应的动作action，接着我们要获得下一个时间步的状态
            q_predict = q_table.loc[state, action]
            if next_state != 'terminal': #要判断一下，下一个时间步的是不是已经取得宝藏了，如果不是，可以按照公式进行更新
                next_action = choose_action(next_state, q_table)
                q_target = reward + GAMMA * q_table.loc[next_state, next_action] #next state is not terminal
            else: #如果已经得到了宝藏，得到的下一个状态不在q_table中，q_target的计算也不同。
                q_target = reward     #next state is terminal
                is_terminal = True    #terminate this episode
                step_counter_times.append(step_counter)  

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  #update
            state = next_state  #move to next state
            action = next_action 

            update_env(state, episode, step_counter)
                        
                                   
    return q_table, step_counter_times

def main():
    q_table, step_counter_times= sarsa_learning()
    print("Q table\n{}\n".format(q_table))
    print('end')
    
    print(step_counter_times)
    plt.plot(step_counter_times,'b-')
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.show()

main() 