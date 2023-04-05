from maze_env import Maze
from RL_brain import DeepQNetwork
import time
def run_maze():
	step = 0
	print("====Start Game====")
	for episode in range(400):
		observation = env.reset()
		step_every_episode = 0
		env.render()
		while True:
			if(episode>350):
				time.sleep(0.1)		
			action = RL.choose_action(observation)
			observation_, reward, done = env.step(action)
			env.render()
			RL.store_transition(observation, action, reward, observation_)
			if (step>200) and (step%5==0):
				RL.learn()
			observation = observation_
			
			step += 1
			step_every_episode += 1
			if done:
				if(reward==1):
					print("挑战成功! 本次所尝试的步数：{}".format(step_every_episode))
				elif(reward==-1):
					print("挑战失败! 本次所尝试的步数：{}".format(step_every_episode))
				break
	print("====Game Over====")
	env.destroy()

if __name__ == '__main__':
	env = Maze()
	RL = DeepQNetwork(env.n_actions, env.n_features,
					learning_rate=0.01,
					reward_decay=0.9,
					e_greedy=0.9,
					replace_target_iter=200,
					memory_size=2000
					)
	env.after(100, run_maze)
	env.mainloop()
	RL.plot_cost()