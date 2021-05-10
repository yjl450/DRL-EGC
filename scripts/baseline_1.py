import gym

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('gym_elevator:Elevator-v0')

# Instantiate the agent
# model = DQN('MlpPolicy', env, verbose=1)
model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-2)
# Train the agent
model.learn(total_timesteps=int(60000))
# Save the agent
model.save("a2c_ele")
# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# model = DQN.load("a2c_ele", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward, std_reward)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
