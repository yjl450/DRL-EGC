from gym.envs.registration import register

STEP = 1000

register(
    id='Elevator-v0',
    entry_point='gym_elevator.envs.environment:ElevatorEnv',
    max_episode_steps=STEP,
    kwargs={'elevator_num': 4, 'elevator_limit': 10, 'floor_num': 10, 'floor_limit': 40,
            'step_size': STEP, 'poisson_lambda': 3, 'seed': 0, "reward_func": 3, 
            "unload_reward": 100, "load_reward": 100, "punish":-100, "discount": 0.99},
)