from gym.envs.registration import register

STEP = 1000

register(
    id='Elevator-v0',
    entry_point='gym_elevator.envs.environment:ElevatorEnv',
    max_episode_steps=STEP,
    kwargs={'elevator_num': 3, 'elevator_limit': 10, 'floor_num': 4, 'floor_limit': 20,
            'step_size': STEP, 'poisson_lambda': 50, 'seed': 1, "reward_func": 2, 
            "unload_reward": None, "load_reward": None},
)
