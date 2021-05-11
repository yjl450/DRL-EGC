from gym.envs.registration import register

STEP = 1000

register(
    id='Elevator-v0',
    entry_point='gym_elevator.envs.environment:ElevatorEnv',
    max_episode_steps=STEP,
    kwargs={'elevator_num': 1, 'elevator_limit': 10, 'floor_num': 2, 'floor_limit': 40,
            'step_size': STEP, 'poisson_lambda': 1, 'seed': 0, "reward_func": 4, 
            "unload_reward": 100, "load_reward": 100, "discount": None},
)