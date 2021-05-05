from gym import register, make

def reg(step = 500):
    register(
        id='Elevator-v0',
        entry_point='environment:ElevatorEnv',
        max_episode_steps=step,
        kwargs={'elevator_num': 4, 'elevator_limit': 13, 'floor_num': 5, 'floor_limit': 40,
                'step_size': 500, 'poisson_lambda': 3, 'seed': 1, "reward_func": 1, "unload_reward": None, "load_reward": None},
    )

def make_env(step):
    reg(step)
    env = make('Elevator-v0')
    env.reset()
    return env


def env_decoder(env):
    elevator_floor = env.state[:env.elevator_num]
    # if env.elevator_num <= 3:
    #     verbose = True
    # if verbose:
    #     for i in range(env.elevator_num):
    #         print("Elevator", i, "Passengers:", env.state[env.elevator_num + i*env.elevator_limit: env.elevator_num + (i+1)*env.elevator_limit])
    #     for i in range(env.floor_num):
    #         print("Floor", i+1, "Passengers:", env.state[env.elevator_num + env.elevator_num * env.elevator_limit + i*env.floor_limit:env.elevator_num + env.elevator_num * env.elevator_limit + (i+1)*env.floor_limit])
    # else:
    #     print("Elevator Passengers:", env.state[env.elevator_num: env.elevator_num + env.elevator_num * env.elevator_limit])
    #     print("Floor Passengers:", env.state[env.elevator_num + env.elevator_num * env.elevator_limit:])
    return env.elevator_num, elevator_floor, env.floor_num
