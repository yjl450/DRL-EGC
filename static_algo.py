from gym import register, make, logger
import numpy as np
from utils import make_env, env_decoder


def dumb_rotater(env, step, render=True):
    if render:
        env.render()
    env.step(env.encodeAction([2] * env.elevator_num))
    if render:
        env.render()
    action = [2] * env.elevator_num
    for i in range(env.elevator_num):
        action[i] = 1
        for j in range(0, i):
            action[j] = env.direction[j]
            if env.state[j] == env.floor_num:
                action[j] = 0
            elif env.state[j] == 1:
                action[j] = 1
        env.step(env.encodeAction(action))
        if render:
            env.render()
        env.step(env.encodeAction([2] * env.elevator_num))
        if render:
            env.render()
    while env.step_index < step:
        action = [2] * env.elevator_num
        for j in range(env.elevator_num):
            action[j] = env.direction[j]
            if env.state[j] == env.floor_num:
                action[j] = 0
            elif env.state[j] == 1:
                action[j] = 1
        env.step(env.encodeAction(action))
        if render:
            env.render()
        env.step(env.encodeAction([2] * env.elevator_num))
        if render:
            env.render()


def nearest_car(env, step, render=True):
    dst = np.zeros((env.elevator_num, env.floor_num))
    if render:
        env.render()
    while env.step_index < step:
        new_call = env.new_hall_call.copy()
        current_floors = env.state[: env.elevator_num]
        current_directions = env.direction
        for i in range(env.elevator_num):
            for j in range(env.elevator_limit):
                if env.state[env.elevator_num + i * env.elevator_num * env.elevator_limit + j]:
                    dst[i, int(env.state[env.elevator_num + i * env.elevator_num * env.elevator_limit + j]) - 1]
        for i in range(env.floor_num):
            if new_call[i][0] == 1: # up call
                nearest = 0
                nearest_distance = 10000
                for j in range(env.elevator_num):
                    if current_directions[j] == 1 and current_floors[j] <= i + 1:
                        distance = i + 1 - current_floors[j]
                    elif current_directions[j] == 1 and current_floors[j] > i + 1:
                        distance = 2 * env.floor_num - current_floors[j] + i
                    elif current_directions[j] != 1:
                        distance = current_floors[j] - 1 + i
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest = j
                dst[nearest, i] = 1
            if new_call[i][1] == 1: # down call
                nearest = 0
                nearest_distance = 10000
                for j in range(env.elevator_num):
                    if current_directions[j] == 0 and current_floors[j] >= i + 1:
                        distance = current_floors[j] - (i + 1)
                    elif current_directions[j] == 0 and current_floors[j] < i + 1:
                        distance = current_floors[j] - 1 + 2 * env.floor_num - 2 - i
                    elif current_directions[j] != 0:
                        distance = env.floor_num - current_floors[j] + env.floor_num - 1 - i
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest = j
                dst[nearest, i] = 1
        env.step(dst_dispatcher(env, dst))
        if render:
            env.render()

def dst_dispatcher(env, dst):
    actions = [2] * env.elevator_num
    for i in range(env.elevator_num):
        current_floor = int(env.state[i])
        current_direction = env.direction[i]
        if dst[i, current_floor - 1]:
            dst[i, current_floor - 1] = 0
            continue
        up = False
        for j in range(current_floor, env.floor_num):
            if dst[i, j]:
                actions[i] = 1
                up = True
                break
        if up:
            continue
        for j in range(0, current_floor - 1):
            if dst[i, j]:
                actions[i] = 0
                up = False
                break
    return env.encodeAction(actions)


def controller(algo, env, step, render=True):
    algos = {'nearest_car': nearest_car, 'dumb_rotater': dumb_rotater}
    algos[algo](env, step, render)


if __name__ == '__main__':
    step = 500

    # env = make_env(step)
    # controller('nearest_car', env, step, render=True)
    # print(env.waited, env.travelled, env.arrived)
    # env.render_close()

    env = make_env(step)
    controller('nearest_car', env, step, render=True)
    print(env.waited, env.travelled, env.arrived)
    env.render_close()
