# Elevator Simulator Environment modified from https://github.com/hse-aai-2019-team3/Applied-AI-Technologies
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import queue
# rendering library
import pyglet
from pyglet import gl


class ElevatorEnv(gym.Env):
    """
    Environment for simulating multiple elevators

    Example
    -------
    In an building with 10 floors and 2 elevators

    there could be 3**2 combinations of actions

    such as [0, 1] or [2， 0]

    which 0 means going down, 1 means going up, 2 means iding on current floor

    This list is then reversed and convert to a decimal number 'm'

    which is then passed to env.step(m) to compute next state
    """

    def __init__(self, elevator_num=10, elevator_limit=13, floor_num=15, floor_limit=40,
                 poisson_lambda=1, step_size=1000, seed=0, reward_func=2, unload_reward=None, load_reward=None, discount=None,punish=-100, capacity=0):
        """
        Initializa a EGC system

        Args:
            elevator_num(int): Number of elevators in the system
            elevator_limit(int): capacity of elevators
            floor_num(int): Number of floors in the building
            floor_limit(int): maximum number of people waiting on each floor
            capacity(int): Number of people in the building (deprecated)
            seed(int): Random seed for the environment

        Variables:
            action_space(Discrete(n)): Int in the range [0, n) of valid actions, get the int by calling env.action_space.n
            state (np.array): state of the environment, 
                [floor of each elevator (elevator_num),
                each passenger's dest inside each elevator (elevator_num * elevator_limit),
                each passenger's dest on each floor (floor_num * floor_limit)]
        """
        self.args = {
            "elevator_num": elevator_num, "elevator_limit": elevator_limit, "floor_num": floor_num, "floor_limit": floor_limit,
            "poisson_lambda": poisson_lambda, "step_size": step_size, "seed": seed, "reward_func": reward_func, 
            "unload_reward": unload_reward, "load_reward": load_reward, "discount": discount, "punish": punish
        }
        self.elevator_num = elevator_num
        self.elevator_limit = elevator_limit
        self.floor_num = floor_num
        self.floor_limit = floor_limit
        # self.capacity = capacity
        # self.waiting_passengers = 0
        self.valid_actions = [0, 1, 2]
        self.step_index = 0
        self.lam = poisson_lambda
        self.step_size = step_size
        self.poisson = None
        self.punish_reward = punish

        self.total_waiting_time = 0  # sum of every passenger's waiting time
        self.total_square_waiting_time = 0
        self.total_elevator_time = 0  # sum of the time in the elevator for each passenger
        self.total_square_elevator_time = 0

        self.waiting_time_table = []  # floor num * floor limit
        self.traveling_time_table = []  # elevator_num * elevator_limit

        # count of passengers who waited
        self.waited = 0
        # count of passengers who entered elevator cas
        self.traveled = 0
        # count of passengers who reached destinated
        self.arrived = 0
        self.direction = [2] * elevator_num
        self.new_hall_call = np.zeros((self.floor_num, 2))

        self.reward_func = reward_func
        self.unload_reward = unload_reward
        self.load_reward = load_reward

        self.reward_3 = [[], []]
        self.discount = discount

        # Index where passenger slots starts in the state array
        self.passenger_start_index = self.elevator_num
        # Index where first slot of first floor starts in the state array
        self.floor_start_index = (
            self.elevator_num + (self.elevator_limit*self.elevator_num))

        self.action_space = spaces.Discrete(
            len(self.valid_actions)**self.elevator_num)

        # self.observation_space = spaces.Discrete(
        #     self.elevator_num
        #     + (self.elevator_num * self.elevator_limit)
        #     # + (self.floor_limit * self.floor_num)
        #     + (self.floor_num * 2)
        # )
        self.observation_space = spaces.MultiDiscrete(
            # [
            # self.elevator_num,
            # (self.elevator_num * self.elevator_limit),
            # # + (self.floor_limit * self.floor_num)
            # (self.floor_num * 2)
            # ]
            [self.floor_num + 1] * self.elevator_num +
            [self.floor_num + 1] * (self.elevator_num * self.elevator_limit) +
            [2] * (self.floor_num * 2)
        )

        self.stateQueue = queue.Queue(maxsize=4)

        # self.seed(hash(time.time()))
        self.seed(seed)
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # returns true if any passenger is destinated to a upper floor and how many
    def passengerToUpperFloor(self, state, current_floor):
        count = 0
        # TODO code müsste für mehraufzüge angepasst werden
        for i in range(self.passenger_start_index, self.passenger_start_index+self.elevator_limit):
            if state[i] > current_floor:
                count += 1
        return (count > 0), count

    # returns true if any passenger is destinated to a lower floor and how many
    def passengerToLowerFloor(self, state, current_floor):
        count = 0
        # TODO code müsste für mehraufzüge angepasst werden
        for i in range(self.passenger_start_index, self.passenger_start_index+self.elevator_limit):
            if state[i] < current_floor:
                count += 1
        return (count > 0), count

    # returns true if any passenger is waiting at a upper floor and how many
    def passengerAtUpperFloor(self, state, current_floor):
        count = 0
        next_floor_index = self.floor_start_index + \
            ((current_floor-1)*self.floor_limit) + self.floor_limit
        last_floor_index = self.floor_start_index + \
            (self.floor_num*self.floor_limit)

        for i in range(next_floor_index, last_floor_index):
            if state[i] != 0:
                count += 1
        return (count > 0), count

     # returns true if any passenger is waiting at a upper floor and how many
    def passengerAtLowerFloor(self, state, current_floor):
        count = 0
        previous_floor_last_index = self.floor_start_index + \
            ((current_floor-1)*self.floor_limit)

        for i in range(self.floor_start_index, previous_floor_last_index):
            if state[i] != 0:
                count += 1
        return (count > 0), count

    def passengerAtFloor(self, state, floor):
        count = 0
        current_floor_index = self.floor_start_index + \
            ((floor - 1)*self.floor_limit)

        for i in range(current_floor_index, current_floor_index+self.floor_limit):
            if state[i] != 0:
                count += 1
        return (count > 0), count

    def passengerInElevator(self, state, which_elevator):
        count = 0
        elevator_index_start = self.elevator_num + \
            (which_elevator*self.elevator_limit)
        elevator_index_end = elevator_index_start + self.elevator_limit

        for i in range(elevator_index_start, elevator_index_end):
            if state[i] != 0:
                count += 1
        return (count > 0), count

    # kick out passengers that wanted to get to this floor
    def unloadPassenger(self, state, which_elevator, floor):
        count = 0
        elevator_index_start = self.elevator_num + \
            (which_elevator*self.elevator_limit)
        elevator_index_end = elevator_index_start + self.elevator_limit

        for i in range(elevator_index_start, elevator_index_end):
            if state[i] == floor:
                state[i] = 0
                count += 1

                # calculate the time in the elevator for each passenger for reward
                curr_elevator_time = self.step_index - \
                    self.traveling_time_table[i-self.elevator_num]
                self.total_elevator_time += curr_elevator_time
                self.total_square_elevator_time += curr_elevator_time**2
                self.reward_3[1].append(curr_elevator_time)
                self.traveling_time_table[i-self.elevator_num] = 0

        # if count > 0: print("Elevator", which_elevator,"unload", count, "passengers at floor", floor)
        self.arrived += count
        return (count > 0), count

    def nextPassengerSlot(self, state, which_elevator):
        index = -1
        elevator_index_start = self.elevator_num + \
            (which_elevator*self.elevator_limit)
        elevator_index_end = elevator_index_start + self.elevator_limit

        for i in range(elevator_index_start, elevator_index_end):
            if state[i] == 0:
                index = i
                break
        return index

    # Load passengers from given floor
    # Returns True if at least a passenger was loaded, how many where loaded, and how many where left at the floor

    def loadPassenger(self, state, which_elevator, floor):
        passengers_before_loading = self.passengerAtFloor(state, floor)[1]
        loaded_passengers = 0
        current_floor_index = self.floor_start_index + \
            ((floor - 1)*self.floor_limit)
        for i in range(current_floor_index, current_floor_index+self.floor_limit):
            if state[i] != 0:
                free_slot = self.nextPassengerSlot(state, which_elevator)
                if free_slot != -1:
                    state[free_slot] = state[i]
                    state[i] = 0
                    loaded_passengers += 1

                    # waiting time finishes for each passenger
                    # calculate the squared waiting time for reward
                    # TODO: SUM of squared waiting time?
                    # DEBUG:
                    # self.total_square_waiting_time += (
                    #     self.step_index - self.waiting_time_table[i-current_floor_index])**2
                    # self.total_waiting_time += self.step_index - self.waiting_time_table[i-current_floor_index]
                    # self.waiting_time_table[i-current_floor_index] = 0
                    self.total_square_waiting_time += (
                        self.step_index - self.waiting_time_table[i-self.floor_start_index])**2
                    self.total_waiting_time += self.step_index - \
                        self.waiting_time_table[i-self.floor_start_index]
                    self.reward_3[0].append(
                        self.step_index - self.waiting_time_table[i-self.floor_start_index])
                    self.waiting_time_table[i-self.floor_start_index] = 0

                    # record the loading time for each passenger
                    self.traveling_time_table[free_slot -
                                              self.elevator_num] = self.step_index

                elif free_slot == -1:
                    break
        #self.waiting_passengers -= loaded_passengers
        # question: self.waiting_passengers refers to wait to be loaded? or to the dest?
        #o.w. self.waiting_passengers -= count in unloadPassenger()
        self.traveled += loaded_passengers
        return (loaded_passengers > 0), loaded_passengers, (passengers_before_loading - loaded_passengers)

    def elevatorMoveDown(self, state, action, which_elevator, current_floor):
        # reward = 0
        if state[which_elevator] == 1:
            # reward -= 1000000
            self.punish = True
        else:
            # check if there is reason to go a floor up
            # is there a passenger waiting at a lower a floor?
            passenger_at_lower_floors, num_passenger_at_lower_floor = self.passengerAtLowerFloor(
                state, current_floor)

            # is a passenger inside the elevator destinated to an upper floor?
            passenger_to_lower_floors, num_passenger_to_lower_floor = self.passengerToLowerFloor(
                state, current_floor)

            # calculate rewards
            # if not passenger_at_lower_floors and not passenger_to_lower_floors:
            #     reward -= 100

            # if num_passenger_at_lower_floor > 0:
            #     reward += 1

            # if num_passenger_to_lower_floor > 0:
            #     reward += 10

            state[which_elevator] -= 1
        return state

    def elevatorMoveUp(self, state, action, which_elevator, current_floor):
        reward = 0
        if state[which_elevator] == self.floor_num:
            # reward -= 1000000
            self.punish = True
        else:
            # check if there is reason to go a floor up

            # is there a passenger waiting at a higher a floor?
            passenger_at_upper_floors, num_passenger_at_upper_floor = self.passengerAtUpperFloor(
                state, current_floor)

            # is a passenger inside the elevator destinated to an upper floor?
            passenger_to_upper_floors, num_passenger_to_upper_floor = self.passengerToUpperFloor(
                state, current_floor)

            # calculate rewards
            # if not passenger_at_upper_floors and not passenger_to_upper_floors:
            #     reward -= 100

            # if num_passenger_at_upper_floor > 0:
            #     reward += 1
            # if num_passenger_to_upper_floor > 0:
            #     reward += 10

            state[which_elevator] += 1
        return state

    def elevatorStop(self, state, action, which_elevator, current_floor):
        # reward = 0
        # Are there any passengers in the elevator that want to leave?
        passengers_left = False
        num_passengers_left = 0
        #print("Passengers in Elevator",self.passengerInElevator(state))
        if self.passengerInElevator(state, which_elevator)[0]:
            passengers_left, num_passengers_left = self.unloadPassenger(
                state, which_elevator, current_floor)
            # reward -= self.passengerInElevator(state)[1]*1
            #print("Passengers left Elevator", passengers_left, num_passengers_left)
            #print("Passengers in Elevator now",self.passengerInElevator(state))
        # Check if there are any people waiting at this floor
        passengers_entered = False
        num_passengers_entered = 0
        num_passenger_left_at_floor = 0
        if self.passengerAtFloor(state, current_floor)[0]:
            #print("Passengers at floor", self.passengerAtFloor(state,current_floor)[1])
            passengers_entered, num_passengers_entered, num_passenger_left_at_floor = self.loadPassenger(
                state, which_elevator, current_floor)
            #print("Passengers entered Elevator", passengers_entered, num_passengers_entered)
            #print("Passengers left at floor", num_passenger_left_at_floor)
            #print("Passengers in Elevator now",self.passengerInElevator(state))

        # if passengers_entered == False and passengers_left == False:
        #     reward = -1000
        #     #print("No one left or entered")

        # reward += num_passengers_left*100
        # reward += num_passengers_entered*10
        # reward -= num_passenger_left_at_floor

        return state, num_passengers_left, num_passengers_entered

    def nextState(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        state = self.state.copy()
        self.reward_3 = [[], []]
        self.punish = False
        # Turns action to a list of actions for each elevator
        actions = self.decodeAction(action, 3)
        #print(action, actions)
        unload_count, load_count = 0, 0
        for i in range(self.elevator_num):
            # update direction
            if actions[i] != 2:
                self.direction[i] = actions[i]

            # get the current floor for this elevator
            current_floor = int(state[i])
            # Decode action for this elevator
            specific_action = actions[i]
            #print("Elevator",i,"goes for action", specific_action)
            if specific_action == 0:
                #print("Elevator",i,"at floor",state[i], "going down")
                state = self.elevatorMoveDown(
                    state, specific_action, i, current_floor)
                # reward += tmp_reward
                #print("Elevator",i,"at floor",state[i], "went down")
            elif specific_action == 1:
                #print("Elevator",i,"at floor",state[i], "going up")
                state = self.elevatorMoveUp(
                    state, specific_action, i, current_floor)
                # reward += tmp_reward
                #print("Elevator",i,"at floor",state[i], "went up")
            elif specific_action == 2:
                state, unload_count, load_count = self.elevatorStop(
                    state, specific_action, i, current_floor)
                # reward += tmp_reward
                #print("Elevator",i,"at floor",state[i], "went stopped")
            #print("Elevator",i,"at floor",state[i])
        reward = self.get_reward(
            unload_count, load_count, self.reward_func, self.unload_reward, self.load_reward, self.discount)

        done = self.step_index >= (self.step_size)
        # if done:
        #     print("ATT:",self.avg_traveling_time, "AWT:", self.avg_waiting_time)

        # Infinite episode so no ending singal

        # if self.waiting_passengers == 0: #! currently no self.waiting_passengers is not used
        #     done = True
        #     reward += 400
        # print(reward)
        return state, reward, done, {}

    # deprecated
    # def eval(self, action):
    #     state, reward, done, obj = self.nextState(action)
    #     return np.array(state), reward, done, obj

    def get_next_passengers(self):
        # get new passengers from the poisson distribution
        new_passengers = self.poisson[self.step_index]
        self.step_index += 1

        # TODO: necessary? for calculating the total waiting time
        # self.total_waiting_time -= self.step_index * new_passengers

        # randomly distribute the current number of new passengers to each floor
        # and for each passenger, generate the destination randomly as well
        # TODO: possible overflow?
        for i in range(new_passengers):

            random_floor = int(self.np_random.uniform(1, self.floor_num + 1))
            random_destination = int(
                self.np_random.uniform(1, self.floor_num + 1))

            while random_floor == random_destination:
                random_destination = int(
                    self.np_random.uniform(1, self.floor_num + 1))

            # floor_index in the state
            floor_index = int(self.elevator_num + (self.elevator_num*self.elevator_limit) + ((random_floor - 1) *
                                                                                             self.floor_limit))
            for k in range(0,  self.floor_limit):
                if self.state[floor_index+k] == 0:
                    self.state[floor_index+k] = random_destination
                    # record the time each passenger showing up
                    self.waiting_time_table[floor_index+k - self.elevator_num - (
                        self.elevator_num*self.elevator_limit)] = self.step_index
                    if random_destination > random_floor:
                        self.new_hall_call[random_floor - 1, 0] = 1
                    elif random_destination < random_floor:
                        self.new_hall_call[random_floor - 1, 1] = 1
                    self.waited += 1
                    break

    def step(self, action):
        state, reward, done, obj = self.nextState(action)
        self.state = state
        self.new_hall_call = np.zeros((self.floor_num, 2))

        self.get_next_passengers()

        # Push first in queue out and add new state as last in queue
        if self.stateQueue.full():
            self.stateQueue.get()
        self.stateQueue.put(self.state.copy())

        # If queue full start checking for oscillation
        if self.stateQueue.full():
            queueList = np.asarray(list(self.stateQueue.queue))
            # if not np.array_equal(queueList[0], queueList[1]):
            #     if np.array_equal(queueList[0], queueList[2]) and np.array_equal(queueList[1], queueList[3]):
            #         reward -= 1000000
        return self.floor_call_mask(), reward, done, obj

    def get_total_waiting_time(self):
        total = 0
        for time in self.waiting_time_table:
            if time >0:
                total += self.step_index - time

        total += self.total_waiting_time
        return total

    def get_total_squared_waiting_time(self):
        total = 0
        for time in self.waiting_time_table:
            if time > 0:
                total += (self.step_index - time)**2

        total += self.total_square_waiting_time
        return total

    def get_total_traveling_time(self):
        total = 0
        for time in self.traveling_time_table:
            if time > 0:
                total += self.step_index - time

        total += self.total_elevator_time
        return total

    def get_total_square_traveling_time(self):
        total = 0
        for time in self.traveling_time_table:
            if time > 0:
                total += (self.step_index - time)**2

        total += self.total_square_elevator_time
        return total

    def reset(self):
        self.total_waiting_time = 0  # sum of every passenger's waiting time
        self.total_square_waiting_time = 0
        self.total_elevator_time = 0  # sum of the time in the elevator for each passenger
        self.total_square_elevator_time = 0

        self.waiting_time_table = []  # floor num * floor limit
        self.traveling_time_table = []  # elevator_num * elevator_limit

        # set here waiting_passengers
        self.state = np.zeros(self.elevator_num + (self.elevator_num *
                              self.elevator_limit) + (self.floor_num * self.floor_limit))

        # print("state 1:", self.state, len(self.state))

        # initial elevator position
        for i in range(self.elevator_num):
            # self.state[i] = self.np_random.randint(1, self.floor_num+1)
            self.state[i] = 1

        # print("state 2:", self.state, len(self.state))

        # Initialize poisson distribution
        self.poisson = self.np_random.poisson(
            lam=self.lam, size=self.step_size+1)
        self.step_index = 0
        self.traveled = 0
        self.waited = 0
        self.arrived = 0
        self.direction = [2] * self.elevator_num
        self.new_hall_call = np.zeros((self.floor_num, 2))
        self.reward_3 = [[], []]
        # Initialize waiting_time_table
        self.waiting_time_table = np.zeros(self.floor_num * self.floor_limit)
        # Initialize
        self.traveling_time_table = np.zeros(
            self.elevator_num * self.elevator_limit)

        self.get_next_passengers()

        self.steps_beyond_done = None
        # self.split_state()
        return self.floor_call_mask()

    def floor_call_mask(self):
        """
        Mask the floor calls with only information of going up or down

        1 means going up

        -1 means going down

        (0 is by default an empty slot)

        Returns:
            masked_state(ndarray): state of the environment with the floor calls masked
        """

        masked_state = self.state.copy()  # deep copy
        masked_floor_call = np.zeros(self.floor_num * 2)
        for i in range(self.floor_num):
            for j in range(self.floor_limit):
                if masked_state[self.floor_start_index + i*self.floor_limit + j]:
                    if masked_state[self.floor_start_index + i*self.floor_limit + j] > i+1:
                        masked_floor_call[i*2] = 1
                    elif masked_state[self.floor_start_index + i*self.floor_limit + j] < i+1:
                        masked_floor_call[i*2 + 1] = 1
                    if (masked_floor_call[i*2] and masked_floor_call[i*2 + 1]):
                        break
        masked_state = np.concatenate(
            (masked_state[:self.floor_start_index], masked_floor_call))
        return masked_state

    @property
    def avg_waiting_time(self):
        return self.get_total_waiting_time()/(self.waited + 0.000000000001)
        # return self.total_waiting_time/sum(self.poisson[:self.step_index])

    @property
    def avg_traveling_time(self):
        return self.get_total_traveling_time()/(self.traveled + 0.000000000001)

    def get_reward(self, unload_count, load_count, reward_func=1, unload_reward=None, load_reward=None, discount=None):
        reward = 0

        def current_waiting_time_sq(step):
            return (self.step_index - step)**2

        def current_waiting_time(step):
            return (self.step_index - step)

        unload_reward = unload_reward if unload_reward else self.floor_num ** 2
        load_reward = load_reward if load_reward else self.floor_num ** 2
        if reward_func == 1:
            # Average squared waiting time reward in https://ieeexplore.ieee.org/document/8998335
            # \[ r_t = \sum_{i=0}^{k_h}(-t^2_{h_i}+\theta_{h_{i}}) + \sum_{j=0}^{k_c}(-t^2_{c_j}+\theta_{c_{j}})\]
            reward += unload_reward * unload_count + load_reward * load_count
            reward -= sum(map(current_waiting_time_sq,
                          self.waiting_time_table))
            reward -= sum(map(current_waiting_time_sq,
                          self.traveling_time_table))
            # reward = abs(1e10/reward)
        elif reward_func == 2:
            # Average waiting time reward
            reward += unload_reward * unload_count + load_reward * load_count
            reward -= sum(map(current_waiting_time, self.waiting_time_table))
            reward -= sum(map(current_waiting_time, self.traveling_time_table))
            # reward = abs(1e6/reward)

        elif reward_func == 3:
            discount = discount or 0.99
            if len(self.reward_3[0]) > 0:
                # print("A")
                for i in self.reward_3[0]:
                    reward += load_reward * (discount ** i)
            if len(self.reward_3[1]) > 0:
                # print("B")
                for i in self.reward_3[1]:
                    reward += unload_reward * (discount ** i)
            if self.punish:
                reward = self.punish_reward
            # if reward != -100: print("A")
            # reward -= sum(self.waiting_time_table) + sum(self.traveling_time_table)

        elif reward_func == 4:
            reward = unload_count * 10000 + load_count * 1000 + 1
        return reward
    # region rendering-related methods & Test Methods

    def render(self, episode=None, step=None, mode='human'):
        from gym.envs.classic_control import rendering
        self.screen_width = 640
        self.screen_height = 480

        if self.viewer is None:

            self.viewer = rendering.Viewer(self.screen_width,
                                           self.screen_height)
        self.transform = rendering.Transform()

        self.floor_padding = (self.screen_height - 100) / self.floor_num
        boxwidth = self.floor_padding / 1.5
        boxheight = self.floor_padding
        l, r, t, b = -boxwidth / 2, boxwidth / 2, boxheight - boxwidth / 2, -boxwidth / 2
        box = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        self.boxtrans = rendering.Transform(
            (self.screen_width / 2 + boxwidth,
                (self.state[0] * self.floor_padding - 30) + 40))
        box.add_attr(self.boxtrans)
        box.set_color(.4, .4, .4)

        for i in range(self.floor_num):
            start = self.floor_num * (i + 1) + self.elevator_limit
            stop = start + self.floor_limit

        self.viewer.add_geom(box)

        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        self.score_label = pyglet.text.Label('HELLO WORLD',
                                             font_size=36,
                                             x=20,
                                             y=self.screen_height * 2.5 /
                                             40.00,
                                             anchor_x='left',
                                             anchor_y='center',
                                             color=(0, 0, 0, 255))
        pixel_scale = 1
        if hasattr(win.context, '_nscontext'):
            pixel_scale = win.context._nscontext.view().backingScaleFactor(
            )  # pylint: disable=protected-access
        VP_W = int(pixel_scale * self.screen_width)
        VP_H = int(pixel_scale * self.screen_height)

        gl.glViewport(0, 0, VP_W, VP_H)

        t.enable()
        self.render_floors()
        self.render_indicators(self.screen_width, self.screen_height)
        self.render_elevators()
        if episode != None and step != None:
            self.render_info(episode, step)
        t.disable()

        win.flip()
        return self.viewer.isopen

    def render_info(self, episode, step):
        info_label = pyglet.text.Label('episode: ' +
                                       str(episode) + ' step: ' + str(step),
                                       font_size=14,
                                       x=10,
                                       y=10,
                                       anchor_x='left',
                                       anchor_y='center',
                                       color=(0, 0, 0, 255))
        info_label.draw()

    def render_floors(self):
        PLAYFIELD = 2000
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1, 1, 1, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)

        # increase range by one to add line on the top
        for floor in range(self.floor_num + 1):
            gl.glColor4f(0, 0, 0, 1)
            gl.glVertex3f(self.screen_width, 50 + self.floor_padding * floor,
                          0)
            gl.glVertex3f(self.screen_width,
                          50 + self.floor_padding * floor + 1, 0)
            gl.glVertex3f(self.screen_width / 2,
                          50 + self.floor_padding * floor + 1, 0)
            gl.glVertex3f(self.screen_width / 2,
                          50 + self.floor_padding * floor, 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1, 1, 1, 1)
        gl.glVertex3f(W / 2, 0, 0)
        gl.glVertex3f(W / 2, H, 0)
        gl.glVertex3f(0, H, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()

        for floor in range(self.floor_num):
            position_x = 20
            position_y = 50 + (self.floor_padding) * \
                floor + (self.floor_padding/2)

            # start = floor * self.floor_limit + self.elevator_limit + 1
            # stop = start + self.floor_limit
            # waiting_passengers_floor = 0
            # for i in range(int(start), int(stop)):
            #     if self.state[i] > 0:
            #         waiting_passengers_floor += 1

            waiting_passengers_floor = self.passengerAtFloor(
                self.state, floor+1)[1]

            score_label = pyglet.text.Label('F' + str(floor + 1) + ', Queue: ' +
                                            str(waiting_passengers_floor),
                                            font_size=14,
                                            x=position_x,
                                            y=position_y,
                                            anchor_x='left',
                                            anchor_y='center',
                                            color=(0, 0, 0, 255))
            score_label.draw()

    def render_elevators(self):
        elevator_width = 60
        if self.elevator_num > 5:
            elevator_width = 300 // self.elevator_num
        gl.glBegin(gl.GL_QUADS)
        for i in range(self.elevator_num):
            current_floor = self.state[i] - 1
            gl.glColor4f(0.3, 0.3, 0.3, 1)
            gl.glVertex3f(self.screen_width / 2 + (elevator_width*(i+1)) + (i*10),
                          50 + self.floor_padding * current_floor, 0)
            gl.glVertex3f(
                self.screen_width / 2 + (elevator_width*(i+1)) + (i*10),
                50 + self.floor_padding * current_floor + self.floor_padding,
                0)
            gl.glVertex3f(
                self.screen_width / 2 + (elevator_width*i) + (i*10),
                50 + self.floor_padding * current_floor + self.floor_padding,
                0)
            gl.glVertex3f(self.screen_width / 2 + (elevator_width*i) + (i*10),
                          50 + self.floor_padding * current_floor, 0)
        gl.glEnd()

    def close(self):
        """
        Stop rendering
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def decodeAction(self, n, base=3):
        """
        Convert a decimal number to a list of actions

        Example: decodeAction(9, 3) = [0, 0, 1]

        0 = going down

        1 = going up

        2 = idling

        Args:
            n (int): encoded actions, a decimal number
            base (int, optional): How many possible actions one elevator has. Defaults to 3.

        Returns:
            (list): a list of actions for every elevator
        """
        actions = [2] * self.elevator_num
        for i in range(self.elevator_num):
            actions[i] = n % base
            n = n//base
        return actions

    def encodeAction(self, l, base=3):
        """
        Convert a list of actions to a decimal number 

        Example: encodeAction([0, 0, 1], 3) = 9

        0 = going down

        1 = going up

        2 = idling

        Args:
            l (list): a list of actions for every elevator
            base (int, optional): How many possible actions one elevator has. Defaults to 3.

        Returns:
            (int): encoded actions, a decimal number
        """
        toReturn = 0
        for i in range(len(l)):
            toReturn += (base**i)*l[i]
        return toReturn

    def act_render(self):
        """
        test function for manual rendering
        """
        while True:
            action = int(input())
            if action == -1:
                self.close()
                return
            self.step(action)
            self.render()
            print(self.decodeAction(action, 3))
            self.split_state()

    def split_state(self, masked_view=False, verbose=True):
        if masked_view:
            raw_state = self.floor_call_mask()
        else:
            raw_state = self.state
        print("Elevator Floor:", raw_state[:self.elevator_num])
        # if self.elevator_num <= 3:
        #     verbose = True
        if verbose:
            for i in range(self.elevator_num):
                print("Elevator", i, "Passengers:", raw_state[self.elevator_num + i *
                      self.elevator_limit: self.elevator_num + (i+1)*self.elevator_limit])
            if masked_view:
                for i in range(self.floor_num):
                    print("Floor", i+1, "Calls: ", end="")
                    if raw_state[self.floor_start_index+i*2]:
                        print("Up", end=" ")
                    else:
                        print("   ", end="")
                    if raw_state[self.floor_start_index+i*2+1]:
                        print("Down", end="")
                    print()
            else:
                for i in range(self.floor_num):
                    print("Floor", i+1, "Passengers:", raw_state[self.elevator_num + self.elevator_num * self.elevator_limit +
                          i*self.floor_limit:self.elevator_num + self.elevator_num * self.elevator_limit + (i+1)*self.floor_limit])
        else:
            print("Elevator Passengers:",
                  raw_state[self.elevator_num: self.elevator_num + self.elevator_num * self.elevator_limit])
            print("Floor Passengers:",
                  raw_state[self.elevator_num + self.elevator_num * self.elevator_limit:])
# endregion


if __name__ == "__main__":
    gym.register(
        id='Elevator-v0',
        entry_point='environment:ElevatorEnv',
        max_episode_steps=1000,
        kwargs={'elevator_num': 3, 'elevator_limit': 10, 'floor_num': 4, 'floor_limit': 20,
                'step_size': 1000, 'poisson_lambda': 50, 'seed': 1, "reward_func": 1, "unload_reward": None, "load_reward": None},
    )
    env = gym.make('Elevator-v0')
    done = False
    env.reset()
    while env.step_index < 1000:
        action = np.random.randint(0, env.action_space.n)
        _, reward, done, _ = env.step(action)
        # print(reward)
        # print(done)
        env.render()
    """
    state (np.array): state of the environment, 
                [floor of each elevator (elevator_num), 3
                each passenger's dest inside each elevator (elevator_num * elevator_limit), 30
                each passenger's dest on each floor (floor_num * floor_limit)] 20


    self.state = np.zeros(self.elevator_num + (self.elevator_num *
                              self.elevator_limit) + (self.floor_num * self.floor_limit))
    """
