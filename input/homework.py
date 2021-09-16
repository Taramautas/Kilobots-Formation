import queue

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import BaseScheduler
from enum import Enum

# Tip: Lookup https://github.com/projectmesa/mesa/tree/main/mesa
# to check the available methods for ContinousSpace, Agent, and Model

IMAGE = 'star_small.png'

# You can modify ITERATIONS+KILOBOT_N if desired
ITERATIONS = 160000

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Frequency of redraw in matplotlib
# Higher values result faster simulation
# Increase to higher values (e.g., 100) later on
# Lower values for debugging
UPDATE_FRAME = 100

KILOBOT_N = 100
KILOBOT_RADIUS = 1.5

GRADIENT_MAX = 999
DISTANCE_MAX = 999
DESIRED_DISTANCE = 2 * KILOBOT_RADIUS

yield_distance = 8 * KILOBOT_RADIUS

startup_time = 10

# You can increase velocity, but be careful to keep in range of other kilobots
VELOCITY = 0.1

# neighborhood distance
g = 3 * KILOBOT_RADIUS


# Can be used for state-machine
class State(Enum):
    START = 1
    WAIT_TO_MOVE = 2
    MOVE_WHILE_OUTSIDE = 3
    MOVE_WHILE_INSIDE = 4
    JOINED_SHAPE = 5


def euklidic_distance(x, x_goal, y, y_goal):
    return math.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)


class KilobotAgent(Agent):
    def __init__(self, unique_id, model, gradient, seed, phi=None, init_wait=25):
        super().__init__(unique_id, model)
        self.gradient = gradient
        self.seed = seed
        self.phi = phi
        # TODO some status variables if desired
        self.prev = DISTANCE_MAX
        self.state = State.START
        self.timer = 0
        self.motion = 0     # 0 = stop, 1 = edge-follow
        self.last_cor = self.pos

    def step(self):
        # TODO state-machine
        # print("movement")
        self.gradient_formation()
        if self.state is State.START:
            # wait fixed amount of time so all robots are on
            if self.seed:
                self.state = State.JOINED_SHAPE
            else:
                self.gradient_formation()
                self.timer += 1
                if self.timer > startup_time:
                    self.state = State.WAIT_TO_MOVE
        elif self.state is State.WAIT_TO_MOVE:
            # print("self.state is State.WAIT_TO_MOVE")
            # check if there is a moving neighbor
            neighborhood = model.space.get_neighbors(self.pos, g, False)
            moving_neighbor = False
            for nb in neighborhood:
                if nb.state is State.MOVE_WHILE_OUTSIDE or nb.state is State.MOVE_WHILE_INSIDE:
                    moving_neighbor = True

            if not moving_neighbor:
                # find highest gradient among neighbors
                h = 0
                neighbors_same_gradient = []
                for nb in neighborhood:
                    if h < nb.gradient:
                        h = nb.gradient
                    if nb.gradient == self.gradient:
                        neighbors_same_gradient.append(nb)
                if self.gradient > h:
                    self.state = State.MOVE_WHILE_OUTSIDE
                    self.motion = 1
                elif self.gradient == h:
                    # check if self.id > ids of all neighbors
                    id_greater = True
                    for nsg in neighbors_same_gradient:
                        if nsg.unique_id > self.unique_id:
                            id_greater = False
                    if id_greater:
                        self.state = State.MOVE_WHILE_OUTSIDE
                        self.motion = 1
        elif self.state is State.MOVE_WHILE_OUTSIDE:
            # print("self.state is State.MOVE_WHILE_OUTSIDE")
            # check if we are inside the shape
            array_as_list = self.model.img.tolist()
            inside_shape = array_as_list[int(self.pos[0])][int(self.pos[1])][0]
            if inside_shape != 1.0:
                self.state = State.MOVE_WHILE_INSIDE
                self.motion = 1
            # find out if there is a edge-following robot thats too close
            neighborhood_yield_distance = model.space.get_neighbors(self.pos, yield_distance, False)
            if len(neighborhood_yield_distance) == 0:
                return
            for nbyd in neighborhood_yield_distance:
                # check if edge_following robot is coming towards us
                if nbyd.motion == 1 and euklidic_distance(self.pos[0], nbyd.last_cor[0], self.pos[1], nbyd.last_cor[1]) < euklidic_distance(self.pos[0], nbyd.pos[0], self.pos[1], nbyd.pos[1]):
                    self.state = State.WAIT_TO_MOVE
                    self.motion = 0
            if self.state is not State.WAIT_TO_MOVE:
                self.edge_following()
                self.motion = 1

        elif self.state is State.MOVE_WHILE_INSIDE:
            # print("self.state is State.MOVE_WHILE_INSIDE")
            array_as_list = self.model.img.tolist()
            inside_shape = array_as_list[int(self.pos[0])][int(self.pos[1])][0]
            if inside_shape == 1.0:
                print("inside_shape == 1.0")
                self.state = State.JOINED_SHAPE

            # find closest neighbor
            neighborhood = model.space.get_neighbors(self.pos, g, False)
            if len(neighborhood) == 0:
                return
            closest_dist = DISTANCE_MAX
            closest_neighbor = None
            for nb in neighborhood:
                if euklidic_distance(self.pos[0], nb.pos[0], self.pos[1], nb.pos[1]) < closest_dist:
                    closest_dist = euklidic_distance(self.pos[0], nb.pos[0], self.pos[1], nb.pos[1])
                    closest_neighbor = nb
            if self.gradient <= closest_neighbor.gradient:
                self.state = State.JOINED_SHAPE
            # find out if there is a edge-following robot thats too close
            neighborhood_yield_distance = model.space.get_neighbors(self.pos, yield_distance, False)
            for nbyd in neighborhood_yield_distance:
                # check if edge_following robot is coming towards us
                if nbyd.motion == 1 and euklidic_distance(self.pos[0], nbyd.last_cor[0], self.pos[1],
                                                          nbyd.last_cor[1]) < euklidic_distance(self.pos[0],
                                                                                                nbyd.pos[0],
                                                                                                self.pos[1],
                                                                                                nbyd.pos[1]):
                    self.state = State.WAIT_TO_MOVE
                    self.motion = 0
            if self.state is not State.WAIT_TO_MOVE:
                self.edge_following()
                self.motion = 1
        elif self.state is State.JOINED_SHAPE:
            self.motion = 0

    def gradient_formation(self):
        if self.seed:
            return
        neighborhood = model.space.get_neighbors(self.pos, g, False)
        if len(neighborhood) == 0:
            self.gradient = 0
        else:
            min_grad_neigh = neighborhood[0]
            for n in neighborhood:
                if n.gradient < min_grad_neigh.gradient:
                    min_grad_neigh = n
            self.gradient = min_grad_neigh.gradient + 1

    def edge_following(self):
        # print("edge_following")
        change_angle = 5

        current = DISTANCE_MAX
        neighbors = model.space.get_neighbors(self.pos, g, False)
        for n in neighbors:
            measured_distance = euklidic_distance(self.pos[0], n.pos[0], self.pos[1], n.pos[1])
            if measured_distance < current:
                current = measured_distance
        if current < DESIRED_DISTANCE:
            if self.prev < current:
                # move straight forward
                pass
            else:
                # move forward and counterclockwise
                x2 = math.cos(0 - change_angle) * self.phi[0] - math.sin(0 - change_angle) * self.phi[1]
                y2 = math.sin(0 - change_angle) * self.phi[0] + math.cos(0 - change_angle) * self.phi[1]
                self.phi = (x2, y2)
        else:
            if self.prev > current:
                # move straight forward
                pass
            else:
                # move forward and clockwise
                x2 = math.cos(change_angle) * self.phi[0] - math.sin(change_angle) * self.phi[1]
                y2 = math.sin(change_angle) * self.phi[0] + math.cos(change_angle) * self.phi[1]
                self.phi = (x2, y2)

        self.last_cor = self.pos
        new_x = self.pos[0] + self.phi[0] * VELOCITY
        new_y = self.pos[1] + self.phi[1] * VELOCITY
        model.space.move_agent(self, (new_x, new_y))
        self.prev = current


class ShapeModel(Model):
    def __init__(self, N, width, height, img):
        self.num_agents = N
        self.space = ContinuousSpace(width, height, torus=True)
        self.schedule = BaseScheduler(self)
        self.agents = []
        # !!!! Each Kilobot can access img (=Map) via self.model.img !!!!
        self.img = img
        # Start Position of Agent 0
        self.y = 20
        self.x = 20 - KILOBOT_RADIUS

    def create_kilobot(self, unique_id, gradient, x, y, seed=False):
        """
        Method for adding Kilobots
        --------------------------
        unique_id: unique identifier for Kilobot
        x: x-coordinates on map
        y: y-coordinates on map
        seed: True if Kilobot is Seed-Robot
        """
        a = KilobotAgent(unique_id, self, gradient, seed, phi=(0, -1))
        self.agents.append(a)
        self.schedule.add(a)
        self.space.place_agent(a, (x, y))
        a.last_cor = a.pos

    def step(self):
        self.schedule.step()

    def seed_kilobots(self):
        # TODO Place the seed robots
        # Tip: Vertical Distance between robots is: 
        # -> 2 * KILOBOT_RADIUS * math.sqrt(2/3)
        # Vertical Distance
        d = 2 * KILOBOT_RADIUS * math.sqrt(2 / 3)

        # Root
        self.create_kilobot(len(self.agents), 0, self.x, self.y, True)

        # Layer 1
        self.create_kilobot(len(self.agents), 1, self.x + 2 * KILOBOT_RADIUS, self.y, True)
        self.create_kilobot(len(self.agents), 1, self.x + KILOBOT_RADIUS, self.y + d, True)
        self.create_kilobot(len(self.agents), 1, self.x + KILOBOT_RADIUS, self.y - d, True)

    def pack_kilobots(self):
        # TODO Place the remaining robots
        # Vertical Distance
        d = 2 * KILOBOT_RADIUS * math.sqrt(2 / 3)

        # Layer 2
        self.create_kilobot(len(self.agents), 2, self.x, self.y - 2 * d)
        self.create_kilobot(len(self.agents), 2, self.x + 2 * KILOBOT_RADIUS, self.y - 2 * d)

        # Layer 3
        self.create_kilobot(len(self.agents), 3, self.x - KILOBOT_RADIUS, self.y - 3 * d)
        self.create_kilobot(len(self.agents), 3, self.x + KILOBOT_RADIUS, self.y - 3 * d)
        self.create_kilobot(len(self.agents), 3, self.x + 3 * KILOBOT_RADIUS, self.y - 3 * d)

        gradient = 4

        for i in range(3):

            if gradient % 2 == 0:
                for j in range(gradient - 1):
                    self.create_kilobot(len(self.agents), gradient, self.x + ((2 * j) - i) * KILOBOT_RADIUS,
                                        self.y - gradient * d)
                self.create_kilobot(len(self.agents), gradient, self.x + KILOBOT_RADIUS * (gradient + 1),
                                    self.y - (gradient - 1) * d)

            elif gradient % 2 == 1:
                for j in range(gradient - 1):
                    self.create_kilobot(len(self.agents), gradient, self.x + (2 * j - i) * KILOBOT_RADIUS,
                                        self.y - gradient * d)
                self.create_kilobot(len(self.agents), gradient, self.x + KILOBOT_RADIUS * (gradient + 1),
                                    self.y - (gradient - 1) * d)

            gradient += 1


# Visualize Map
def update_map(img, model, ax):
    ax.clear()
    plt.imshow(img.transpose(1, 0, 2), origin='lower')
    for a in model.agents:
        plt.gca().add_patch(plt.Circle((a.pos[0], a.pos[1]), KILOBOT_RADIUS))
        plt.text(a.pos[0] - KILOBOT_RADIUS / 2, a.pos[1] - KILOBOT_RADIUS / 2,
                 a.gradient, fontsize=10, color='red')


fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(10)

# The shape-bitmap
img = mpimg.imread(IMAGE).transpose(1, 0, 2)

# Create Mesa-model
model = ShapeModel(KILOBOT_N, 100, 100, img)
# Place Kilobots
model.seed_kilobots()
model.pack_kilobots()

# model.create_kilobot(300, 1, 19, 19, seed=False)
# array_as_list = model.img.tolist()
# my_agent = model.agents[0]
# print(my_agent.unique_id)
# inside_shape = array_as_list[int(my_agent.pos[0])][int(my_agent.pos[1])][0]
# if inside_shape == 1.0:
#     print("not inside")
# else:
#     print("inside")

# Run simulation
for i in range(ITERATIONS):
    model.step()
    # agent_queue.get().step()
    # model.agents[-7].edge_following()
    if i % UPDATE_FRAME == 0:
        update_map(img, model, ax)
        plt.pause(0.5)
    if i == 159999:
        print("End")
    if i % 100 == 0:
        for a in model.agents:
            print(a.state)
plt.show()
