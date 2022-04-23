import os
import random

import numpy as np
import gym
from gym.utils import seeding
from z3 import *

from mjrl.envs.maze_env import MazeEnv
from mjrl.envs.ant_maze_env import AntMazeEnv
from mjrl.envs.spec_env import SpecEnv
from mjrl.envs.ant_hrl import AntMaze, AntPush, AntFall

MOVE = 2

def assert_maze(solver, maze, structure):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            solver.add(maze[i][j] == structure[i][j])


def encode(s, i, xpre, ypre, maze, w, l, straight=False):
    x = Int('x'+str(i))
    y = Int('y'+str(i))

    # robot move along 8 directions on the abstract structure.
    dx = Int('dx'+str(i))
    dy = Int('dy'+str(i))
    s.add(x == xpre + dx, y == ypre + dy, dx <= 1, dx >= -1, dy <= 1, dy >= -1)
    if straight:
        s.add(Or(dx == 0, dy == 0))

    # robot avoids any obstacles or falling down
    s.add(maze[x][y] != 1, maze[x][y] != -1)
    # the path to the next loc must be on a clear path.
    s.add(Implies(And(dx != 0, dy != 0), Or(maze[xpre][ypre+dy] == 0, maze[xpre+dx][ypre] == 0)))
    # the move must be in range.
    s.add(x >= 0, x < w, y >= 0, y < l)
    # move the movable block and the robot can only afford to move 1 block.
    s.add(Implies(maze[x][y] == MOVE, Or (maze[x+dx][y+dy] == 0, maze[x+dx][y+dy] == -1)))

    # update the maze if a movable block is moved.
    md = Int('md'+str(i))
    s.add(Implies(maze[x+dx][y+dy] == -1, md == 0))
    s.add(Implies(maze[x+dx][y+dy] ==  0, md == MOVE))
    a = Store(maze, x, Store(maze[x], y, 0))
    a = Store(a, x+dx, Store(a[x+dx], y+dy, md))
    new_maze = Array('maze'+str(i), IntSort(), ArraySort(IntSort(), IntSort()))
    s.add(If (maze[x][y] == MOVE, new_maze == a, new_maze == maze))

    return x, y, new_maze


def bmc(x0, y0, xf, yf, structure, silent=True, straight=False):
    assert(x0 != xf or y0 != yf)

    s = Solver()

    maze = Array('maze', IntSort(), ArraySort(IntSort(), IntSort()))
    assert_maze(s, maze, structure)

    reached = False
    x = Int('x')
    y = Int('y')

    s.add(x == x0, y == y0)
    i = 1
    positions = [(x, y)]
    mazes = [maze]

    while not reached:
        x, y, maze = encode(s, i, x, y, maze, len(structure), len(structure[0]), straight=straight)
        positions.append((x, y))
        mazes.append(maze)
        s.push()
        s.add(x == xf, y == yf)
        reached = (s.check() == sat)
        if reached:
            model = s.model()
        s.pop()
        i += 1

        if i >= len(structure) * len(structure[0]):
            print ('No path found and the constraint is encoded as follows.')
            print (s)
            return

    if not silent:
        print ('[Solution Path]')

    for j in range(i):
        if not silent:
            print (f'x[{j}] = {model[positions[j][0]]} and y[{j}] = {model[positions[j][1]]}')
        positions[j] = ((model[positions[j][0]].as_long(), model[positions[j][1]].as_long()))
        mazes[j] = [ [ model.evaluate(mazes[j][s][t]).as_long() for t in range(len(structure[0])) ] for s in range(len(structure)) ]

    if straight and i > 2:
        for j in reversed(range(1,i-1)):
            if positions[j-1][0] == positions[j+1][0] or positions[j-1][1] == positions[j+1][1]:
                del positions[j]

    return positions, mazes


def reach(goal, err):
    def predicate(state):
        return (min([state[0] - goal[0],
                     goal[0] - state[0],
                     state[1] - goal[1],
                     goal[1] - state[1]]) + err)
    return predicate


def avoid(obstacles):
    def predicate(state):
        dist = 0.
        for obstacle in obstacles:
            dist += min(max([obstacle[0] - state[0],
                    obstacle[1] - state[1],
                    state[0] - obstacle[2],
                    state[1] - obstacle[3]]), 0.)
        return dist
    return predicate


class SpecAntMaze(gym.Env, SpecEnv):

    def __init__(self, silent=True):

        # original
        # structure = [
        #     [1, 1, 1, 1, 1],
        #     [1, 'r', 0, 0, 1],
        #     [1, 0, 1, 1, 1],
        #     [1, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1],
        # ]

        # flipped
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

        scale = 8
        offset_x = -12
        offset_y = -12

        x0, y0 = 3, 1
        xf, yf = 1, 1

        if not silent:
            print ('[Maze]')
            print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in structure]))
        # generate the high level plan
        positions, mazes = bmc(x0, y0, xf, yf, structure, silent=silent, straight=True)

        # initial ant position
        px, py = positions[0][1], (len(structure)-1) - positions[0][0]
        self.ix, self.iy = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y
        if not silent:
            print (f'ant initial position {self.ix}, {self.iy}')

        specs = []
        centerx_list = [0.0]
        centery_list = [0.0]

        k = 0
        for position in positions[1:]:
            # ant positions to avoid
            obstacles = []
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if (mazes[k][i][j] == 1 or mazes[k][i][j] == -1 or (mazes[k][i][j] == MOVE and position != (i, j))):
                        obstacles.append(np.array([j * scale + offset_x, ((len(structure)-1)-i) * scale + offset_y, (j+1)*scale + offset_x, (((len(structure)-1)-i)+1)*scale + offset_y]))

            # ant positions to reach
            px, py = position[1], (len(structure)-1) -  position[0]
            centerx, centery = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y

            centerx_list.append(centerx)
            centery_list.append(centery)
            specs.append({'reach': reach(np.array([centerx, centery]), 1), 'points':np.array([[centerx_list[k], centery_list[k]], [centerx, centery]])})

            k += 1

        if not silent:
            for i in range(len(centerx_list)):
                if i != 0:
                    print (f'ant needs to reach {centerx_list[i]}, {centery_list[i]}')

        SpecEnv.__init__(self, specs)

        self.hrl_env = AntMaze()

    @property
    def observation_space(self):
        return self.hrl_env.observation_space

    @property
    def action_space(self):
        return self.hrl_env.action_space

    def step(self, a):
        ob, _, _, info = self.hrl_env.step(a)
        reward, done = self.reward_func(ob)

        # normalized distance
        distance = np.linalg.norm(ob[:2] - np.array([0, 16]))
        info['distance'] = (distance - 1) / (16 - 1) if distance > 1.0 else 0.0
        info['finished'] = done

        return ob, reward, done, info

    def reset(self):
        ob = self.hrl_env.reset()
        return self.reset_func(ob)

    def seed(self, s):
        self.hrl_env.maze_env.wrapped_env.seed(s)


class SpecAntFall(gym.Env, SpecEnv):

    def __init__(self, silent=True):

        # original
        # structure = [
        #     [1,  1,    1,  1],
        #     [1,  'r',    0,  1],
        #     [1,  0, Move.YZ,  1],
        #     [1, -1,   -1,  1],
        #     [1,  0,    0,  1],
        #     [1,  1,    1,  1],
        # ]

        # flipped
        structure = [
            [1,  1,    1,  1],
            [1,  0,    0,  1],
            [1, -1,   -1,  1],
            [1,  0, MOVE,  1],
            [1,  0,    0,  1],
            [1,  1,    1,  1],
        ]

        scale = 8
        offset_x = -12
        offset_y = -12

        x0, y0 = 4, 1
        xf, yf = 1, 1

        if not silent:
            print ('[Maze]')
            print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in structure]))

        # generate the high level plan
        positions, mazes = bmc(x0, y0, xf, yf, structure, silent=silent)

        # initial ant position
        px, py = positions[0][1], (len(structure)-1) - positions[0][0]
        self.ix, self.iy = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y
        if not silent:
            print (f'ant initial position {self.ix}, {self.iy}')

        specs = []
        centerx_list = [0.0]
        centery_list = [0.0]

        k = 0
        for position in positions[1:]:
            # ant positions to avoid
            obstacles = []
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if (mazes[k][i][j] == -1):
                        obstacles.append(np.array([j * scale + offset_x, ((len(structure)-1)-i) * scale + offset_y, (j+1)*scale + offset_x, (((len(structure)-1)-i)+1)*scale + offset_y]))

            # ant positions to reach
            px, py = position[1], (len(structure)-1) - position[0]
            # see if moving towrads into a moving box and if the moving box would fall into a gap.
            if k + 2 < len(positions) and mazes[k][position[0]][position[1]] == MOVE and mazes[k][positions[k+2][0]][positions[k+2][1]] == -1:
                # move the box all the way to the end
                move_along_y_axis = (positions[k+2][0] - positions[k][0]) != 0
                move_along_x_axis = (positions[k+2][1] - positions[k][1]) != 0
                centerx, centery = (px + 0.9) * scale + offset_x if move_along_x_axis else (px + 0.5) * scale + offset_x, \
                                (py + 0.9) * scale + offset_y if move_along_y_axis else (py + 0.5) * scale + offset_y
            else:
                centerx, centery = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y

            dim = 0.5
            # to match other work
            if k == 3:
                centerx, centery, dim = 0, 27.0, 1

            # under construction
            centerx_list.append(centerx)
            centery_list.append(centery)
            specs.append({'reach': reach(np.array([centerx, centery]), dim), 'points':np.array([[centerx_list[k], centery_list[k]], [centerx, centery]])})
            k += 1

        if not silent:
            for i in range(len(centerx_list)):
                if i != 0:
                    print (f'========== Phase {i} ===========')
                    print (f'ant needs to reach {centerx_list[i]}, {centery_list[i]}')

        SpecEnv.__init__(self, specs)

        self.hrl_env = AntFall()

    @property
    def observation_space(self):
        return self.hrl_env.observation_space

    @property
    def action_space(self):
        return self.hrl_env.action_space

    def step(self, a):

        ob, _, _, info = self.hrl_env.step(a)
        reward, done = self.reward_func(ob)

        # normalized distance
        distance = np.linalg.norm(ob[:2] - np.array([0, 27]))
        info['distance'] = (distance - 1) / (27 - 1) if distance > 1.0 else 0.0
        info['finished'] = done

        return ob, reward, done, info

    def reset(self):
        ob = self.hrl_env.reset()
        return self.reset_func(ob)

    def seed(self, s):
        self.hrl_env.maze_env.wrapped_env.seed(s)


class SpecAntPush(gym.Env, SpecEnv):

    def __init__(self, silent=True):

        # original
        # structure = [
        #     [1, 1,  1,  1,   1],
        #     [1, 0, 'r', 1,   1],
        #     [1, 0,  Move.XY, 0, 1],
        #     [1, 1,  0,  1,   1],
        #     [1, 1,  1,  1,   1],
        # ]

        # flipped
        structure = [
            [1, 1,    1, 1, 1],
            [1, 1,    0, 1, 1],
            [1, 0, MOVE, 0, 1],
            [1, 0,    0, 1, 1],
            [1, 1,    1, 1, 1],
        ]

        scale = 8
        offset_x = -20
        offset_y = -12

        x0, y0 = 3, 2
        xf, yf = 1, 2

        if not silent:
            print ('[Maze]')
            print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in structure]))

        # generate the high level plan
        positions, mazes = bmc(x0, y0, xf, yf, structure, silent=silent)

        # initial ant position
        px, py = positions[0][1], (len(structure)-1) - positions[0][0]
        self.ix, self.iy = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y
        if not silent:
            print (f'ant initial position {self.ix}, {self.iy}')

        specs = []
        centerx_list = [0.0]
        centery_list = [0.0]

        k = 0
        for position in positions[1:]:
            # ant positions to avoid
            obstacles = []
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if (mazes[k][i][j] == 1 or mazes[k][i][j] == -1 or (mazes[k][i][j] == MOVE and position != (i, j))):
                        obstacles.append(np.array([j * scale + offset_x, ((len(structure)-1)-i) * scale + offset_y, (j+1)*scale + offset_x, (((len(structure)-1)-i)+1)*scale + offset_y]))

            # ant positions to reach
            px, py = position[1], (len(structure)-1) -  position[0]
            centerx, centery = (px + 0.5) * scale + offset_x, (py + 0.5) * scale + offset_y

            # to match other work
            dim = 0.5
            if k == 2:
                centerx, centery, dim = 0, 18, 1

            # under construction
            centerx_list.append(centerx)
            centery_list.append(centery)
            specs.append({'reach': reach(np.array([centerx, centery]), dim), 'points':np.array([[centerx_list[k], centery_list[k]], [centerx, centery]])})
            k += 1

        if not silent:
            for i in range(len(centerx_list)):
                if i != 0:
                    print (f'========== Phase {i} ===========')
                    print (f'ant needs to reach {centerx_list[i]}, {centery_list[i]}')

        SpecEnv.__init__(self, specs)

        self.hrl_env = AntPush()

    @property
    def observation_space(self):
        return self.hrl_env.observation_space

    @property
    def action_space(self):
        return self.hrl_env.action_space

    def step(self, a):
        ob, _, _, info = self.hrl_env.step(a)
        reward, done = self.reward_func(ob)

        # normalized distance
        distance = np.linalg.norm(ob[:2] - np.array([0, 18]))
        info['distance'] = (distance - 1) / (18 - 1) if distance > 1.0 else 0.0
        info['finished'] = done

        return ob, reward, done, info

    def reset(self):
        ob = self.hrl_env.reset()
        return self.reset_func(ob)

    def seed(self, s):
        self.hrl_env.maze_env.wrapped_env.seed(s)
