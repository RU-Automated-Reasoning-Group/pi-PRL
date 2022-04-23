import sys
import random
import dill as pickle
import getopt
import os

os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import print_planning
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_prog import ProgPolicy
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.trpo import TRPO
from mjrl.utils.train_agent import train_agent


def parse_command_line_options():
    options, remainder = getopt.getopt(sys.argv[1:], 'e:s:d:')
    
    # default setting
    domain = 'ant_maze'
    seed = 123
    folder = 'data'

    for option, arg in options:
        if option in ('-e'):
            env = int(arg)
            if env == 0:
                domain = 'ant_maze'
            elif env == 1:
                domain = 'ant_push'
            elif env == 2:
                domain = 'ant_fall'
            else:
                print('please check your domain')
                exit() 
        if option in ('-s'):
            seed = int(arg)
        if option in ('-d'):
            folder = arg
    
    if not folder.startswith('/'):
        base = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base, folder)

    if not os.path.exists(folder):
        os.makedirs(folder)

    print('###### Command Line Options ######')
    print('# Domain : {}'.format(domain))
    print('# SEED   : {}'.format(seed))
    print('# Folder : {}'.format(folder))
    print('##################################')

    flags = {
        'domain': domain,
        'seed': seed,
        'folder': folder
    }
    return flags


def f0_ant_hrl(x):
    # return T
    T = Variable(x[:, 29:30], requires_grad=False)
    return T
    
def f1_ant_hrl(x):
    # return XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.cat((X, Y), dim=1), requires_grad=False)

def f2_ant_hrl(x):
    # return THETA_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.atan2(Y, X), requires_grad=False)

def f3_ant_hrl(x):
    # return DISTANCE_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.linalg.norm(torch.cat((X, Y), dim=1), dim=1, keepdim=True), requires_grad=False)

def f4_ant_hrl(x):
    # return BOX_YZ
    BOX_Y = Variable(x[:, 31:32], requires_grad=False)
    BOX_Z = Variable(x[:, 32:33], requires_grad=False)
    return Variable(torch.cat((BOX_Y, BOX_Z), dim=1), requires_grad=False)


# function entities and their width
ANT_HRL_FUNCTIONS = [(f0_ant_hrl, 1), (f1_ant_hrl, 2), (f2_ant_hrl, 1), (f3_ant_hrl, 1), (f4_ant_hrl, 2)]


# all valid encoding and width of functions
DESIGNED = [([0, 1, 0, 0, 0], 2)]
ALL_ANT_HRL_FUNCTIONS = [([0, 1, 1, 1, 0], 4)]
ALL_ANT_HRL_FUNCTIONS_BOX_FALL = [([0, 1, 1, 1, 1], 6)]


# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))


def parse_domain(domain):
    if domain == 'ant_maze':
        input_dict = dict(
            models=ANT_LOW_TORQUE_MODELS, 
            functions=ANT_HRL_FUNCTIONS, 
            all_functions=ALL_ANT_HRL_FUNCTIONS,
            input_dim=30, 
            num_action_space=8, 
            index_action_space=range(2, 29),
            elevated=False)
    
    elif domain == 'ant_push':
        input_dict = dict(
            models=ANT_LOW_TORQUE_MODELS, 
            functions=ANT_HRL_FUNCTIONS, 
            all_functions=ALL_ANT_HRL_FUNCTIONS,
            input_dim=30, 
            num_action_space=8, 
            index_action_space=range(2, 29),
            elevated=False)
    
    elif domain == 'ant_fall':
        input_dict = dict(
            models=ANT_LOW_TORQUE_MODELS, 
            functions=ANT_HRL_FUNCTIONS, 
            all_functions=ALL_ANT_HRL_FUNCTIONS_BOX_FALL, 
            input_dim=33, 
            num_action_space=8,
            index_action_space=range(2, 29), 
            elevated=True)
    
    else:
        print('Please check your domain')
        exit()
    
    return input_dict


class WEIGHTCell(nn.Module):
    def __init__(self, index, num_models):
        super().__init__()

        self.index = index
        self.num_models = num_models

        self.W = nn.Parameter(torch.zeros(1, self.num_models))
        torch.nn.init.ones_(self.W)

    def forward(self, x):
        return self.W


class SYMBOLICCell(nn.Module):
    def __init__(self, index, functions, all_functions):
        super().__init__()

        self.index = index
        self.functions = functions
        self.all_functions = all_functions  # actually only one

        assert len(self.all_functions) == 1
        self.P = nn.Linear(self.all_functions[0][1], 1, bias=True)

    def forward(self, x):

        f_code = self.all_functions[0][0]

        SYMBOLIC = []
        for j in range(len(self.functions)):
            if f_code[j]:
                SYMBOLIC.append(self.functions[j][0](x))

        sym_input = torch.cat(SYMBOLIC, dim=1)

        P = torch.sigmoid(self.P(sym_input))        
        return P


# nested ITE program
class Program(nn.Module):
    def __init__(self, 
                 depth, 
                 models,
                 functions,
                 all_functions,
                 input_dim,
                 num_action_space,
                 index_action_space,
                 elevated=False,
                 beta=0.5):
        super().__init__()
        
        self.depth = depth  # program AST depth
        self.models = models
        self.num_models = len(self.models)

        self.functions = functions
        self.all_functions = all_functions
        self.input_dim = input_dim
        self.num_action_space = num_action_space
        self.index_action_space = index_action_space
        self.elevated = elevated
        self.beta = beta
        self.guided = False

        self.cells = nn.ModuleList()
        self.num_cells = self.depth * 2 - 1

        for i in range(self.num_cells):
            if i % 2 == 0 and i != self.num_cells - 1:
                self.cells.append(SYMBOLICCell(i, self.functions, self.all_functions))
            else:
                self.cells.append(WEIGHTCell(i, self.num_models))

        # P_map
        self.P_map = np.eye(self.depth, self.depth - 1)
        for i in range(self.P_map.shape[0]):
            for j in range(self.P_map.shape[1]):
                if i > j:
                    self.P_map[i, j] = -1

    @staticmethod
    def get_action(model, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
        return torch.from_numpy(action)

    def forward(self, x):

        # NOTE: In AntFall environment, the whole environment
        #       is elevated by a box with height of 4.
        if self.elevated:
            x[:, 2] -= 4
        
        if self.depth == 1:
            W = [self.cells[0](x)]
            W_coefficient = [1]

        else:
            # compute without influence
            P = []
            W = []

            for i in range(self.num_cells):
                c = self.cells[i]
                if i % 2 == 0 and i != self.num_cells - 1:
                    P.append(c(x))
                else:
                    W.append(c(x))

            # compute with influence
            W_coefficient = [1 for i in range(len(W))]
            
            P_map = self.P_map
            for i in range(P_map.shape[0]):
                for j in range(P_map.shape[1]):
                    if P_map[i, j] == 1:
                        W_coefficient[i] *= P[j]
                    elif P_map[i, j] == -1:
                        W_coefficient[i] *= 1 - P[j]
                    elif P_map[i, j] == 0:
                        pass

        w = torch.zeros(x.shape[0], self.num_models)
        for i in range(len(W)):
            w += W[i] * W_coefficient[i]

        w = (w / self.beta).softmax(dim=1)

        action = torch.zeros(x.shape[0], self.num_action_space)
        for i in range(self.num_models):
            action += w[:, i:i+1] * self.get_action(self.models[i], x[:, self.index_action_space])

        return action

    def guide(self, points):

        if self.guided:
            return

        startx, starty = points[0, 0], points[0, 1]
        endx, endy = points[1, 0], points[1, 1]
        dx, dy = endx - startx, endy - starty
        
        guide = np.array([[0, 0, 0, 0]], dtype=np.float32)  # [UP, DOWN, LEFT, RIGHT]

        if dx != 0 and dy != 0:
            short = min(abs(dx), abs(dy))  # shorest to one
            dx, dy = dx / short, dy / short
            
            if dx < 0:
                # LEFT
                guide[0, 2] += abs(dx)
            elif dx > 0:
                # RIGHT
                guide[0, 3] += dx

            if dy < 0:
                # DOWN
                guide[0, 1] += abs(dy)
            elif dy > 0:
                # UP
                guide[0, 0] += dy
        
        elif (dx == 0 and dy != 0) or (dx != 0 and dy == 0):
            if dx < 0:
                # LEFT
                guide[0, 2] += 1.0
            elif dx > 0:
                # RIGHT
                guide[0, 3] += 1.0

            if dy < 0:
                # DOWN
                guide[0, 1] += 1.0
            elif dy > 0:
                # UP
                guide[0, 0] += 1.0

        # to match the real primitive policies
        actual_guide = guide[:, [3, 2, 0, 1]]

        for cell in self.cells:
            if isinstance(cell, WEIGHTCell):
                cell.W = nn.Parameter(torch.from_numpy(actual_guide))

        self.guided = True


# parse command input
flags = parse_command_line_options()

# set up seed
SEED = flags['seed']
torch.manual_seed(SEED)
random.seed(SEED)

# set up folder to save
folder = flags['folder']

# set up the domain
domain = flags['domain']

# print the planning for once
print_planning(domain)

# set up the environment
if domain == 'ant_maze':
    e = GymEnv('mjrl_spec_ant_maze-v1')
    depth_list = [3, 3, 3]
    niter_list = [201, 401, 401]
    num_traj = 100

elif domain == 'ant_push':
    e = GymEnv('mjrl_spec_ant_push-v1')
    depth_list = [3, 5, 5]
    niter_list = [301, 301, 301]
    num_traj = 100

elif domain == 'ant_fall':
    e = GymEnv('mjrl_spec_ant_fall-v1')
    depth_list = [3, 3, 5, 5]
    niter_list = [101, 201, 201, 201]
    num_traj = 150

# restrict number of steps in each phase of learning
time_limit = e.env._max_episode_steps / e.env.env.specs_size()
print (f'episode_steps for each learning phase {time_limit}')
e.env._max_episode_steps = time_limit
e.env.env.set_timelimit(time_limit)
e._horizon = time_limit

# phase learning
for i in range(e.env.specs_size()):
    prog = Program(depth=depth_list[i], **parse_domain(domain))
    
    # provides guidance
    prog.guide(e.env.env.specs[i]['points'])
    
    policy = ProgPolicy(e.spec, prog=prog, seed=SEED)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
    agent = TRPO(e, policy, baseline, kl_dist=None, normalized_step_size=0.02,seed=SEED, save_logs=True)
    
    print (f"------------- Training phase {i} -------------")
    job_name = folder + '/' + domain + '_' + str(i) + '-[' + str(SEED) + ']' 
    train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=niter_list[i],
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=10,
            sample_mode='trajectories',
            num_traj=num_traj,
            save_freq=10,
            evaluation_rollouts=None,
            plot_keys=['stoc_pol_mean', 'running_score'])

    # NOTE: new best_policy.pickle will substitue previous one after
    #       exit and resume, it is safer to specify a indexed policy.
    
    #pi = job_name + '/iterations/best_policy.pickle'
    pi = job_name + '/iterations/policy_' + str(niter_list[i] - 1) + '.pickle'

    policy = pickle.load(open(pi, 'rb'))
    e.env.advance(policy)
