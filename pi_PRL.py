import sys
import random
import copy
import getopt
import os

os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_prog import ProgPolicy
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.trpo import TRPO
from mjrl.utils.train_agent import train_agent, train_agent_flip


def parse_command_line_options():
    options, remainder = getopt.getopt(sys.argv[1:], 'e:s:d:')

    # default setting
    domain = 'ant_cross_maze'
    seed = 123
    folder = 'data'

    for option, arg in options:
        if option in ('-e'):
            env = int(arg)
            if env == 0:
                domain = 'ant_cross_maze'
            elif env == 1:
                domain = 'ant_random_goal'
            elif env == 2:
                domain = 'half_cheetah_hurdle'
            elif env == 3:
                domain = 'pusher_2d'
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


def f0_pusher(x):
    # return x (input)
    return x


def f1_pusher(x):
    # return PALM_XY
    return Variable(x[:, -6:-4], requires_grad=False)


def f2_pusher(x):
    # return OBJ_XY
    return Variable(x[:, -3:-1], requires_grad=False) 


def f0_ant(x):
    # return XY + GXY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    GX = Variable(x[:, -2:-1], requires_grad=False)
    GY = Variable(x[:, -1:], requires_grad=False)
    return Variable(torch.cat((X, Y, GX, GY), dim=1), requires_grad=False)


def f1_ant(x):
    # return THETA_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.atan2(Y, X), requires_grad=False)


def f2_ant(x):
    # return DISTANCE_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.linalg.norm(torch.cat((X, Y), dim=1), dim=1, keepdim=True), requires_grad=False)


def f0_hc(x):
    # return X, current position
    return Variable(x[:, 0:1], requires_grad=False)


def f1_hc(x):
    # reuturn NEXT, next hurdle position
    return Variable(x[:, -2:-1], requires_grad=False)


def f2_hc(x):
    # return BF_TO_NEXT, back foot to next hurdle
    return Variable(x[:, -1:], requires_grad=False)


# function entities and their width
PUSHER_FUNCTIONS = [(f0_pusher, 15), (f1_pusher, 2), (f2_pusher, 2)]
ANT_FUNCTIONS = [(f0_ant, 4), (f1_ant, 1), (f2_ant, 1)]
HC_FUNCTIONS = [(f0_hc, 1), (f1_hc, 1), (f2_hc, 1)]

# all valid encoding and width of functions
ALL_PUSHER_FUNCTIONS = [([0, 1, 1], 4)]
ALL_ANT_FUNCTIONS = [([1, 1, 1], 6)]
ALL_HC_FUNCTIONS = [([0, 1, 1], 2)]

# all primitive policies
PUSHER_MODELS = []
for name in ['left', 'down']:
    filename = os.getcwd() + '/primitives/pusher2d/' + name + '.pt'
    PUSHER_MODELS.append(torch.load(filename))

ANT_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant/' + direction + '.pt'
    ANT_MODELS.append(torch.load(filename))

HC_MODELS = []
for name in ['forward', 'jump']:
    filename = os.getcwd() + '/primitives/hc/' + name + '.pt'
    HC_MODELS.append(torch.load(filename))


def parse_domain(domain):
    if domain == 'ant_random_goal':
        input_dict = dict(models=ANT_MODELS, functions=ANT_FUNCTIONS, all_functions=ALL_ANT_FUNCTIONS, input_dim=115, num_action_space=8, index_action_space=range(2, 113))
    elif domain == 'ant_cross_maze':
        input_dict = dict(models=ANT_MODELS, functions=ANT_FUNCTIONS, all_functions=ALL_ANT_FUNCTIONS, input_dim=115, num_action_space=8, index_action_space=range(2, 113))
    elif domain == 'pusher_2d':
        input_dict = dict(models=PUSHER_MODELS, functions=PUSHER_FUNCTIONS, all_functions=ALL_PUSHER_FUNCTIONS, input_dim=15, num_action_space=3, index_action_space=range(15))
    elif domain == 'half_cheetah_hurdle':
        input_dict = dict(models=HC_MODELS, functions=HC_FUNCTIONS, all_functions=ALL_HC_FUNCTIONS, input_dim=20, num_action_space=6, index_action_space=range(1, 18))
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
                 beta=0.5):
        super().__init__()
        
        self.depth = depth
        self.models = models
        self.num_models = len(self.models)

        self.functions = functions
        self.all_functions = all_functions
        self.input_dim = input_dim
        self.num_action_space = num_action_space
        self.index_action_space = index_action_space
        self.beta = beta

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


# architecture
class SimpleSearchMap(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.depth = depth
        self.type = 'simple'

        # NOTE: for simplicity, we use softmax over a vector
        #       to represent the distribution of the programs.
        #       If the production rules of the DSL to expand
        #       programs are more than two, maintaining independent
        #       parameters for choosing programs in each layer 
        #       of derivation graph is recommended.

        self.v = nn.Parameter(torch.zeros(self.depth), requires_grad=True)
        torch.nn.init.ones_(self.v)

    def freeze(self):
        self.v.requires_grad = False

    def unfreeze(self):
        self.v.requires_grad = True

# architecture
class ArchitectureSearchMap(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.depth = depth
        self.type = 'architecture'

        # layer by layer
        self.options = nn.ParameterList()
        for _ in range(self.depth-1):
            self.options.append(nn.Parameter(torch.rand(2), requires_grad=True))

    def freeze(self):
        for option in self.options:
            option.requires_grad = False

    def unfreeze(self):
        for option in self.options:
            option.requires_grad = True



# fusion ITE programs
# parameters
class FusionPrograms(nn.Module):
    def __init__(self, 
                 depth,
                 models,
                 functions,
                 all_functions,
                 input_dim,
                 num_action_space,
                 index_action_space,
                 beta=0.5):

        super().__init__()
        
        self.depth = depth
        self.models = models
        self.num_models = len(self.models)

        self.functions = functions
        self.all_functions = all_functions
        self.input_dim = input_dim
        self.num_action_space = num_action_space
        self.index_action_space = index_action_space

        self.beta = beta

        # shared cells
        self.cells = nn.ModuleList()
        self.num_shared_cells = 2 * self.depth - 2

        for i in range(self.num_shared_cells):
            if i % 2 == 0:
                self.cells.append(SYMBOLICCell(i, self.functions, self.all_functions))
            else:
                self.cells.append(WEIGHTCell(i, self.num_models))

        # exclusive cells
        self.ex_cells = nn.ModuleList()
        self.num_exclusive_cells = self.depth

        for i in range(self.num_exclusive_cells):
            self.ex_cells.append(WEIGHTCell(i, self.num_models))

        # P_maps
        self.P_maps = []
        for d in range(self.depth):
            depth = d + 1
            P_map = np.eye(depth, depth - 1)
            for i in range(P_map.shape[0]):
                for j in range(P_map.shape[1]):
                    if i > j:
                        P_map[i, j] = -1
            self.P_maps.append(P_map)

    @staticmethod
    def get_action(model, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
        return torch.from_numpy(action)

    def forward(self, x, search_map):

        # For SimpleSearchMap
        if search_map.type == 'simple':
            v = F.softmax(search_map.v, dim=0)
        # For ArchitectureSearchMap
        elif search_map.type == 'architecture':
            v = nn.Parameter(torch.ones(self.depth), requires_grad=False)
            for i in range(len(v)):
                options = search_map.options
                if i == 0:
                    v[i] = options[0].softmax(dim=0)[0]
                else:
                    prev = 1
                    for j in range(i):
                        prev *= options[j].softmax(dim=0)[1]
                    if i == len(v) - 1:
                        v[i] = prev
                    else:
                        option_value = options[i].softmax(dim=0)
                        v[i] = prev * option_value[0]

        action = torch.zeros(x.shape[0], self.num_action_space)

        # primitive policies actions
        primitive_actions = []
        for i in range(self.num_models):
            primitive_actions.append(self.get_action(self.models[i], x[:, self.index_action_space]))
        
        if self.depth == 1:
            w = self.ex_cells[0](x)
            w = (w / self.beta).softmax(dim=1)
            for i in range(self.num_models):
                action += w[:, i:i+1] * primitive_actions[i]
            
        else:
            P = []
            W = []
            ex_W = []
            
            for i in range(self.num_shared_cells):
                c = self.cells[i]
                if i % 2 == 0:
                    P.append(c(x))
                else:
                    W.append(c(x))

            for i in range(self.num_exclusive_cells):
                c = self.ex_cells[i]
                ex_W.append(c(x))

            for d in range(self.depth):

                depth = d + 1
                if depth == 1:
                    w = ex_W[0]
                    w = (w / self.beta).softmax(dim=1)
                    for i in range(self.num_models):
                        action += v[0] * w[:, i:i+1] * primitive_actions[i]
                
                else:
                    # independent P_map and W_coefficient for each program
                    P_map = self.P_maps[d]

                    W_coefficient = [1 for i in range(P_map.shape[0])]
                    for i in range(P_map.shape[0]):
                        for j in range(P_map.shape[1]):
                            if P_map[i, j] == 1:
                                W_coefficient[i] *= P[j]
                            elif P_map[i, j] == -1:
                                W_coefficient[i] *= 1 - P[j]
                            elif P_map[i, j] == 0:
                                pass

                    w = torch.zeros(x.shape[0], self.num_models)
                    for i in range(len(W_coefficient) - 1):
                        w += W[i] * W_coefficient[i]
                    w += ex_W[d] * W_coefficient[i + 1]
                    w = (w / self.beta).softmax(dim=1)

                    for i in range(self.num_models):
                        action += v[d] * w[:, i:i+1] * primitive_actions[i]

        return action

    def freeze(self):
        for cell in self.cells:
            for param in cell.parameters():
                param.requires_grad = False

        for cell in self.ex_cells:
            for param in cell.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for cell in self.cells:
            for param in cell.parameters():
                param.requires_grad = True

        for cell in self.ex_cells:
            for param in cell.parameters():
                param.requires_grad = True


# program derivation graph
class SearchFusionProgram(nn.Module):
    def __init__(self, graph_depth, domain, beta=0.5, search_map_type='architecture'):
        super().__init__()

        # NOTE: graph_depth is different from program AST depth,
        #       e.g., when graph_depth is set to 2, our algorithm 
        #       would only search for a single program with AST
        #       depth = 1, i.e., a simple low-level controller.
        #       For  the program AST depth, "if" is on upper 
        #       level, "then" and "else" are on the lower level.

        self.domain = domain
        self.graph_depth = graph_depth
        self.search_map_type = search_map_type
        if search_map_type == 'architecture':
            self.search_map = ArchitectureSearchMap(depth=self.graph_depth - 1)
        elif search_map_type == 'simple':
            self.search_map = SimpleSearchMap(depth=self.graph_depth - 1)
        self.fusion_programs = FusionPrograms(depth=self.graph_depth - 1, **parse_domain(domain), beta=beta)

        self.search_map.unfreeze()
        self.fusion_programs.freeze()
        self.pointer = 0  # 0 for optimizing architecture, 1 for optimizing programs

    def flip(self):
        if self.pointer == 0:
            self.pointer = 1
            self.search_map.freeze()
            self.fusion_programs.unfreeze()
        elif self.pointer == 1:
            self.pointer = 0
            self.search_map.unfreeze()
            self.fusion_programs.freeze()

    def forward(self, x):
        action = self.fusion_programs(x, self.search_map)
        return action

   # return a discrete program
    def extract(self):

        # NOTE: there is no better method to extract a program
        #       over others. For simplicity, we can select a 
        #       program simply by its proportion.

        if self.search_map_type == 'simple':
            index = self.search_map.v.argmax()
        elif self.search_map_type == 'architecture':
            v = nn.Parameter(torch.ones(self.graph_depth-1), requires_grad=False)
            for i in range(len(v)):
                options = self.search_map.options
                if i == 0:
                    v[i] = options[0].softmax(dim=0)[0]
                else:
                    prev = 1
                    for j in range(i):
                        prev *= options[j].softmax(dim=0)[1]
                    if i == len(v) - 1:
                        v[i] = prev
                    else:
                        option_value = options[i].softmax(dim=0)
                        v[i] = prev * option_value[0]
            index = v.argmax()

        prog = Program(depth=index + 1, **parse_domain(self.domain))

        # empty the cells
        prog.cells = nn.ModuleList()
        
        # extract shared cells
        for i in range(2 * index):
            prog.cells.append(copy.deepcopy(self.fusion_programs.cells[i]))
        
        # extract exclusive cell
        prog.cells.append(copy.deepcopy(self.fusion_programs.ex_cells[index]))
        
        return prog


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

# set up the environment
if domain == 'ant_cross_maze':
    e = GymEnv('mjrl_cross_maze_ant_random-v1')
    exp_name = 'cross'
    train_iter = 50
    num_traj = 50
    tune_iter = 250
    arch_iter = 1
    prog_iter = 10
    
elif domain == 'ant_random_goal':
    e = GymEnv('mjrl_random_goal_ant-v1')
    exp_name = 'random'
    train_iter = 150
    tune_iter = 150
    num_traj = 80
    arch_iter = 1
    prog_iter = 1

elif domain == 'half_cheetah_hurdle':
    e = GymEnv('mjrl_half_cheetah_hurdle-v3')
    exp_name = 'hc'
    train_iter = 100
    tune_iter = 100
    num_traj = 50
    arch_iter = 1
    prog_iter = 1

elif domain == 'pusher_2d':
    e = GymEnv('mjrl_pusher2d-v1')
    exp_name = 'pusher'
    train_iter = 10
    tune_iter = 100
    num_traj = 50
    arch_iter = 1
    prog_iter = 1

 
# program derivation graph training
prog = SearchFusionProgram(graph_depth=6, domain=domain, beta=0.25, search_map_type='architecture')
policy = ProgPolicy(e.spec, prog=prog, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = TRPO(e, policy, baseline, kl_dist=None, normalized_step_size=0.02,seed=SEED, save_logs=True)

train_agent_flip(job_name=folder + '/' + str(exp_name) + '-[' + str(SEED) + ']',
            agent=agent,
            seed=SEED,
            niter=train_iter,
            gamma=0.95,
            gae_lambda=0.97,
            arch_kl_dist=0.02,
            arch_iter=arch_iter,
            prog_kl_dist=0.02,
            prog_iter=prog_iter,
            num_cpu=10,
            sample_mode='trajectories',
            num_traj=num_traj,
            save_freq=5,
            plot_keys=['stoc_pol_mean', 'running_score'])


# extracted program fine-tuning
extracted_prog = agent.policy.extract()
policy = ProgPolicy(e.spec, prog=extracted_prog, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = TRPO(e, policy, baseline, kl_dist=None, normalized_step_size=0.02,seed=SEED, save_logs=True)

train_agent(job_name=folder + '/' + str(exp_name) + '-[extracted]-[' + str(SEED) + ']',
            agent=agent,
            seed=SEED,
            niter=tune_iter,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=10,
            sample_mode='trajectories',
            num_traj=num_traj,
            save_freq=10,
            plot_keys=['stoc_pol_mean', 'running_score'])
