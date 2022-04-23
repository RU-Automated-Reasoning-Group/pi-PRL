import numpy as np


class SpecEnv():
    def __init__(self, spec):
        self.specs_ind = 0
        self.specs = spec
        self.policies = []
        self.timelimit = 0

    def specs_size(self):
        return len(self.specs)

    def set_timelimit(self, timelimit):
        self.timelimit = timelimit

    def advance(self, policy):
        self.specs_ind += 1
        self.policies.append(policy)

    def reset_func(self, x):
        if (self.specs_ind == 0):
            return x
        
        t = 0
        i = 0
        horizon = self.specs_ind * self.timelimit
    
        while t < horizon and i < self.specs_ind:
            
            a = self.policies[i].get_action(x)[1]['evaluation']
            
            x, _, _, _ = self.step(a)
            reach_rwd = self.specs[i]['reach'](x)

            if reach_rwd > 0 or t + 1 >= self.timelimit * (i+1):
                i += 1 # move to the next phase
            t = t + 1

        return x

    def reward_func(self, x):
        safe_rwd = 0

        done = False
        reach_rwd = self.specs[self.specs_ind]['reach'](x)
        if reach_rwd > 0:
            done = True

        rwd = safe_rwd + reach_rwd
        return rwd, done

    def eval(self, horizon, num_episodes=1, mode='exploration', discrete=False):
        for ep in range(num_episodes):
            t = 0
            self.specs_ind = 0
            x = self.reset()
            score = 0.0
            while t < horizon and self.specs_ind < self.specs_size():
                self.render()
                a = self.policies[self.specs_ind].get_action(x, discrete=discrete)[0] \
                    if mode == 'exploration' else self.policies[self.specs_ind].get_action(x, discrete=discrete)[1]['evaluation']
                x, r, _, _ = self.step(a)
                reach_rwd = self.specs[self.specs_ind]['reach'](x)
                if reach_rwd > 0 or t + 1 >= self.timelimit * (self.specs_ind+1):
                    self.specs_ind += 1 # move to the next phase
                t = t + 1
                score = score + r
            self.render()
            print(f"Episode score = {score} within {t} steps")
