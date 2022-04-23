from mjrl.envs.ant_hrl_planning import SpecAntMaze, SpecAntPush, SpecAntFall

def print_planning(domain):
    
    if domain == 'ant_maze':
        e = SpecAntMaze(silent=False)
    elif domain == 'ant_push':
        e = SpecAntPush(silent=False)
    elif domain == 'ant_fall':
        e = SpecAntFall(silent=False)
    else:
        print('Please check your domain')
        exit()
