from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------


register(
    id='mjrl_ant_up-v3',
    entry_point='mjrl.envs:AntEnvUp',
    max_episode_steps=1000
)

register(
    id='mjrl_ant_down-v3',
    entry_point='mjrl.envs:AntEnvDown',
    max_episode_steps=1000
)

register(
    id='mjrl_ant_left-v3',
    entry_point='mjrl.envs:AntEnvLeft',
    max_episode_steps=1000
)

register(
    id='mjrl_ant_right-v3',
    entry_point='mjrl.envs:AntEnvRight',
    max_episode_steps=1000
)

register(
    id='mjrl_cross_maze_ant-v1',
    entry_point='mjrl.envs:CrossMazeAntEnv',
    max_episode_steps=1000
)


register(
    id='mjrl_cross_maze_ant_random-v1',
    entry_point='mjrl.envs:CrossMazeAntRandomEnv',
    max_episode_steps=1000
)


register(
    id='mjrl_random_goal_ant-v1',
    entry_point='mjrl.envs:RandomGoalAntEnv',
    max_episode_steps=500
)


register(
    id='mjrl_half_cheetah_forward-v3',
    entry_point='mjrl.envs:HalfCheetahForwardEnv',
    max_episode_steps=1000
)


register(
    id='mjrl_half_cheetah_jump-v3',
    entry_point='mjrl.envs:HalfCheetahJumpEnv',
    max_episode_steps=1000
)


register(
    id='mjrl_half_cheetah_hurdle-v3',
    entry_point='mjrl.envs:HalfCheetahHurdleEnv',
    max_episode_steps=1000
)


register(
    id='mjrl_pusher2d_left-v1',
    entry_point='gym_envs.envs:Pusher2dLeftEnv',
    max_episode_steps=100
)


register(
    id='mjrl_pusher2d_down-v1',
    entry_point='mjrl.envs:Pusher2dDownEnv',
    max_episode_steps=100
)


register(
    id='mjrl_pusher2d-v1',
    entry_point='mjrl.envs:Pusher2dEnv',
    max_episode_steps=100
)


register(
    id='mjrl_ant_push-v1',
    entry_point='mjrl.envs:AntPush',
    max_episode_steps=1000
)

register(
    id='mjrl_ant_fall-v1',
    entry_point='mjrl.envs:AntFall',
    max_episode_steps=1000
)

register(
    id='mjrl_ant_maze-v1',
    entry_point='mjrl.envs:AntMaze',
    max_episode_steps=1000
)

register(
    id='mjrl_spec_ant_maze-v1',
    entry_point='mjrl.envs:SpecAntMaze',
    max_episode_steps=1000
)

register(
    id='mjrl_spec_ant_fall-v1',
    entry_point='mjrl.envs:SpecAntFall',
    max_episode_steps=1000
)

register(
    id='mjrl_spec_ant_push-v1',
    entry_point='mjrl.envs:SpecAntPush',
    max_episode_steps=1000
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.ant_v3 import AntEnvUp, AntEnvDown, AntEnvLeft, AntEnvRight
from mjrl.envs.cross_maze_ant import CrossMazeAntEnv, CrossMazeAntRandomEnv
from mjrl.envs.random_goal_ant import RandomGoalAntEnv
from mjrl.envs.half_cheetah_v3 import HalfCheetahForwardEnv, HalfCheetahJumpEnv
from mjrl.envs.half_cheetah_hurdle import HalfCheetahHurdleEnv
from mjrl.envs.pusher_2d import Pusher2dEnv, Pusher2dLeftEnv, Pusher2dDownEnv
from mjrl.envs.ant_hrl import AntPush, AntFall, AntMaze
from mjrl.envs.ant_hrl_planning import SpecAntMaze, SpecAntFall, SpecAntPush