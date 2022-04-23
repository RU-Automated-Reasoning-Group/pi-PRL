# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from mjrl.envs.ant_maze_env import AntMazeEnv

def create_maze_env(env_name=None, top_down_view=False, max_timesteps=0):
  
  n_bins = 0
  manual_collision = False
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 8

  else:
    assert False, 'unknown env %s' % env_name

  maze_id = None
  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze':
    maze_id = 'Maze'
  elif env_name == 'Push':
    maze_id = 'Push'
  elif env_name == 'Fall':
    maze_id = 'Fall'
  else:
    raise ValueError('Unknown maze environment %s' % env_name)


  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'observe_blocks': observe_blocks,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  
  return gym_env
