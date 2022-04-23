import numpy as np
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mjrl.envs.ant_v3 import AntEnv
#from gym.envs.mujoco import AntEnv


class CrossMazeAntEnv(AntEnv):
    def __init__(self,
                goal_position=[6, -6],
                goal_reward_weight=3e-1,
                goal_radius=1):
        
        self.goal_position = np.array(goal_position)
        self.goal_reward_weight = goal_reward_weight
        self.goal_radius = goal_radius

        # distance from starting point to goal position
        self.total_distance = np.linalg.norm(self.goal_position, ord=2)
        
        # note that this is very different from default configuration
        AntEnv.__init__(self, 
            xml_file='cross_maze_ant.xml',
            ctrl_cost_weight=1e-2,
            contact_cost_weight=1e-3,
            healthy_reward=5e-2,
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False)
        

    def step(self, action):
        
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        # check if reach the goal
        goal_distance = np.linalg.norm(xy_position_after - self.goal_position, ord=2)
        goal_reached = goal_distance < self.goal_radius

        # update goal_distance if reached
        goal_distance = 0.0 if goal_reached else goal_distance

        goal_reward = (5 - goal_distance) * self.goal_reward_weight  # dense
        
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        healthy_reward = self.healthy_reward

        rewards = goal_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        
        # the easiest way to terminate current trajectory
        done = True if goal_reached else self.done

        observation = self._get_obs()

        # percentage of finished process, can be negative
        progress = (1.0 - goal_distance / self.total_distance) * 100
        
        info = {
            'reward_goal': goal_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'finished': goal_reached,
            'distance': goal_distance,
            'progress': progress,
        }

        return observation, reward, done, info

    # copied from up-to-date implementation
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -60
        self.viewer.cam.lookat[0] = 7
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.azimuth = 0

# with three possible goal positions
class CrossMazeAntRandomEnv(AntEnv):
    def __init__(self,
                possible_goal_position=[[6, -6], [12, 0], [6, 6]],
                goal_reward_weight=3e-1,
                goal_radius=1,
                **kwargs):
        
        self.possible_goal_positions = possible_goal_position
        self.goal_position = self.possible_goal_positions[
            np.random.choice(len(self.possible_goal_positions))]
        self.goal_reward_weight = goal_reward_weight
        self.goal_radius = goal_radius

        # distance from starting point to goal position
        self.total_distance = np.linalg.norm(self.goal_position, ord=2)
        
        # note that this is very different from default configuration
        AntEnv.__init__(self, 
            xml_file='cross_maze_ant.xml',
            ctrl_cost_weight=1e-2,
            contact_cost_weight=1e-3,
            healthy_reward=5e-2,
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False)
        

    def step(self, action):
        
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        # check if reach the goal
        goal_distance = np.linalg.norm(xy_position_after - self.goal_position, ord=2)
        goal_reached = goal_distance < self.goal_radius

        # update goal_distance if reached
        goal_distance = 0.0 if goal_reached else goal_distance - self.goal_radius

        goal_reward = (5 - goal_distance) * self.goal_reward_weight  # dense

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        healthy_reward = self.healthy_reward

        rewards = goal_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        
        # the easiest way to terminate current trajectory
        done = True if goal_reached else self.done

        observation = np.concatenate([self._get_obs(), self.goal_position])

        # percentage of finished process, can be negative
        progress = (1.0 - goal_distance / (self.total_distance - self.goal_radius)) * 100
        
        info = {
            'reward_goal': goal_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'finished': goal_reached,
            'distance': goal_distance,
            'progress': progress,
        }

        return observation, reward, done, info

    # copied from up-to-date implementation
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def reset_model(self):

        # reset goal position
        self.goal_position = self.possible_goal_positions[
            np.random.choice(len(self.possible_goal_positions))]

        # distance from starting point to goal position
        self.total_distance = np.linalg.norm(self.goal_position, ord=2)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = np.concatenate([self._get_obs(), self.goal_position])

        return observation

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -60
        self.viewer.cam.lookat[0] = 7
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.azimuth = 0