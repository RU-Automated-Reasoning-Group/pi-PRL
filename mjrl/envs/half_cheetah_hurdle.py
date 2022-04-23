import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer


class HalfCheetahHurdleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_hurdle.xml',
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())
        
        self.exteroceptive_observation =[12.0, 0.0, 0.5]

        self.hurdles_xpos=[-15., -13., -9., -5., -1., 3., 7., 11., 15.]

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def step(self, action):

        x_position_before = self.sim.data.qpos[0]
        z_position_before = self.sim.data.qpos[1]
        
        self.do_simulation(action, self.frame_skip)
        xyz_position = self.get_body_com('torso')

        x_position_after = self.sim.data.qpos[0]
        z_position_after = self.sim.data.qpos[1]

        x_velocity = ((x_position_after - x_position_before) / self.dt)
        z_velocity = np.abs(((z_position_after - z_position_before) / self.dt))
        
        run_reward = x_velocity
        jump_reward = z_velocity
        
        observation= self._get_obs()

        if self.isincollision():
            collision_penality = -2.0
        else:
            collision_penality = 0.0
        
        hurdle_reward = self.get_hurdle_reward()
        done = False
        goal_reward = 0
        goal_distance = np.linalg.norm(xyz_position - self.exteroceptive_observation)
        
        if (goal_distance) < 1.0:
            done = True
            goal_reward = 1000
            goal_distance = 0.0
        else:
            goal_distance -= 1
            done = False
            
        reward = -1e-1 * goal_distance + hurdle_reward + goal_reward + run_reward + 3e-1 * jump_reward + collision_penality
        
        total_distance = np.linalg.norm(np.array([0, 0, 0]) - self.exteroceptive_observation)
        progress = (1.0 - goal_distance / (total_distance - 1)) * 100 

        info = {
            'distance': goal_distance,
            'progress': progress,
            'finished': True if done else False
        }  # 'progress' is also needed

        return observation, reward, done, info

    def isincollision(self):
        hurdle_size=[0.05,1.0,0.03]
        x_pos =self.get_body_com('ffoot')[0]
        matches = [x for x in self.hurdles_xpos if x >= x_pos]
        if len(matches)==0:
            return False
        hurdle_pos =[matches[0],0.0,0.20]
        names=['ffoot']
        xyz_pos=[]
        for i in range(0,len(names)):
            xyz_pos.append(self.get_body_com(names[i]))
        for i in range(0,len(names)):
            cf=True
            for j in range(0,1):
                if abs(hurdle_pos[j]-xyz_pos[i][j])>1.5*hurdle_size[j]:
                    cf=False
                    break
            if cf:
                return True
        return False

    def get_hurdle_reward(self):
        hurdle_size = [0.05, 1.0, 0.03]
        x_pos = self.get_body_com('bfoot')[0]
        matches = [x for x in self.hurdles_xpos if x >= x_pos]
        hurdle_reward = -1.0 * len(matches)
        
        return hurdle_reward

    def _get_obs(self):
        
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        proprioceptive_observation = np.concatenate((position, velocity)).ravel()
        
        x_pos1 =self.get_body_com('ffoot')[0]
        x_pos2 =self.get_body_com('bfoot')[0]
        matches = [x for x in self.hurdles_xpos if x >= x_pos2]
        next_hurdle_x_pos = [matches[0]]
        ff_dist_frm_next_hurdle=[np.linalg.norm(matches[0] - x_pos1)]
        bf_dist_frm_next_hurdle=[np.linalg.norm(matches[0] - x_pos2)]
        
        observation =np.concatenate([proprioceptive_observation, next_hurdle_x_pos, bf_dist_frm_next_hurdle]).reshape(-1)
        return observation

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

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -60
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.azimuth = 0