import numpy as np
from UTIL.tensor_ops import my_view, copy_clone
try:
    from numba import jit
except:
    from UTIL.tensor_ops import dummy_decorator as jit


def to_cpu_numpy(x):
    return x.cpu().numpy() if hasattr(x,'cpu') else x

class CoopAlgConfig():
    reserve = None
    
class DummyAlgorithmFoundationHI3D():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        super().__init__()
        self.n_agent = n_agent
        ScenarioConfig = GlobalConfig.ScenarioConfig
        self.num_entity = ScenarioConfig.num_entity
        self.landmark_uid = ScenarioConfig.uid_dictionary['landmark_uid']
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.entity_uid = ScenarioConfig.uid_dictionary['entity_uid']
        self.pos_decs = ScenarioConfig.obs_vec_dictionary['pos']
        self.vel_decs = ScenarioConfig.obs_vec_dictionary['vel']
        self.num_landmarks = len(self.landmark_uid)

        self.invader_uid = ScenarioConfig.uid_dictionary['invader_uid']

        self.n_entity = ScenarioConfig.num_entity
        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_thread = n_thread
        self.attack_target = [None] * self.n_thread

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def get_previous(self, team_intel):
        info = copy_clone(team_intel['Latest-Obs'])
        Env_Suffered_Reset = copy_clone(team_intel['Env-Suffered-Reset'])
        return info, Env_Suffered_Reset

    def interact_with_env(self, State_Recall):
        main_obs, Env_Suffered_Reset = self.get_previous(State_Recall)
        action = np.ones(shape=(main_obs.shape[0], main_obs.shape[1], 1)) * -1

        n_thread = main_obs.shape[0]
        about_all_objects = main_obs[:,0,:]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim]) # select one agent
        
        invader_emb     = objects_emb[:, self.invader_uid, :]
        landmark_emb    = objects_emb[:, self.landmark_uid,:]

        invader_pos = invader_emb[:, :, self.pos_decs]
        invader_vel = invader_emb[:, :, self.vel_decs]

        landmark_pos = landmark_emb[:, :, self.pos_decs]

        # 为每一个invader设置一个随机目标，当且仅当step == 0 时（episode刚刚开始）
        self.set_nearest_target(Env_Suffered_Reset, invader_pos, landmark_pos)

        n_thread = self.n_thread
        n_agent = self.n_agent
        attack_target =  np.array(self.attack_target)
        action = self.get_action(action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread)

        assert not (action == -1).any()
        actions_list = []
        for i in range(self.n_agent):
            actions_list.append(action[:, i])
        return np.array(actions_list), None


        
    # @jit(nopython=True)
    # @staticmethod
    @jit(forceobj=True)
    def get_action(self, action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread):
        posit_vec = np.zeros_like(invader_vel)
        for thread in range(n_thread):
            for agent in range(n_agent):
                posit_vec[thread,agent] = landmark_pos[thread, attack_target[thread][agent]] - invader_pos[thread, agent]


        return self.dir_to_action3d(vec=posit_vec,vel=invader_vel)



    @staticmethod
    def dir_to_action3d(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        desired_speed = 0.8
        vec = np_mat3d_normalize_each_line(vec)*desired_speed
        return vec




    def set_nearest_target(self, Env_Suffered_Reset, invader_pos, landmark_pos):
        for thread, env_suffered_reset_ in enumerate(Env_Suffered_Reset):
            if env_suffered_reset_:
                invader_attack_target = [None] * self.n_agent
                for i in range(self.n_agent):
                    posit_vec = np.array([landmark_pos[thread, j] - invader_pos[thread, i] for j in range(self.num_landmarks)])
                    dis_arr = np.linalg.norm(posit_vec, axis=-1)
                    assigned_target = np.argmin(dis_arr)
                    # assigned_target = np.random.randint(low=0, high=self.num_landmarks)
                    invader_attack_target[i] = assigned_target
                self.attack_target[thread] = np.array(invader_attack_target)
class DummyAlgorithmFoundationHI3D_old():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        super().__init__()
        self.n_agent = n_agent
        ScenarioConfig = GlobalConfig.ScenarioConfig
        self.num_entity = ScenarioConfig.num_entity
        self.landmark_uid = ScenarioConfig.uid_dictionary['landmark_uid']
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.entity_uid = ScenarioConfig.uid_dictionary['entity_uid']
        self.pos_decs = ScenarioConfig.obs_vec_dictionary['pos']
        self.vel_decs = ScenarioConfig.obs_vec_dictionary['vel']
        self.num_landmarks = len(self.landmark_uid)

        self.invader_uid = ScenarioConfig.uid_dictionary['invader_uid']

        self.n_entity = ScenarioConfig.num_entity
        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_thread = n_thread
        self.attack_target = [None] * self.n_thread

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def get_previous(self, team_intel):
        info = copy_clone(team_intel['Latest-Obs'])
        Env_Suffered_Reset = copy_clone(team_intel['Env-Suffered-Reset'])
        return info, Env_Suffered_Reset

    def interact_with_env(self, State_Recall):
        main_obs, Env_Suffered_Reset = self.get_previous(State_Recall)
        action = np.ones(shape=(main_obs.shape[0], main_obs.shape[1], 1)) * -1

        n_thread = main_obs.shape[0]
        about_all_objects = main_obs[:,0,:]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim]) # select one agent
        
        invader_emb     = objects_emb[:, self.invader_uid, :]
        landmark_emb    = objects_emb[:, self.landmark_uid,:]

        invader_pos = invader_emb[:, :, self.pos_decs]
        invader_vel = invader_emb[:, :, self.vel_decs]

        landmark_pos = landmark_emb[:, :, self.pos_decs]

        # 为每一个invader设置一个随机目标，当且仅当step == 0 时（episode刚刚开始）
        self.set_random_target(Env_Suffered_Reset)

        n_thread = self.n_thread
        n_agent = self.n_agent
        attack_target =  np.array(self.attack_target)
        action = self.get_action(action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread)

        assert not (action == -1).any()
        actions_list = []
        for i in range(self.n_agent):
            actions_list.append(action[:, i])
        return np.array(actions_list), None


        
    # @jit(nopython=True)
    # @staticmethod
    @jit(forceobj=True)
    def get_action(self, action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread):
        posit_vec = np.zeros_like(invader_vel)
        for thread in range(n_thread):
            for agent in range(n_agent):
                posit_vec[thread,agent] = landmark_pos[thread, attack_target[thread][agent]] - invader_pos[thread, agent]


        return self.dir_to_action3d(vec=posit_vec,vel=invader_vel)


    @staticmethod
    @jit(forceobj=True)
    def dir_to_action3d(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        vec = np_mat3d_normalize_each_line(vec)

        e_u = np.array([0  ,1  , 0 ])
        e_d = np.array([0  ,-1 , 0 ])
        e_r = np.array([1  ,0  , 0 ])
        e_l = np.array([-1 ,0  , 0 ])
        e_a = np.array([0  ,0  , 1 ])
        e_b = np.array([0  ,0  ,-1 ])

        vel_u = np_mat3d_normalize_each_line(vel + e_u * 0.1)
        vel_d = np_mat3d_normalize_each_line(vel + e_d * 0.1)
        vel_r = np_mat3d_normalize_each_line(vel + e_r * 0.1)
        vel_l = np_mat3d_normalize_each_line(vel + e_l * 0.1)
        vel_a = np_mat3d_normalize_each_line(vel + e_a * 0.1)
        vel_b = np_mat3d_normalize_each_line(vel + e_b * 0.1)

        proj_u = (vel_u * vec).sum(-1)
        proj_d = (vel_d * vec).sum(-1)
        proj_r = (vel_r * vec).sum(-1)
        proj_l = (vel_l * vec).sum(-1)
        proj_a = (vel_a * vec).sum(-1)
        proj_b = (vel_b * vec).sum(-1)

        _u = ((vec * e_u).sum(-1)>0).astype(np.int)
        _d = ((vec * e_d).sum(-1)>0).astype(np.int)
        _r = ((vec * e_r).sum(-1)>0).astype(np.int)
        _l = ((vec * e_l).sum(-1)>0).astype(np.int)
        _a = ((vec * e_a).sum(-1)>0).astype(np.int)
        _b = ((vec * e_b).sum(-1)>0).astype(np.int)

        proj_u = proj_u + _u*2
        proj_d = proj_d + _d*2
        proj_r = proj_r + _r*2
        proj_l = proj_l + _l*2
        proj_a = proj_a + _a*2
        proj_b = proj_b + _b*2

        dot_stack = np.stack([proj_u, proj_d, proj_r, proj_l, proj_a, proj_b])
        direct = np.argmax(dot_stack, 0)

        action = np.where(direct == 0, 2, 0)
        action += np.where(direct == 1, 4, 0)
        action += np.where(direct == 2, 1, 0)
        action += np.where(direct == 3, 3, 0)

        action += np.where(direct == 4, 5, 0)
        action += np.where(direct == 5, 6, 0)

        return np.expand_dims(action, axis=-1)




    def set_random_target(self, Env_Suffered_Reset):
        for thread, env_suffered_reset_ in enumerate(Env_Suffered_Reset):
            if env_suffered_reset_:
                invader_attack_target = [None] * self.n_agent
                for i in range(self.n_agent):
                    assigned_target = np.random.randint(low=0, high=self.num_landmarks)
                    invader_attack_target[i] = assigned_target
                self.attack_target[thread] = np.array(invader_attack_target)


class IHDummyAlgorithmFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        super().__init__()
        self.n_agent = n_agent
        ScenarioConfig = GlobalConfig.ScenarioConfig
        self.num_entity = ScenarioConfig.num_entity
        self.landmark_uid = ScenarioConfig.uid_dictionary['landmark_uid']
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.invader_uid = ScenarioConfig.uid_dictionary['invader_uid']
        self.n_entity = ScenarioConfig.num_entity
        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_object = ScenarioConfig.num_object
        self.n_thread = n_thread
        self.num_landmarks = ScenarioConfig.num_landmarks
        self.attack_target = [None] * self.n_thread

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def get_previous(self, team_intel):
        info = copy_clone(team_intel['Latest-Obs'])
        done = copy_clone(team_intel['Env-Suffered-Reset'])
        return info, done
    '''
        info, done = self.get_previous(team_intel)
        current_step = info[:,0,-1]
        object_info = my_view(info[:,0,:-1],[0,-1,5])


        worker_emb = object_info[:, self.worker_uid]
        cargo_emb = object_info[:, self.cargo_uid]

        worker_pos = worker_emb[:,:,self.dec_pos]
        worker_vel = worker_emb[:,:,self.dec_vel]
        worker_drag = worker_emb[:,:,self.dec_other]

        cargo_dropoff_pos = cargo_emb[:,:,self.dec_pos]
        cargo_dropoff_weight = cargo_emb[:,:,self.dec_other]

        cargo_pos = cargo_dropoff_pos[:, :self.n_cargo]
        dropoff_pos = cargo_dropoff_pos[:, self.n_cargo:]
        cargo_weight = cargo_dropoff_weight[:, :self.n_cargo]
    '''
    def interact_with_env(self, State_Recall):
        info, done = self.get_previous(State_Recall)
        current_step = info[:,0,-1]
        entity_pure_emb = my_view(info[:,0,:-1],shape=[0,-1,5])

        action = np.ones(shape=(info.shape[0], info.shape[1], 1)) * -1

        entity_pos = entity_pure_emb[:, :, (0,1)]
        entity_vel = entity_pure_emb[:, :, (2,3)]

        invader_vel = entity_vel[:, self.invader_uid]
        invader_pos = entity_pos[:, self.invader_uid]
        landmark_pos = entity_pos[:, self.landmark_uid]

        # 为每一个invader设置一个随机目标，当且仅当step == 0 时（episode刚刚开始）
        self.set_random_target(current_step)

        n_thread = self.n_thread
        n_agent = self.n_agent
        attack_target =  np.array(self.attack_target)
        self.get_action(action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread)

        assert not (action == -1).any()
        actions_list = []
        for i in range(self.n_agent):
            actions_list.append(action[:, i])
        return np.array(actions_list), None

    @staticmethod
    @jit(nopython=True)
    def get_action(action, attack_target, invader_pos, invader_vel, landmark_pos, n_agent, n_thread):
        def Norm(x):
            return np.linalg.norm(x)
        for thread in range(n_thread):
            for agent in range(n_agent):
                speed_vec = invader_vel[thread, agent]
                posit_vec = landmark_pos[thread, attack_target[thread][agent]] - invader_pos[thread, agent]

                posit_norm = Norm(posit_vec)
                if posit_norm != 0:
                    posit_vec = posit_vec / posit_norm

                speed_norm = Norm(speed_vec)
                if speed_norm != 0:
                    speed_vec = speed_vec / speed_norm

                up = np.sum(posit_vec * np.array([0, 1]))
                dn = np.sum(posit_vec * np.array([0, -1]))
                ri = np.sum(posit_vec * np.array([1, 0]))
                le = np.sum(posit_vec * np.array([-1, 0]))

                up_v = np.sum(speed_vec * np.array([0, 1]))
                dn_v = np.sum(speed_vec * np.array([0, -1]))
                ri_v = np.sum(speed_vec * np.array([1, 0]))
                le_v = np.sum(speed_vec * np.array([-1, 0]))

                dot_product = np.array([up, dn, ri, le])
                dot_product_v = np.array([up_v, dn_v, ri_v, le_v])

                # situation 1
                bool_ = (dot_product > dot_product_v) & (dot_product > 0)
                direct = bool_.astype(np.int64)

                if np.sum(direct) != 1:  # 向量重合，或者速度为0，不再对比速度方向
                    direct = np.argmax(dot_product)
                else:
                    # assert sum(direct) == 1 #检查
                    direct = np.argmax(direct)

                # stay_no_acc?[0], left[1], right[2], DOWN[3], Up[4]
                if direct == 0:  # Up
                    action[thread, agent, 0] = 2
                elif direct == 1:  # DOWN
                    action[thread, agent, 0] = 4
                elif direct == 2:  # right
                    action[thread, agent, 0] = 1
                elif direct == 3:  # left
                    action[thread, agent, 0] = 3


    def set_random_target(self, step_env_cnt_cnt):
        for thread, step_env_cnt in enumerate(step_env_cnt_cnt):
            if step_env_cnt == 0:
                invader_attack_target = [None] * self.n_agent
                for i in range(self.n_agent):
                    assigned_target = np.random.randint(low=0, high=self.num_landmarks)
                    invader_attack_target[i] = assigned_target
                self.attack_target[thread] = np.array(invader_attack_target)




class DummyAlgorithmFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        super().__init__()
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.mcv = mcv
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']

        self.n_cargo = GlobalConfig.ScenarioConfig.n_cargo

        self.worker_uid = GlobalConfig.ScenarioConfig.uid_dictionary['agent_uid']
        self.cargo_uid = GlobalConfig.ScenarioConfig.uid_dictionary['entity_uid']

        self.dec_pos = GlobalConfig.ScenarioConfig.obs_vec_dictionary['pos']
        self.dec_vel = GlobalConfig.ScenarioConfig.obs_vec_dictionary['vel']
        self.dec_other = GlobalConfig.ScenarioConfig.obs_vec_dictionary['mass']
        self.vec_len = GlobalConfig.ScenarioConfig.obs_vec_length


    def interact_with_env(self, team_intel):
        info, done = self.get_previous(team_intel)
        current_step = info[:,0,-1]
        object_info = my_view(info[:,0,:-1],[0,-1,self.vec_len])


        worker_emb = object_info[:, self.worker_uid]
        cargo_emb = object_info[:, self.cargo_uid]

        worker_pos = worker_emb[:,:,self.dec_pos]
        worker_vel = worker_emb[:,:,self.dec_vel]
        worker_drag = worker_emb[:,:,self.dec_other]

        cargo_dropoff_pos = cargo_emb[:,:,self.dec_pos]
        cargo_dropoff_weight = cargo_emb[:,:,self.dec_other]

        cargo_pos = cargo_dropoff_pos[:, :self.n_cargo]
        dropoff_pos = cargo_dropoff_pos[:, self.n_cargo:]
        cargo_weight = (cargo_dropoff_weight[:, :self.n_cargo]+1)*(self.n_agent/self.n_cargo)

        worker_target_sel = np.zeros(shape=(self.n_thread,self.n_agent, 1))
        for t in range(self.n_thread):
            p = 0
            for c, cw in enumerate(cargo_weight[t]):
                if cw > self.n_agent: continue
                for j in range(int(p), int(p+cw)):
                    worker_target_sel[t,j] = c if worker_drag[t,j] < 0 else (c+self.n_cargo)
                p = p+cw

        target_pos = np.take_along_axis(cargo_dropoff_pos,worker_target_sel.astype(np.long),1)

        actions_list = []

        act = np.random.randint(low=0,high=5,size=(self.n_thread, self.n_agent, 1))
        act = self.dir_to_action(vec=target_pos-worker_pos, vel=worker_vel)
        for i in range(self.n_agent):
            actions_list.append(act[:, i])
        return actions_list, None

    def get_previous(self, team_intel):
        info = copy_clone(team_intel['Latest-Obs'])
        done = copy_clone(team_intel['Env-Suffered-Reset'])
        return info, done



    @staticmethod
    def dir_to_action(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        vec = np_mat3d_normalize_each_line(vec)

        e_u = np.array([0,1])
        e_d = np.array([0,-1])
        e_r = np.array([1,0])
        e_l = np.array([-1,0])

        vel_u = np_mat3d_normalize_each_line(vel + e_u * 0.1)
        vel_d = np_mat3d_normalize_each_line(vel + e_d * 0.1)
        vel_r = np_mat3d_normalize_each_line(vel + e_r * 0.1)
        vel_l = np_mat3d_normalize_each_line(vel + e_l * 0.1)

        proj_u = (vel_u * vec).sum(-1)
        proj_d = (vel_d * vec).sum(-1)
        proj_r = (vel_r * vec).sum(-1)
        proj_l = (vel_l * vec).sum(-1)

        _u = ((vec * e_u).sum(-1)>0).astype(np.int)
        _d = ((vec * e_d).sum(-1)>0).astype(np.int)
        _r = ((vec * e_r).sum(-1)>0).astype(np.int)
        _l = ((vec * e_l).sum(-1)>0).astype(np.int)

        proj_u = proj_u + _u*2
        proj_d = proj_d + _d*2
        proj_r = proj_r + _r*2
        proj_l = proj_l + _l*2

        dot_stack = np.stack([proj_u, proj_d, proj_r, proj_l])
        direct = np.argmax(dot_stack, 0)

        action = np.where(direct == 0, 2, 0)
        action += np.where(direct == 1, 4, 0)
        action += np.where(direct == 2, 1, 0)
        action += np.where(direct == 3, 3, 0)

        return np.expand_dims(action, axis=-1)
