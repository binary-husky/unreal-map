import os, torch
import numpy as np
from numba import njit, jit
from colorful import *
from gcortex import GNet
from rlcore.algo.ppo import PPO
from z_config import GlobalConfig
from trajectory import BatchTrajManager
from my_utils import copy_clone, my_view, add_onehot_id_at_last_dim, add_obs_container_subject
import pickle
DEBUG = True
class ReinforceAlgorithmFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.n_cluster = GlobalConfig.g_num
        self.n_basic_dim = GlobalConfig.basic_vec_len
        self.n_entity = GlobalConfig.scenario_config.num_entity
        self.agent_uid = GlobalConfig.scenario_config.uid_dictionary['agent_uid']
        self.entity_uid = GlobalConfig.scenario_config.uid_dictionary['entity_uid']
        self.n_object = GlobalConfig.scenario_config.num_object
        self.max_internal_step = GlobalConfig.max_internal_step
        self.head_start_cnt = GlobalConfig.head_start_cnt
        self.decision_interval = GlobalConfig.decision_interval
        self.head_start_hold_n = GlobalConfig.head_start_hold_n
        # self.test_env = GlobalConfig.test_env
        self.device = GlobalConfig.device
        self.policy = GNet(num_agents=self.n_agent, num_entities=self.n_entity).to(self.device)
        # self.policy = torch.nn.DataParallel(self.policy).module
        # print(self.policy)
        self.trainer = PPO(self.policy, mcv=mcv)

        self.batch_traj_manager = BatchTrajManager(n_env=n_thread, traj_limit=GlobalConfig.traj_limit, trainer_hook=self.trainer.train_on_traj)
        self.cold_start = True

        self._division_obsR_init = None
        self._division_obsL_init = None
        self.load_checkpoint = GlobalConfig.load_checkpoint


        note = GlobalConfig.note
        if not os.path.exists('./checkpoint/%s/'%note): 
            os.makedirs('./checkpoint/%s/'%note)
        if self.load_checkpoint:
            print黄('加载检查点')
            self.policy.load_state_dict(torch.load('./checkpoint/%s/model.pt'%note))
        
        
        t = [np.ceil(self.max_internal_step) if x<self.head_start_cnt  else 1.0 if x%self.decision_interval==0 else 0.0  
                for x in range(50)]
        print('control_squence:', t)
        print('hold_squence:', [np.ceil(self.head_start_hold_n / 4**x ) if x<self.head_start_cnt  else 1.0  for x in range(50)])


    def interact_with_env(self, State_Recall):
        self.train()
        return self.action_making(State_Recall) # state_recall dictionary will preserve states for next action making

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
            print('self.decision_interval', self.decision_interval)
            update_cnt = self.batch_traj_manager.train_and_clear_traj_pool()
            self.save_model(update_cnt)

    def get_internal_step(self, n_step):
        #n_internal_step = [np.ceil(self.max_internal_step  / 4**x ) if x<10 else 1.0 for x in n_step]

        n_internal_step = [np.ceil(self.max_internal_step) if x<self.head_start_cnt 
                                else 1.0 if x%self.decision_interval==0 else 0.0  for x in n_step]  # [5, 2, 1, 1, 1, 1, 1
        n_internal_step = np.array(n_internal_step, dtype=np.int)

        hold_n = [np.ceil(self.head_start_hold_n / 4**x ) if x<self.head_start_cnt  else 1.0  for x in n_step]        # [5, 2, 1, 0, 0, 0,
        hold_n = np.array(hold_n, dtype=np.int)

        return n_internal_step, hold_n

    def action_making(self, State_Recall):
        # 使用上一次的traj_frag和刚获取的奖励值，向轨迹中加入新的样本点
        raw_obs, co_step, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L = self.read_loopback(State_Recall)
        just_got_reset = copy_clone(State_Recall['Env-Suffered-Reset'])
        all_emb, act_dec = self.regroup_obs(raw_obs, div_R=subj_div_R, div_L=subj_div_L, g=True)

        # ________RL_Policy_Core_______
        thread_internal_step_o,  hold_n_o = self.get_internal_step(State_Recall['Current-Obs-Step'])
        thread_internal_step = thread_internal_step_o
        iter_n = np.max(thread_internal_step)
        # print紫(thread_internal_step_o[0], hold_n[0], State_Recall['Current-Obs-Step'][0])

        for _ in range(iter_n):
            threads_active_flag = thread_internal_step > 0
            if not threads_active_flag.any():
                continue

            Active_raw_obs = raw_obs[threads_active_flag]
            Active_div_R = subj_div_R[threads_active_flag]
            Active_div_L = subj_div_L[threads_active_flag]
            Active_cter_fifoR = cter_fifoR[threads_active_flag]
            Active_cter_fifoL = cter_fifoL[threads_active_flag]
            hold_n = hold_n_o[threads_active_flag]
            reset = just_got_reset[threads_active_flag]
            Active_emb, Active_act_dec = self.regroup_obs(Active_raw_obs, div_R=Active_div_R, div_L=Active_div_L)
            with torch.no_grad():
                Active_action, Active_value_top, Active_value_bottom, Active_action_log_prob_R, Active_action_log_prob_L = self.policy.act(Active_emb)
            self.batch_traj_manager.feed_traj(
                {   'skip':                 ~threads_active_flag,           'done':                 False,
                    'value_R':              Active_value_top,               'value_L':              Active_value_bottom,
                    'g_actionLogProbs_R':   Active_action_log_prob_R,       'g_actionLogProbs_L':   Active_action_log_prob_L,
                    'g_obs':                Active_emb,                     'g_actions':            Active_action,
                    'ctr_mask_R':           (Active_cter_fifoR < 0).all(2).astype(np.long),  'num_each_cluster_R': (Active_cter_fifoR > 0).sum(2), 
                    'ctr_mask_L':           (Active_cter_fifoL < 0).all(2).astype(np.long), 
                    'reward':               np.zeros_like(Active_value_top)
                }, require_hook=False)

            # _______Internal_Environment_Step________
            container_actR = copy_clone(Active_action[:,(0,1)])
            cluster_entity = Active_action[:,2:]
            # cluster_entity[:, 0] = 0
            # cluster_entity[:, 1] = 1
            # cluster_entity[:, 2] = 2
            # cluster_entity[:, 3] = 3 # --> 8
            # cluster_entity[:, 4] = 4 # --> 9
            # for x, r in enumerate(reset):
            #     if not r:
            #         cluster_entity[x, 3] =  8
            #         cluster_entity[x, 4] =  9

            assert cluster_entity.shape[1] == self.n_cluster

            Active_div_L = cluster_entity
            # container_actL = copy_clone(Active_action[:,(2,3)])
            Active_div_R, Active_cter_fifoR = self.根据动作交换组成员(container_actR, div=Active_div_R, fifo=Active_cter_fifoR, hold_n=hold_n)
            # Active_div_L, Active_cter_fifoL = self.根据动作交换组成员(container_actL, div=Active_div_L, fifo=Active_cter_fifoL)
            # if threads_active_flag[0]: print绿(Active_div_R[0], Active_div_L[0], Active_action[0])
            subj_div_R[threads_active_flag] = Active_div_R
            cter_fifoR[threads_active_flag] = Active_cter_fifoR
            subj_div_L[threads_active_flag] = Active_div_L
            cter_fifoL[threads_active_flag] = Active_cter_fifoL
            thread_internal_step = thread_internal_step - 1

        traj_frag = {
            'skip': False, 'g_obs': None, 'value_R': None,'value_L': None, 'g_actions': None, 
            'g_actionLogProbs_R': None, 'g_actionLogProbs_L': None, 'ctr_mask_R': None, 'ctr_mask_L': None, 'num_each_cluster_R':None
        }

        delta_pos, agent_entity_div = self.组成员目标分配(subj_div_R, subj_div_L, act_dec)
        # if thread_internal_step_o[0] > 0: print绿(agent_entity_div[0])
        all_action = self.dir_to_action(vec=delta_pos, vel=act_dec['agent_vel']) # 矢量指向selected entity
        actions_list = []
        for i in range(self.n_agent): actions_list.append(all_action[:,i,:])


        # return necessary handles to main platform
        traj_hook = self.batch_traj_manager.feed_traj(traj_frag, require_hook=True)
        State_Recall['_hook_'] = traj_hook  # leave a hook to grab the reward signal just a moment later
        State_Recall = self.loopback_state(State_Recall, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L)
        if self.cold_start: self.cold_start = False
        return actions_list, State_Recall # state_recall dictionary will preserve states for next action making




    def read_loopback(self, State_Recall):
        _n_cluster = self.n_cluster if not GlobalConfig.one_more_container else self.n_cluster+1
        n_container_R = _n_cluster
        n_subject_R = self.n_agent
        n_container_L = self.n_entity if not GlobalConfig.reverse_container else _n_cluster
        n_subject_L = _n_cluster if not GlobalConfig.reverse_container else self.n_entity

        if self.cold_start:
            assert not '_division_obsR_' in State_Recall  # 第一次运行
            State_Recall['_division_obsR_'] = np.zeros(shape=(self.n_thread, n_subject_R), dtype=np.long)
            State_Recall['_division_obsL_'] = np.zeros(shape=(self.n_thread, n_subject_L), dtype=np.long)
            State_Recall['_division_fifoR_'] = np.ones(shape=(self.n_thread, n_container_R, n_subject_R), dtype=np.long) * -1
            State_Recall['_division_fifoL_'] = np.ones(shape=(self.n_thread, n_container_L, n_subject_L), dtype=np.long) * -1

        raw_obs = copy_clone(State_Recall['Latest-Obs'])
        just_got_reset = copy_clone(State_Recall['Env-Suffered-Reset'])
        co_step = copy_clone(State_Recall['Current-Obs-Step'])
        self.处理组成员初始化(just_got_reset, State_Recall)
        # if a var named with with _x_ format, it will loop back at next iteration
        subj_div_R = copy_clone(State_Recall['_division_obsR_'])  
        subj_div_L = copy_clone(State_Recall['_division_obsL_'])  
        cter_fifoR = copy_clone(State_Recall['_division_fifoR_'])
        cter_fifoL = copy_clone(State_Recall['_division_fifoL_'])
        return raw_obs, co_step, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L

    @staticmethod
    def loopback_state(State_Recall, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L):
        # -----------------load state  ---------------#
        State_Recall['_division_obsR_'] = subj_div_R  # overwrite
        State_Recall['_division_fifoR_'] = cter_fifoR  # overwrite
        State_Recall['_division_obsL_'] = subj_div_L  # overwrite
        State_Recall['_division_fifoL_'] = cter_fifoL  # overwrite
        
        return State_Recall

    
    @staticmethod
    def __random_select_init_value_(n_container, n_subject):
        t_final = []; entropy = np.array([])
        for _ in range(20): # max entropy in samples
            tmp = np.random.randint(low=0, high=n_container, size=(n_subject,), dtype=np.long); t_final.append(tmp)
            entropy = np.append(entropy, sum([ -(sum(tmp==i)/n_subject)*np.log(sum(tmp==i)/n_subject) if sum(tmp==i)!=0 else -np.inf for i in range(n_container)]))
        return t_final[np.argmax(entropy)]


    def 处理组成员初始化(self, just_got_reset, State_Recall):

        _n_cluster = self.n_cluster if not GlobalConfig.one_more_container else self.n_cluster+1
        n_container_R = _n_cluster
        n_subject_R = self.n_agent
        n_container_L = self.n_entity if not GlobalConfig.reverse_container else _n_cluster
        n_subject_L = _n_cluster if not GlobalConfig.reverse_container else self.n_entity
 
        # fixed random init
        if self.cold_start and GlobalConfig.use_fixed_random_start:
            assert self._division_obsR_init is None
            note = GlobalConfig.note
            if self.load_checkpoint:
                pkl_file = open('./checkpoint/%s/init.pkl'%note, 'rb')
                dict_data = pickle.load(pkl_file)
                self._division_obsR_init = dict_data["_division_obsR_init"]
                self._division_obsL_init = dict_data["_division_obsL_init"]
            else:
                self._division_obsR_init = self.__random_select_init_value_(n_container_R, n_subject_R)
                self._division_obsL_init = self.__random_select_init_value_(n_container_L, n_subject_L)
                pickle.dump({"_division_obsR_init":self._division_obsR_init,\
                             "_division_obsL_init":self._division_obsL_init}, open('./checkpoint/%s/init.pkl'%note,'wb+'))

        for procindex in range(self.n_thread):
            if not just_got_reset[procindex]: continue # otherwise reset
            if GlobalConfig.use_zero_start:
                _division_obsR_= np.zeros(shape=(n_subject_R, ), dtype=np.long)
                _division_obsL_= np.zeros(shape=(n_subject_L, ), dtype=np.long)
            elif GlobalConfig.use_fixed_random_start:
                assert self._division_obsR_init is not None
                _division_obsR_ = self._division_obsR_init
                _division_obsL_ = self._division_obsL_init
            elif GlobalConfig.use_empty_container:
                # 最后一个容器是全体，其他的全空
                _division_obsR_= np.ones(shape=(n_subject_R, ), dtype=np.long) * (n_container_R - 1)
                _division_obsL_= np.ones(shape=(n_subject_L, ), dtype=np.long) * (n_container_L - 1)
            elif GlobalConfig.use_complete_random:
                _division_obsR_ = self.__random_select_init_value_(n_container_R, n_subject_R)
                _division_obsL_ = self.__random_select_init_value_(n_container_L, n_subject_L)
                if procindex == 0: print绿(_division_obsR_)
            else:
                assert False

            State_Recall['_division_obsR_'][procindex,:] = _division_obsR_
            State_Recall['_division_obsL_'][procindex,:] = _division_obsL_
            for container in range(n_container_R):
                State_Recall['_division_fifoR_'][procindex,container] = np.ones(n_subject_R) *-1
                index_ = np.where(_division_obsR_ == container)[0]
                State_Recall['_division_fifoR_'][procindex,container,:len(index_)] = index_
            for container in range(n_container_L):
                State_Recall['_division_fifoL_'][procindex,container] = np.ones(n_subject_L) *-1
                index_ = np.where(_division_obsL_ == container)[0]
                State_Recall['_division_fifoL_'][procindex,container,:len(index_)] = index_
            pass

            if procindex == 0:
                print('!!!初始化分布:',(State_Recall['_division_fifoR_'][procindex] > 0).sum(1))
        pass

    def save_model(self, update_cnt):
        note = GlobalConfig.note
        if os.path.exists('./checkpoint/%s/save_now.txt'%note) or update_cnt%100==99:
            print绿('保存模型中')
            torch.save(self.policy.state_dict(), './checkpoint/%s/model.pt'%note)
            print绿('保存模型完成')
            try:
                os.remove('./checkpoint/%s/save_now.txt'%note)
            except:
                print('清除保存标志:异常')
        # if update_cnt%200==199:
        #     if self.decision_interval > 20: 
        #         self.decision_interval -= 10
        #     if self.decision_interval< 10:
        #         self.decision_interval = 10

    def 组成员目标分配(self, agent_cluster_div, cluster_entity_div, act_dec):
        agent_cluster_div_ = agent_cluster_div.copy()
        cluster_entity_div = cluster_entity_div.copy()   # 每个cluster在哪个entity容器中
        n_thread = agent_cluster_div.shape[0]
        entity_pos, agent_pos = (act_dec['entity_pos'], act_dec['agent_pos'])

        if DEBUG:

            # part 1
            # worker_target_sel = np.zeros(shape=(n_thread,self.n_agent, 1))
            # for t in range(n_thread):
            #     p = 0
            #     for c in range(self.n_cargo):
            #         if self.cargo_lifted[t,c]:
            #             cluster_entity_div[t,c] = c+self.n_cargo
            #         else:
            #             cluster_entity_div[t,c] = c
            #     cluster_entity_div[t,self.n_cargo] = 1+self.n_cargo
            # part 2


            worker_target_sel = np.zeros(shape=(n_thread,self.n_agent, 1))
            for t in range(n_thread):
                p = 0
                for c, cw in enumerate(self.cargo_weight[t]):
                    if cw > self.n_agent: continue
                    for j in range(int(p), int(p+cw)):
                        agent_cluster_div_[t,j] = c
                    p = p+cw



        agent_entity_div = np.take_along_axis(cluster_entity_div, axis=1, indices=agent_cluster_div_)
        final_indices = np.expand_dims(agent_entity_div, axis=-1).repeat(2, axis=-1)
        if not GlobalConfig.reverse_container:
            final_sel_pos = entity_pos
        else:   # 为没有装入任何entity的container解析一个nan动作
            final_sel_pos = np.concatenate( (entity_pos,  np.zeros(shape=(n_thread, 1, 2))+np.nan ) , axis=1)
            
        sel_entity_pos  = np.take_along_axis(final_sel_pos, axis=1, indices=final_indices)  # 6 in final_indices /cluster_entity_div
        delta_pos = sel_entity_pos - agent_pos
        return delta_pos, agent_entity_div

    @staticmethod
    @njit
    def 根据动作交换组成员(act, div, fifo, hold_n=None):
        def push(vec, item):
            insert_pos=0; len_vec = len(vec)
            while insert_pos<len_vec and vec[insert_pos]!=-1: insert_pos+=1
            assert insert_pos < len_vec
            vec[insert_pos] = item
        def pop(vec):
            p = vec[0]; assert p>=0
            vec[:-1]=vec[1:]; vec[-1] = -1
            return p
        n_thread = act.shape[0]
        if hold_n is None:
            hold_n = np.ones((n_thread,), np.int_)
        act_switch_1 = act[:, 0]
        act_switch_2 = act[:, 1]
        for 目标组, 移除组, i in zip(act_switch_1, act_switch_2, range(n_thread)):
            if 目标组 == 移除组:  continue
            else:
                for _ in range(hold_n[i]):     # check this       
                    移除组智能体成员 = np.where(div[i] == 移除组)[0]
                    if len(移除组智能体成员) == 0: continue  # 已经是空组别
                    转移体 = pop(fifo[i, 移除组])
                    div[i, 转移体] = 目标组
                    push(fifo[i, 目标组], 转移体)
        new_div = div
        new_fifo = fifo

        return new_div, new_fifo



    def regroup_obs(self,  main_obs, div_R, div_L, g=False):
        _n_cluster = self.n_cluster if not GlobalConfig.one_more_container else self.n_cluster+1

        n_thread = main_obs.shape[0]
        about_all_objects = main_obs[:,0,:self.n_object*self.n_basic_dim]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim])
        agent_pure_emb = objects_emb[:,self.agent_uid,:]
        entity_pure_emb = objects_emb[:,self.entity_uid,:]
        cluster_pure_emb = np.zeros(shape=(n_thread, _n_cluster, 0)) # empty

        agent_hot_emb = add_onehot_id_at_last_dim(agent_pure_emb)
        entity_hot_emb = add_onehot_id_at_last_dim(entity_pure_emb)
        cluster_hot_emb = add_onehot_id_at_last_dim(cluster_pure_emb)

        cluster_hot_emb, agent_hot_emb  = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=agent_hot_emb, div=div_R)
        if not GlobalConfig.reverse_container:
            entity_hot_emb, cluster_hot_emb = add_obs_container_subject(container_emb=entity_hot_emb, subject_emb=cluster_hot_emb, div=div_L)
        else:
            cluster_hot_emb, entity_hot_emb = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=entity_hot_emb, div=div_L)

        agent_final_emb = agent_hot_emb
        entity_final_emb = entity_hot_emb
        cluster_final_emb = cluster_hot_emb

        agent_emb  = objects_emb[:, self.agent_uid, :]
        agent_pos  = agent_emb[:, :, (0,1)]
        agent_vel  = agent_emb[:, :, (2,3)]
        entity_pos = entity_pure_emb[:, :, (0,1)]
        entity_vel = entity_pure_emb[:, :, (2,3)]

        all_emb = {
            'agent_final_emb':agent_final_emb,      # for RL
            'entity_final_emb': entity_final_emb,   # for RL
            'cluster_final_emb': cluster_final_emb, # for RL

        }
        act_dec = {
            'agent_pos': agent_pos,  # for decoding action
            'agent_vel': agent_vel,  # for decoding action
            'entity_pos': entity_pos,  # for decoding action
            'entity_vel': entity_vel  # for decoding action
        }



        if DEBUG and g:
            objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim])

            self.cargo_uid = GlobalConfig.scenario_config.uid_dictionary['entity_uid']
            self.worker_uid = GlobalConfig.scenario_config.uid_dictionary['agent_uid']
            self.dec_pos = GlobalConfig.scenario_config.obs_vec_dictionary['pos']
            self.dec_mass = GlobalConfig.scenario_config.obs_vec_dictionary['mass']
            self.dec_other = GlobalConfig.scenario_config.obs_vec_dictionary['other']
            self.n_cargo = GlobalConfig.scenario_config.n_cargo


            cargo_emb = objects_emb[:, self.cargo_uid]
            worker_emb = objects_emb[:, self.worker_uid]


            cargo_dropoff_pos = cargo_emb[:,:,self.dec_pos]
            cargo_dropoff_weight = cargo_emb[:,:,self.dec_mass]

            self.cargo_lifted = (cargo_emb[:,:,self.dec_other][:,:self.n_cargo] == 0)

            self.worker_drag = worker_emb[:,:,self.dec_mass]
            self.cargo_weight = (cargo_dropoff_weight[:, :self.n_cargo]+1)*(self.n_agent/self.n_cargo)













        return all_emb, act_dec



    @staticmethod
    @jit(forceobj=True)
    def dir_to_action(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)

        dis2target = np.linalg.norm(vec, axis=2)
        vec = np_mat3d_normalize_each_line(vec) #self.step

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

        action = (dis2target>0.05).astype(np.int)*action

        # make sure that all nan vec become invalid act 0, 
        # be careful when a different numpy version is used
        assert (action[np.isnan(np.sum(dot_stack,0))] == 0).all()
        # action *= 0
        return np.expand_dims(action, axis=-1)

