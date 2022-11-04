import os, torch
import numpy as np
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit
from UTIL.colorful import *
from .gcortex import GNet
from .ppo import PPO
from .trajectory import BatchTrajManager
from .my_utils import copy_clone, my_view, add_onehot_id_at_last_dim, add_obs_container_subject
from UTIL.tensor_ops import __hash__
import pickle

class CoopAlgConfig(object):
    g_num = 5
    max_internal_step = 15
    decision_interval = 5
    head_start_cnt = 1 # first 3 step have 
    head_start_hold_n = 1 # how many to control at first few step

    eval_mode = False

    checkpoint_reload_cuda = False
    load_checkpoint = False
    load_specific_checkpoint = ''
    one_more_container = False
    reverse_container = False
    use_fixed_random_start = True
    use_zero_start = False 
    use_empty_container = False
    use_complete_random = False

    # PPO part
    clip_param = 0.2
    ppo_epoch = 4
    n_pieces_batch_division = 32    # 8: the batch size in each ppo update is 23280; x/8 *1.5 = x/y, y=8/1.5
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4
    balance = 0.5
    
    gamma = 0.99
    tau = 0.95
    # ?
    train_traj_needed = 128
    upper_training_epoch = 5
    h_reward_on_R = True
    continues_type_ceil = True
    invalid_penalty = 0.1
    upper_training_epoch = 5
    use_normalization = True
    
class ReinforceAlgorithmFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.n_cluster = CoopAlgConfig.g_num
        ScenarioConfig = GlobalConfig.ScenarioConfig
        self.note = GlobalConfig.note

        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_entity = ScenarioConfig.num_entity
        self.ObsAsUnity = False
        if hasattr(ScenarioConfig, 'ObsAsUnity'):
            self.ObsAsUnity = ScenarioConfig.ObsAsUnity
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.entity_uid = ScenarioConfig.uid_dictionary['entity_uid']

        self.pos_decs = ScenarioConfig.obs_vec_dictionary['pos']
        self.vel_decs = ScenarioConfig.obs_vec_dictionary['vel']
        self.max_internal_step = CoopAlgConfig.max_internal_step
        self.head_start_cnt = CoopAlgConfig.head_start_cnt
        self.decision_interval = CoopAlgConfig.decision_interval
        self.head_start_hold_n = CoopAlgConfig.head_start_hold_n

        self.device = GlobalConfig.device
        cuda_n = 'cpu' if 'cpu' in self.device else GlobalConfig.device

        self.policy = GNet(num_agents=self.n_agent, num_entities=self.n_entity, basic_vec_len=self.n_basic_dim).to(self.device)
        self.trainer = PPO(self.policy, mcv=mcv)

        self.batch_traj_manager = BatchTrajManager(n_env=n_thread, traj_limit=ScenarioConfig.MaxEpisodeStep*3, trainer_hook=self.trainer.train_on_traj)

        self._division_obsR_init = None
        self._division_obsL_init = None
        self.load_checkpoint = CoopAlgConfig.load_checkpoint
        self.cnt = 0

        self.logdir = GlobalConfig.logdir
        if not os.path.exists('%s/history_cpt/'%self.logdir): os.makedirs('%s/history_cpt/'%self.logdir)
        if self.load_checkpoint:
            manual_dir = CoopAlgConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt'%self.logdir if manual_dir=='' else '%s/%s'%(self.logdir, manual_dir)
            print黄('加载检查点:', ckpt_dir)
            if not CoopAlgConfig.checkpoint_reload_cuda:
                self.policy.load_state_dict(torch.load(ckpt_dir))
            else:
                self.policy.load_state_dict(torch.load(ckpt_dir, map_location=cuda_n))
        
        
        t = [int(np.ceil(self.max_internal_step)) if x<self.head_start_cnt  else 1 if x%self.decision_interval==0 else 0
                for x in range(50)]
        print('control_squence:', t)
        print('hold_squence:', [int(np.ceil(self.head_start_hold_n / 4**x )) if x<self.head_start_cnt  else 1  for x in range(50)])
        self.patience = 500 # skip currupt data detection after patience exhausted


    def interact_with_env(self, State_Recall):
        self.train()
        return self.action_making(State_Recall) # state_recall dictionary will preserve states for next action making

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
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
        # print亮紫(self.cnt); self.cnt+=1
        test_mode = State_Recall['Test-Flag']
        active_env = ~State_Recall['ENV-PAUSE']
        # 使用上一次的traj_frag和刚获取的奖励值，向轨迹中加入新的样本点
        raw_obs, co_step, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L = self.read_loopback(State_Recall)
        all_emb, act_dec = self.regroup_obs(raw_obs, div_R=subj_div_R, div_L=subj_div_L)

        # ________RL_Policy_Core_______
        thread_internal_step_o,  hold_n_o = self.get_internal_step(State_Recall['Current-Obs-Step'])
        thread_internal_step = thread_internal_step_o
        iter_n = np.max(thread_internal_step)
        
        for _ in range(iter_n):
            threads_active_flag = active_env & (thread_internal_step > 0)
            if not threads_active_flag.any():
                continue

            Active_raw_obs = raw_obs[threads_active_flag]
            Active_div_R = subj_div_R[threads_active_flag]
            Active_div_L = subj_div_L[threads_active_flag]
            Active_cter_fifoR = cter_fifoR[threads_active_flag]
            Active_cter_fifoL = cter_fifoL[threads_active_flag]
            hold_n = hold_n_o[threads_active_flag]
            
            Active_emb, Active_act_dec = self.regroup_obs(Active_raw_obs, div_R=Active_div_R, div_L=Active_div_L)
            with torch.no_grad():
                Active_action, Active_value_top, Active_value_bottom, Active_action_log_prob_R, Active_action_log_prob_L = self.policy.act(Active_emb, test_mode=test_mode)
            traj_frag = {   'skip':                 ~threads_active_flag,           'done':                 False,
                            'value_R':              Active_value_top,               'value_L':              Active_value_bottom,
                            'g_actionLogProbs_R':   Active_action_log_prob_R,       'g_actionLogProbs_L':   Active_action_log_prob_L,
                            'g_obs':                Active_emb,                     'g_actions':            Active_action,
                            'ctr_mask_R':           (Active_cter_fifoR < 0).all(2).astype(np.long),
                            'ctr_mask_L':           (Active_cter_fifoL < 0).all(2).astype(np.long),
                            'reward':               np.array([0.0 for _ in range(self.n_thread)])}
            # _______Internal_Environment_Step________
            container_actR = copy_clone(Active_action[:,(0,1)])
            container_actL = copy_clone(Active_action[:,(2,3)])
            Active_div_R, Active_cter_fifoR = self.根据动作交换组成员(container_actR, div=Active_div_R, fifo=Active_cter_fifoR, hold_n=hold_n)
            Active_div_L, Active_cter_fifoL = self.根据动作交换组成员(container_actL, div=Active_div_L, fifo=Active_cter_fifoL)
            # if threads_active_flag[0]: print红(Active_div_R[0], Active_div_L[0], Active_action[0])
            subj_div_R[threads_active_flag] = Active_div_R
            cter_fifoR[threads_active_flag] = Active_cter_fifoR
            subj_div_L[threads_active_flag] = Active_div_L
            cter_fifoL[threads_active_flag] = Active_cter_fifoL
            if not test_mode: self.batch_traj_manager.feed_traj(traj_frag, require_hook=False)
            thread_internal_step = thread_internal_step - 1

        traj_frag = {
            'skip': copy_clone(State_Recall['ENV-PAUSE']), 'g_obs': None, 'value_R': None,'value_L': None, 'g_actions': None,
            'g_actionLogProbs_R': None, 'g_actionLogProbs_L': None, 'ctr_mask_R': None, 'ctr_mask_L': None,
        }

        delta_pos, target_vel, agent_entity_div = self.组成员目标分配(subj_div_R, subj_div_L, cter_fifoR, cter_fifoL, act_dec)
        # if thread_internal_step_o[0] > 0: print红(agent_entity_div[0])
        all_action = self.dir_to_action3d(vec=delta_pos, vel=target_vel) # 矢量指向selected entity
        actions_list = []
        for i in range(self.n_agent): actions_list.append(all_action[:,i,:])
        actions_list = np.array(actions_list)

        # return necessary handles to main platform
        wait_reward_hook = self.commit_frag_hook(traj_frag, require_hook = True) if not test_mode else self.__dummy_hook
        # traj_hook = self.batch_traj_manager.feed_traj(traj_frag, require_hook=True) if not test_mode else self.__dummy_hook
        State_Recall['_hook_'] = wait_reward_hook  # leave a hook to grab the reward signal just a moment later
        State_Recall = self.loopback_state(State_Recall, cter_fifoR, subj_div_R, cter_fifoL, subj_div_L)
        # if self.cold_start: self.cold_start = False
        return actions_list, State_Recall # state_recall dictionary will preserve states for next action making

    def commit_frag_hook(self, f1, require_hook = True):
        if not hasattr(self, '__incomplete_frag__'): self.__incomplete_frag__ = None
        assert self.__incomplete_frag__ is None
        self.__incomplete_frag__ = f1
        self.__check_data_hash() # this is important!
        if require_hook: return lambda f2: self.rollout_frag_hook(f2) # leave hook
        return

    def rollout_frag_hook(self, f2):
        '''   <2>  hook is called, reward and next moment observation is ready,
                        now feed them into trajectory manager    '''
        # do data curruption check at beginning, this is important!
        self.__check_data_curruption()
        # put the fragment into memory
        self.__incomplete_frag__.update(f2)
        __completed_frag = self.__incomplete_frag__
        __completed_frag.pop('info')
        __completed_frag.pop('Latest-Obs')
        __completed_frag.pop('Terminal-Obs-Echo')
        self.batch_traj_manager.feed_traj(__completed_frag, require_hook=False)
        self.__incomplete_frag__ = None

    def __dummy_hook(self, f2): 
        return

    def read_loopback(self, State_Recall):
        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1
        n_container_R = _n_cluster
        n_subject_R = self.n_agent
        n_container_L = self.n_entity if not CoopAlgConfig.reverse_container else _n_cluster
        n_subject_L = _n_cluster if not CoopAlgConfig.reverse_container else self.n_entity

        raw_obs = copy_clone(State_Recall['Latest-Obs'])
        just_got_reset = copy_clone(State_Recall['Env-Suffered-Reset'])
        co_step = copy_clone(State_Recall['Current-Obs-Step'])

        if '_division_obsR_' not in State_Recall: # 算法初始化时，第一次测试环境时
            State_Recall['_division_obsR_'] = np.zeros(shape=(self.n_thread, n_subject_R), dtype=np.long)
            State_Recall['_division_obsL_'] = np.zeros(shape=(self.n_thread, n_subject_L), dtype=np.long)
            State_Recall['_division_fifoR_'] = np.ones(shape=(self.n_thread, n_container_R, n_subject_R), dtype=np.long) * -1
            State_Recall['_division_fifoL_'] = np.ones(shape=(self.n_thread, n_container_L, n_subject_L), dtype=np.long) * -1

        if self._division_obsR_init is None: # 仅算法初始化时
            self.处理组成员初始化(just_got_reset, State_Recall, init=True)

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


    def 处理组成员初始化(self, just_got_reset, State_Recall, init=False):

        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1
        n_container_R = _n_cluster
        n_subject_R = self.n_agent
        n_container_L = self.n_entity if not CoopAlgConfig.reverse_container else _n_cluster
        n_subject_L = _n_cluster if not CoopAlgConfig.reverse_container else self.n_entity
 
        # fixed random init
        if init and CoopAlgConfig.use_fixed_random_start:
            assert self._division_obsR_init is None
            if self.load_checkpoint:
                pkl_file = open('%s/history_cpt/init.pkl'%self.logdir, 'rb')
                dict_data = pickle.load(pkl_file)
                self._division_obsR_init = dict_data["_division_obsR_init"]
                self._division_obsL_init = dict_data["_division_obsL_init"]
            else:
                self._division_obsR_init = self.__random_select_init_value_(n_container_R, n_subject_R)
                self._division_obsL_init = self.__random_select_init_value_(n_container_L, n_subject_L)
                pickle.dump({"_division_obsR_init":self._division_obsR_init,\
                             "_division_obsL_init":self._division_obsL_init}, open('%s/history_cpt/init.pkl'%self.logdir,'wb+'))

        for procindex in range(self.n_thread):
            if not just_got_reset[procindex]: continue # otherwise reset
            if CoopAlgConfig.use_zero_start:
                _division_obsR_= np.zeros(shape=(n_subject_R, ), dtype=np.long)
                _division_obsL_= np.zeros(shape=(n_subject_L, ), dtype=np.long)
            elif CoopAlgConfig.use_fixed_random_start:
                assert self._division_obsR_init is not None
                _division_obsR_ = self._division_obsR_init
                _division_obsL_ = self._division_obsL_init
            elif CoopAlgConfig.use_empty_container:
                # 最后一个容器是全体，其他的全空
                _division_obsR_= np.ones(shape=(n_subject_R, ), dtype=np.long) * (n_container_R - 1)
                _division_obsL_= np.ones(shape=(n_subject_L, ), dtype=np.long) * (n_container_L - 1)
            elif CoopAlgConfig.use_complete_random:
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

            # if procindex == 0:
                # print('!!!初始化分布:',(State_Recall['_division_fifoR_'][procindex] > 0).sum(1))
        pass

    def save_model(self, update_cnt):
        if update_cnt!=0 and update_cnt%200==0:
            print绿('保存模型中')
            torch.save(self.policy.state_dict(), '%s/history_cpt/model%d.pt'%(self.logdir, update_cnt))
            torch.save(self.policy.state_dict(), '%s/model.pt'%(self.logdir))
            print绿('保存模型完成')

    def 组成员目标分配(self, agent_cluster_div, div2, cter_fifoR, cter_fifoL, act_dec):
        entity_pos, agent_pos, target_vel = (act_dec['entity_pos'], act_dec['agent_pos'], act_dec['entity_vel'])
        n_thread = agent_cluster_div.shape[0]
        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1

        if not CoopAlgConfig.reverse_container:
            cluster_entity_div = div2   # 每个cluster在哪个entity容器中
        else:   # figure out cluster_entity_div with fifo # 每个cluster指向那个entity
            cluster_entity_div = np.ones(shape=(n_thread, _n_cluster), dtype=np.long) * self.n_entity #point to n_entity+1
            for thread, jth_cluster, pos in  np.argwhere(cter_fifoL >= 0):
                cluster_entity_div[thread, jth_cluster] = cter_fifoL[thread, jth_cluster, pos]    # 指向队列中的最后一个目标
            if CoopAlgConfig.one_more_container: 
                cluster_entity_div[:,self.n_cluster] = self.n_entity

        agent_entity_div = np.take_along_axis(cluster_entity_div, axis=1, indices=agent_cluster_div)
        final_indices = np.expand_dims(agent_entity_div, axis=-1).repeat(3, axis=-1)
        if not CoopAlgConfig.reverse_container:
            final_sel_pos = entity_pos
        else:   # 为没有装入任何entity的container解析一个nan动作
            final_sel_pos = np.concatenate( (entity_pos,  np.zeros(shape=(n_thread, 1, 3))+np.nan ) , axis=1)
            
        sel_entity_pos  = np.take_along_axis(final_sel_pos, axis=1, indices=final_indices)  # 6 in final_indices /cluster_entity_div
        sel_target_vel  = np.take_along_axis(target_vel, axis=1, indices=final_indices)  # 6 in final_indices /cluster_entity_div
        delta_pos = sel_entity_pos - agent_pos
        return delta_pos, sel_target_vel, agent_entity_div

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
        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1
        if self.ObsAsUnity:
            about_all_objects = main_obs
        else:
            about_all_objects = main_obs[:,0,:]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim]) # select one agent

        agent_pure_emb = objects_emb[:,self.agent_uid,:]
        entity_pure_emb = objects_emb[:,self.entity_uid,:]
        
        n_thread = main_obs.shape[0]
        cluster_pure_emb = np.zeros(shape=(n_thread, _n_cluster, 0)) # empty

        agent_hot_emb = add_onehot_id_at_last_dim(agent_pure_emb)
        entity_hot_emb = add_onehot_id_at_last_dim(entity_pure_emb)
        cluster_hot_emb = add_onehot_id_at_last_dim(cluster_pure_emb)
        cluster_hot_emb, agent_hot_emb  = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=agent_hot_emb, div=div_R)
        if not CoopAlgConfig.reverse_container:
            entity_hot_emb, cluster_hot_emb = add_obs_container_subject(container_emb=entity_hot_emb, subject_emb=cluster_hot_emb, div=div_L)
        else:
            cluster_hot_emb, entity_hot_emb = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=entity_hot_emb, div=div_L)

        agent_final_emb = agent_hot_emb
        entity_final_emb = entity_hot_emb
        cluster_final_emb = cluster_hot_emb

        agent_emb  = objects_emb[:, self.agent_uid, :]
        agent_pos  = agent_emb[:, :, self.pos_decs]
        agent_vel  = agent_emb[:, :, self.vel_decs]
        entity_pos = entity_pure_emb[:, :, self.pos_decs]
        entity_vel = entity_pure_emb[:, :, self.vel_decs]

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



    @staticmethod
    # @jit(forceobj=True)
    def dir_to_action3d(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        # desired_speed = 0.8
        vec_dx = np_mat3d_normalize_each_line(vec)
        vec_dv = np_mat3d_normalize_each_line(vel)*0.8
        vec = np_mat3d_normalize_each_line(vec_dx+vec_dv)
        return vec

        # def np_mat3d_normalize_each_line(mat):
        #     return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        # vec = np_mat3d_normalize_each_line(vec)

        # e_u = np.array([0  ,1  , 0 ])
        # e_d = np.array([0  ,-1 , 0 ])
        # e_r = np.array([1  ,0  , 0 ])
        # e_l = np.array([-1 ,0  , 0 ])
        # e_a = np.array([0  ,0  , 1 ])
        # e_b = np.array([0  ,0  ,-1 ])

        # vel_u = np_mat3d_normalize_each_line(vel + e_u * 0.1)
        # vel_d = np_mat3d_normalize_each_line(vel + e_d * 0.1)
        # vel_r = np_mat3d_normalize_each_line(vel + e_r * 0.1)
        # vel_l = np_mat3d_normalize_each_line(vel + e_l * 0.1)
        # vel_a = np_mat3d_normalize_each_line(vel + e_a * 0.1)
        # vel_b = np_mat3d_normalize_each_line(vel + e_b * 0.1)

        # proj_u = (vel_u * vec).sum(-1)
        # proj_d = (vel_d * vec).sum(-1)
        # proj_r = (vel_r * vec).sum(-1)
        # proj_l = (vel_l * vec).sum(-1)
        # proj_a = (vel_a * vec).sum(-1)
        # proj_b = (vel_b * vec).sum(-1)

        # _u = ((vec * e_u).sum(-1)>0).astype(np.int)
        # _d = ((vec * e_d).sum(-1)>0).astype(np.int)
        # _r = ((vec * e_r).sum(-1)>0).astype(np.int)
        # _l = ((vec * e_l).sum(-1)>0).astype(np.int)
        # _a = ((vec * e_a).sum(-1)>0).astype(np.int)
        # _b = ((vec * e_b).sum(-1)>0).astype(np.int)

        # proj_u = proj_u + _u*2
        # proj_d = proj_d + _d*2
        # proj_r = proj_r + _r*2
        # proj_l = proj_l + _l*2
        # proj_a = proj_a + _a*2
        # proj_b = proj_b + _b*2

        # dot_stack = np.stack([proj_u, proj_d, proj_r, proj_l, proj_a, proj_b])
        # direct = np.argmax(dot_stack, 0)

        # action = np.where(direct == 0, 2, 0)
        # action += np.where(direct == 1, 4, 0)
        # action += np.where(direct == 2, 1, 0)
        # action += np.where(direct == 3, 3, 0)

        # action += np.where(direct == 4, 5, 0)
        # action += np.where(direct == 5, 6, 0)

        # return np.expand_dims(action, axis=-1)

    # debugging functions
    def __check_data_hash(self):
        if self.patience > 0: 
            self.hash_debug = {}
            # for debugging, to detect write protection error
            for key in self.__incomplete_frag__:
                item = self.__incomplete_frag__[key]
                if isinstance(item, dict):
                    self.hash_debug[key]={}
                    for subkey in item:
                        subitem = item[subkey]
                        self.hash_debug[key][subkey] = __hash__(subitem)
                else:
                    self.hash_debug[key] = __hash__(item)

    def __check_data_curruption(self):
        if self.patience > 0: 
            assert self.__incomplete_frag__ is not None
            assert self.hash_debug is not None
            for key in self.__incomplete_frag__:
                item = self.__incomplete_frag__[key]
                if isinstance(item, dict):
                    for subkey in item:
                        subitem = item[subkey]
                        assert self.hash_debug[key][subkey] == __hash__(subitem), ('Currupted data! 发现腐败数据!')
                else:
                    assert self.hash_debug[key] == __hash__(item), ('Currupted data! 发现腐败数据!')
            self.patience -= 1
