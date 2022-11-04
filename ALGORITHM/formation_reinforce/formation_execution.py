import numpy as np
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, repeat_at, np_softmax, gather_righthand
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit
class main_formation(object):
    def __init__(self):
        return
    
    def get_action(self, state_recall):
            #     I_State_Recall = {'obs':obs[~ENV_PAUSE], 
            # 'Test-Flag':State_Recall['Test-Flag'], 
            # 'threads_active_flag':~ENV_PAUSE, 
            # 'Latest-Team-Info':State_Recall['Latest-Team-Info'][~ENV_PAUSE]}
            
        obs = state_recall['obs']
        thread_active = state_recall['threads_active_flag']
        n_active = len(obs)
        parsed_obs = self.parse_obs(obs)
        Delta = self.decode_neighbors(parsed_obs)
        action = self.dir_to_action(Delta)
        # for thread in range(n_active):
        #     pass            

        return np.squeeze(action,-1)

    def init_vs(self, n_agent):
        t = np.arange(n_agent)
        y_Struct = t/50 * 8 # t/50 * 8
        x_Struct = np.sin(y_Struct*2)
        V_Struct = np.stack((x_Struct, y_Struct)).transpose()   # 50,2

        _theta = -45 * np.pi / 180.
        vs_rotation = np.array([ # note! _theta -> -_theta
            [np.cos(_theta),-np.sin(_theta)], 
            [np.sin(_theta),np.cos(_theta)]
        ])
        V_Struct = np.matmul(vs_rotation, np.expand_dims(V_Struct,-1)).squeeze()
        vs_offset = np.array([2,2])
        V_Struct += vs_offset
        # self.show_points(V_Struct)
        return V_Struct 

    def decode_neighbors(self, parsed_obs):
        f_id = parsed_obs['F_ID']
        pos = parsed_obs['POS']
        f_pos = parsed_obs['F_POS']
        id = parsed_obs['ID']
        f_id = parsed_obs['F_ID']
        n_agent = f_id.shape[1]
        V_Struct = self.init_vs(n_agent)

        # calculate delta_xy in virtual struture
        target_vs_pos = V_Struct[f_id.astype(np.int)] 
        self_vs_pos = V_Struct[id.astype(np.int)] 
        vs_Dpos = target_vs_pos-repeat_at(self_vs_pos, insert_dim=-2, n_times=target_vs_pos.shape[-2])
        # -1 index will point to wrong position, which must be fixed later
        
        # calculate delta_xy via observation
        Dpos = f_pos - repeat_at(pos, insert_dim=-2, n_times=f_pos.shape[-2])

        DDpos = Dpos - vs_Dpos
        sel_non_empty = f_id >=0 
        sel_empty = ~sel_non_empty
        DDpos[sel_empty] = 0
        DDpos_length = np.linalg.norm(DDpos,axis=-1)
        # sm_DDpos_length = np_softmax(DDpos_length,-1)
        sel_move_to = np.expand_dims(np.argmax(DDpos_length,-1),-1)
        resDDpos = gather_righthand(src=DDpos, index=sel_move_to)
        # resDDpos_length = gather_righthand(src=DDpos_length, index=sel_move_to)
        Delta = resDDpos.sum(-2)

        # Delta = repeat_at(V_Struct,0,pos.shape[0])  - pos
        return Delta

    @staticmethod
    def reverse_binary(binary, bit=8):
        assert bit==8
        mask_and_id = np.zeros_like(binary)[..., 0]
        for i in range(8):
            mask_and_id += binary[..., i]* 2**i
        return mask_and_id

    def parse_obs(self, obs):
        N_FPH = 6; N_HPH = 6
        _obs = obs # _2cpu2numpy(obs[0])
        if len(_obs.shape) == 3: _obs = my_view(_obs, [0 ,0, 12, 15])
        _idb = _obs[...,7:]
        _id = self.reverse_binary(_idb)
        dead_hs = (_obs==0).all(-1) # 因为序号0 和空观测0 都会导致0， 
        _id = np.where(dead_hs, -1, _id)
        _pos = _obs[...,1:3]
        _vel = _obs[...,4:6]
        _ang = _obs[...,4]
        parsed = {
            'POS':_pos[:,:,0],          'FVEL':_vel[:,:,0],         'ANG':_ang[:,:,0],         'ID':_id[:,:,0],
            'F_POS':_pos[:,:,1:N_FPH],  'F_VEL':_vel[:,:,1:N_FPH],  'F_ANG':_ang[:,:,1:N_FPH], 'F_ID':_id[:,:,1:N_FPH],
            'H_POS':_pos[:,:,-N_HPH:],  'H_VEL':_vel[:,:,-N_HPH:],  'H_ANG':_ang[:,:,-N_HPH:], 'H_ID':_id[:,:,-N_HPH:] 
        }
        return parsed

    def show_points(self, arr):
        import time
        from config import GlobalConfig as cfg
        if cfg.matlab_logger is None:
            from VISUALIZE.mcom import mcom
            mcv = mcom(path='%s/v2d_logger/'%cfg.logdir,
                        digit=16, rapid_flush=True, draw_mode='Native')
            mcv.v2d_init()
            cfg.matlab_logger = mcv
        mcv = cfg.matlab_logger
        mcv.v2d_clear()
        arr = arr.squeeze(); assert len(arr.shape)==2; assert arr.shape[1]==2
        n_points = len(arr)
        for i in range(n_points):
            mcv.v2dx('cir|%d|r|0.04'%(i), arr[i][0], arr[i][1])

        mcv.v2d_show()
        time.sleep(0.5)


    def showoff(self, obs, atten, attn_offset=0, top_n=1):
        obs = torch.nan_to_num_(obs, 0)
        import numpy as np
        import time
        from config import GlobalConfig as cfg
        from UTIL.tensor_ops import _2tensor, _2cpu2numpy

        if cfg.matlab_logger is None:
            from VISUALIZE.mcom import mcom
            mcv = mcom(path='%s/v2d_logger/'%cfg.logdir,
                        digit=16, rapid_flush=True, draw_mode='Native')
            mcv.v2d_init()
            cfg.matlab_logger = mcv
        mcv = cfg.matlab_logger
        mcv.v2d_clear()


        _obs = _2cpu2numpy(obs[0])
        if len(_obs.shape) == 2: _obs = my_view(_obs, [0, 12, 15])
        _id = _obs[:,:,7:]
        __id = reverse_binary(_2cpu2numpy(_id))
        dead_hs = (_obs==0).all(-1) # 因为序号0 和空观测0 都会导致0， 
        __id = np.where(dead_hs, -1, __id)
        _pos = _obs[:,:,1:3]


        share_id = __id.flatten()
        share_obs = my_view(_pos, [-1, 2])
        for i in range(len(share_id)):
            if share_id[i]==0 and (share_obs[i]==0).all(): continue
            if share_id[i]<0: continue
            if share_id[i] >=50:
                mcv.v2dx('cir|%d|r|0.04'%(share_id[i]), share_obs[i][0], share_obs[i][1])
            else:
                mcv.v2dx('cir|%d|g|0.04'%(share_id[i]), share_obs[i][0], share_obs[i][1])
        # mcv.v2d_show()
        # time.sleep(0.5)

        # _atten = atten[0]
        # _atten = _2cpu2numpy(_atten.squeeze())
        # _atten_sort = np.argsort(-_atten, axis=-1)   # 需要用降序排列，先取倒数
        # _atten_sort_top = _atten_sort[..., :top_n] + attn_offset
        # obs_focus_ = np.take_along_axis(__id, axis=-1, indices=_atten_sort_top)
        dead = (my_view(_obs, [0, -1])==0).all(-1)
        for i in range(50): 
            if dead[i]: continue    
            for j in range(6):
                t = __id[i,j+6]
                if t<0: continue
                mcv.v2d_line_object(i, t)

        mcv.v2d_show()
        time.sleep(0.5)
        return

    @staticmethod
    @jit(forceobj=True)
    def dir_to_action(vec, vel=None):
        if vel is None:
            vel = np.zeros_like(vec)
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

        action = np.where(direct == 0, 3, 0)    # 3 up
        action += np.where(direct == 1, 4, 0)   # 4 down
        action += np.where(direct == 2, 1, 0)   # 1 right
        action += np.where(direct == 3, 2, 0)   # 2 left

        action = (dis2target>0.05).astype(np.int)*action
        # make sure that all nan vec become invalid act 0, 
        # be careful when a different numpy version is used
        assert (action[np.isnan(np.sum(dot_stack,0))] == 0).all()
        # action *= 0
        return np.expand_dims(action, axis=-1)
