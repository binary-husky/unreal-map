import time
from UTIL.tensor_ops import __hash__, repeat_at
from UTIL.colorful import *
from .alg_base import AlgorithmBase

# model IO
class RLAlgorithmBase(AlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        super().__init__(n_agent, n_thread, space, mcv, team)
        
        # data integraty check
        self._unfi_frag_ = None

        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 1000



    def interact_with_env(self, team_intel):
        raise NotImplementedError

    def save_model(self, update_cnt, info=None):
        raise NotImplementedError

    def process_framedata(self, traj_framedata):
        raise NotImplementedError

    # Rollout Processor 准备提交Rollout，以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
    # note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
    def process_framedata(self, traj_framedata):
        ''' 
            hook is called when reward and next moment observation is ready,
            now feed them into trajectory manager.
            Rollout Processor | 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
            note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
        '''
        # strip info, since it is not array
        items_to_pop = ['info', 'Latest-Obs']
        for k in items_to_pop:
            if k in traj_framedata:
                traj_framedata.pop(k)
        # the agent-wise reward is supposed to be the same, so averge them
        if self.ScenarioConfig.RewardAsUnity:
            traj_framedata['reward'] = repeat_at(traj_framedata['reward'], insert_dim=-1, n_times=self.n_agent)
        # change the name of done to be recognised (by trajectory manager)
        traj_framedata['_DONE_'] = traj_framedata.pop('done')
        traj_framedata['_TOBS_'] = traj_framedata.pop(
            'Terminal-Obs-Echo') if 'Terminal-Obs-Echo' in traj_framedata else None
        # mask out pause thread
        traj_framedata = self.mask_paused_env(traj_framedata)
        # put the frag into memory
        self.batch_traj_manager.feed_traj(traj_framedata)


    def check_reward_type(self, AlgorithmConfig):
        if self.ScenarioConfig.RewardAsUnity != AlgorithmConfig.TakeRewardAsUnity:
            assert self.ScenarioConfig.RewardAsUnity
            assert not AlgorithmConfig.TakeRewardAsUnity
            print亮紫(
                'Warning, the scenario (MISSION) provide `RewardAsUnity`, but AlgorithmConfig does not `TakeRewardAsUnity` !')
            print亮紫(
                'If you continue, team reward will be duplicated to serve as individual rewards, wait 3s to proceed...')
            time.sleep(3)

    def mask_paused_env(self, frag):
        running = ~frag['_SKIP_']
        if running.all():
            return frag
        for key in frag:
            if not key.startswith('_') and hasattr(frag[key], '__len__') and len(frag[key]) == self.n_thread:
                frag[key] = frag[key][running]
        return frag


    '''
        Get event from hmp task runner, called when each test rotinue is complete.
    '''
    def on_notify(self, message, **kargs):
        self.save_model(
            update_cnt=self.traj_manager.update_cnt,
            info=str(kargs)
        )




    ''' 
        function to be called when reward is received
    '''
    def commit_traj_frag(self, unfi_frag, req_hook=True):
        assert self._unfi_frag_ is None
        self._unfi_frag_ = unfi_frag
        self._check_data_hash()  # check data integraty
        if req_hook:
            # leave a hook
            return self.traj_waiting_hook
        else:
            return None





    def traj_waiting_hook(self, new_frag):
        ''' 
            This function will be called from <multi_team.py::deal_with_hook()>
            hook is called when reward and next moment observation is ready
        '''
        # do data curruption check at beginning, this is important!
        self._check_data_curruption()
        # finish the frame data with new data feedin
        fi_frag = self._unfi_frag_
        fi_frag.update(new_frag)
        # call upper level function to deal with frame data
        self.process_framedata(traj_framedata=fi_frag)
        # delete data reference
        self._unfi_frag_ = None


    def _no_hook(self, new_frag):
        return



    # protect data from overwriting
    def _check_data_hash(self):
        if self.patience > 0:
            self.patience -= 1
            self.hash_db = {}
            # for debugging, to detect write protection error
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    self.hash_db[key] = {}
                    for subkey in item:
                        subitem = item[subkey]
                        self.hash_db[key][subkey] = __hash__(subitem)
                else:
                    self.hash_db[key] = __hash__(item)

    # protect data from overwriting
    def _check_data_curruption(self):
        if self.patience > 0:
            self.patience -= 1
            assert self._unfi_frag_ is not None
            assert self.hash_db is not None
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    for subkey in item:
                        subitem = item[subkey]
                        assert self.hash_db[key][subkey] == __hash__(subitem), ('Currupted data!')
                else:
                    assert self.hash_db[key] == __hash__(item), ('Currupted data!')

