import torch, time
import pickle, os

from UTIL.colorful import print亮红
from .tensor_ops import __hash__
from UTIL.exp_helper import singleton

@singleton
class SynWorker:

    def __init__(self, mod) -> None:
        self.sychronize_FILE_hashdict = 'TEMP/sychronize_hashdict'
        self.sychronize_FILE_cnt = 'TEMP/sychronize_cnt'
        self.mod = mod
        self.sychronize_internal_hashdict = {}
        self.sychronize_internal_cnt = {}
        self.follow_cnt = {}
        print亮红('warning, SynWorker init, mod is', mod)
        time.sleep(5)
        if mod == 'follow':
            with open(self.sychronize_FILE_hashdict, 'rb') as f:
                self.sychronize_internal_hashdict = pickle.load(f)
            with open(self.sychronize_FILE_cnt, 'rb') as f:
                self.sychronize_internal_cnt = pickle.load(f)
        else:
            try:
                os.remove(self.sychronize_FILE_hashdict)
                os.remove(self.sychronize_FILE_cnt)
            except: pass

    def dump_sychronize_data(self):
        if self.mod == 'follow':
            return
        with open(self.sychronize_FILE_hashdict, 'wb+') as f:
            pickle.dump(self.sychronize_internal_hashdict, f)
        with open(self.sychronize_FILE_cnt, 'wb+') as f:
            pickle.dump(self.sychronize_internal_cnt, f)

    def sychronize_experiment(self, key, data, reset_when_close=False):
        if self.mod == 'lead':
            hash_code = __hash__(data)
            if key not in self.sychronize_internal_hashdict:
                self.sychronize_internal_cnt[key] = 0
                self.sychronize_internal_hashdict[key] = [
                    {
                        'hash_code':hash_code,
                        'data': data,
                    }
                    ,
                ]
            else:
                self.sychronize_internal_hashdict[key].append({
                        'hash_code':hash_code,
                        'data': data,
                })

            self.sychronize_internal_cnt[key] += 1

        if self.mod == 'follow':
            hash_code = __hash__(data)
            if key not in self.follow_cnt:
                self.follow_cnt[key] = 0

            if hash_code != self.sychronize_internal_hashdict[key][self.follow_cnt[key]]['hash_code']:
                if not (torch.isclose(self.sychronize_internal_hashdict[key][self.follow_cnt[key]]['data'],data).all()) or (not isinstance(data, torch.Tensor)):
                    print('%s: error expected hash: %s, get hash %s, data %s'%(key,
                        self.sychronize_internal_hashdict[key][self.follow_cnt[key]]['hash_code'],
                        hash_code,
                        str(data)
                    ))
                else:
                    print('%s: error expected hash, but very very close (<1e-5)'%key)
                    if reset_when_close:
                        return data
            self.follow_cnt[key] += 1