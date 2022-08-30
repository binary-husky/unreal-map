import numpy as np
import copy


class spring():
    def __init__(self, buflen) -> None:
        self.buflen = buflen
        self.buf = [None for _ in range(self.buflen)]
        self.vis = ['None' for _ in range(self.buflen)]
        self.push_seq = [1, 8, 32, 64, 128, 512, 1024, 2048, 4096]
        self.cnt = 0
        
    def push_into(self, content, counter):
        for k in reversed(range(1, self.buflen)):
            if self.cnt % self.push_seq[k] ==0:
                self.buf[k] = self.buf[k-1]
                self.vis[k] = self.vis[k-1]
        self.buf[0] = content
        self.vis[0] = str(counter)
        self.cnt += 1
        
        
class PolicyGroupRollBuffer():
    def __init__(self, n_spring, buflen) -> None:
        self.n_spring = n_spring
        self.buflen = buflen
        self.springs = [spring(buflen=self.buflen) for _ in range(self.n_spring)]
        self.cnt = 0

    def is_empty(self):
        buf = self.flat_buf()
        return len(buf) == 0

    def push_policy_group(self, policy_group):
        self.push_into(policy_group
            # copy.deepcopy([p.state_dict() for p in policy_group])
        )

    def random_link(self, static_groups):
        pick_n = len(static_groups)
        buf = self.flat_buf()
        avail_n = len(buf)
        
        ar = np.arange( avail_n )
        
        if avail_n < pick_n:
            indices = np.random.choice(ar, size=pick_n, replace=True)
        else:
            indices = np.random.choice(ar, size=pick_n, replace=False)
            
        # 加载参数
        for i, static_nets in enumerate(static_groups):
            pick = indices[i]
            for k, p in enumerate(buf[pick]):
                static_nets[k].load_state_dict(p, strict=True)
                assert static_nets[k].static
                assert not static_nets[k].forbidden

    def push_into(self, content):
        j = np.random.choice(np.arange(self.n_spring), 1)[0]
        self.springs[j].push_into(content, self.cnt)
        self.cnt += 1
        for s in self.springs:
            print(s.vis)
            
    def flat_buf(self):
        res = []
        for i in range(1, self.buflen):
            for s in self.springs:
                t = s.buf[i]
                if t is not None:
                    res.append(t)
        return res

    def __len__(self):
        res = 0
        for i in range(1, self.buflen):
            for s in self.springs:
                t = s.buf[i]
                if t is not None:
                    res += 1
        return res
    
    
pgrb = PolicyGroupRollBuffer(6,6)

for i in range(2000):
    print(i)
    pgrb.push_policy_group(i)