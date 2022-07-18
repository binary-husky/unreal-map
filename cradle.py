import torch, time
from ALGORITHM.commom.norm import DynamicNormFix

input_size = 1
only_for_last_dim = True
dynamic_norm = DynamicNormFix(input_size, only_for_last_dim, exclude_one_hot=True, exclude_nan=False)

for _ in range(101100):
    
    # mask = (torch.randn(60, 1, out=None) > 0)
    # x = torch.where(mask,
    #                 torch.randn(60, 1, out=None)*10,
    #                 torch.randn(60, 1, out=None)*5,
    #                 )
    # 左边
    std = 0.01; offset = -0.01;  num = 5
    x3 = torch.randn(num, 1, out=None) * std + offset
  
    # 中间
    std = 0.01; offset = 0;    num = 500
    x2 = torch.randn(num, 1, out=None) * std + offset

    # 右边
    std = 0.01; offset = 1;   num = 5
    x1 = torch.randn(num, 1, out=None) * std + offset
        
        
    # # 左边
    # std = 1; offset = -10;  num = 5
    # x3 = torch.randn(num, 1, out=None) * std + offset
  
    # # 中间
    # std = 1; offset = 5;    num = 500
    # x2 = torch.randn(num, 1, out=None) * std + offset

    # # 右边
    # std = 1; offset = 5;   num = 5
    # x1 = torch.randn(num, 1, out=None) * std + offset
    
    x = torch.cat((x1,x2,x3), 0)
    y = dynamic_norm(x)
    
print(y)
time.sleep(60)