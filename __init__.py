import torch

# 创建一个大小为 (3, 4, 5) 的张量
x = torch.randn(3, 4, 5)
print(x)
# 在最后一个维度上求和
sum_result = torch.sum(x, dim=-1)

print(sum_result.size())  # 输出：torch.Size([3, 4])
print(sum_result)
# 写个归并排序










