# test_torch.py
import torch

print("=== PyTorch 环境测试 ===")
print(f"torch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 创建张量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b
print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {c}")

# 简单神经网络层测试
linear = torch.nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])
out = linear(x)
print(f"\n神经网络输出：{out}")

print("\n✅ 全部运行完成，PyTorch CPU环境工作正常！")
