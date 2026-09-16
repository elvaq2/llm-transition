import torch
import torch.nn as nn

# 1. 准备训练数据
# 输入 x
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 真实标签 y=2x+1
y_data = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. 定义模型：单层全连接
model = nn.Linear(1, 1)#输入一个数字，输出一个数字

# 3. 损失函数：均方误差 MSE
loss_fn = nn.MSELoss()#均方误差，衡量预测值与真实值相差多大
# 优化器：SGD 随机梯度下降，学习率 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#更新权重W和偏置b

print("===== 训练前参数 =====")
print(f"W: {model.weight.item():.4f}, b: {model.bias.item():.4f}\n")

# 4. 开始训练循环
epoch_total = 1000
for epoch in range(epoch_total):
    # 前向传播：计算预测值
    y_pred = model(x_data)
    # 计算损失
    loss = loss_fn(y_pred, y_data)

    # 反向传播 + 更新参数
    optimizer.zero_grad()  # 清空上一轮梯度，非常重要！
    loss.backward()        # 反向求导，计算梯度
    optimizer.step()       # 根据梯度更新W,b

    # 每100轮打印一次信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epoch_total}], Loss: {loss.item():.4f}, W:{model.weight.item():.4f}, b:{model.bias.item():.4f}")

print("\n===== 训练完成 =====")
print(f"最终 W: {model.weight.item():.4f}, b: {model.bias.item():.4f}")
print("期待 W≈2，b≈1")

# 5. 用训练好的模型做预测
x_test = torch.tensor([[5.0]])
y_test_pred = model(x_test)
print(f"\n预测 x=5 时，y = {y_test_pred.item():.4f}")
print("理论值 y=2*5+1=11")
