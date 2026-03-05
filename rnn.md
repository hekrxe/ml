# 循环神经网络(RNN)
- 基于文本内容及其前后信息进行预测
- 基于目标不同时刻状态进行预测
- 基于数据历史信息进行预测
- 前部序列的信息经过处理后, 作为输入信息传递到后部序列

缺点:
- 梯度消失/爆炸问题:在长序列中, 梯度在反向传播时会被不断衰减或放大, 导致训练困难
- 无法处理长期依赖:RNN在处理长序列时, 无法记住长期依赖关系, 只能记住短期依赖关系

## 基本RNN公式

### 隐藏状态更新
对于时间步 \( t \), 隐藏状态 \( h_t \) 的计算:

$$
h_t = g_h \left( W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h \right)
$$

### 输出计算
时间步 \( t \) 的输出 \( y_t \):

$$
y_t = g_y \left( W_{hy} \cdot h_t + b_y \right)
$$


## 变量说明

| 符号 | 含义 |
|------|------|
| \( x_t \) | 时间步 \( t \) 的输入向量 |
| \( h_t \) | 时间步 \( t \) 的隐藏状态向量 |
| \( h_{t-1} \) | 时间步 \( t-1 \) 的隐藏状态向量（记忆） |
| \( y_t \) | 时间步 \( t \) 的输出向量 |
| \( W_{hh} \) | 隐藏状态到隐藏状态的权重矩阵 |
| \( W_{xh} \) | 输入到隐藏状态的权重矩阵 |
| \( W_{hy} \) | 隐藏状态到输出的权重矩阵 |
| \( b_h \) | 隐藏状态的偏置向量 |
| \( b_y \) | 输出的偏置向量 |
| \( g_h \) | 隐藏层激活函数（如 tanh、ReLU） |
| \( g_y \) | 输出层激活函数（如 softmax、sigmoid） |


## 关键特性

- **循环性**:隐藏状态 \( h_t \) 依赖于前一个隐藏状态 \( h_{t-1} \), 实现了信息的传递
- **参数共享**:同一权重矩阵在所有时间步共享, 减少参数数量
- **记忆能力**:通过隐藏状态保存历史信息, 适用于序列数据


## 变体公式

### LSTM（长短期记忆网络）
$$
\begin{align*}
i_t &= \sigma(W_{xi} \cdot x_t + W_{hi} \cdot h_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} \cdot x_t + W_{hf} \cdot h_{t-1} + b_f) \\
o_t &= \sigma(W_{xo} \cdot x_t + W_{ho} \cdot h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_{xc} \cdot x_t + W_{hc} \cdot h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t) \\
y_t &= g_y(W_{hy} \cdot h_t + b_y)
\end{align*}
$$

### GRU（门控循环单元）
$$
\begin{align*}
z_t &= \sigma(W_{xz} \cdot x_t + W_{hz} \cdot h_{t-1} + b_z) \\
r_t &= \sigma(W_{xr} \cdot x_t + W_{hr} \cdot h_{t-1} + b_r) \\
\tilde{h}_t &= \tanh(W_{x\tilde{h}} \cdot x_t + r_t \odot (W_{h\tilde{h}} \cdot h_{t-1}) + b_{\tilde{h}}) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \\
y_t &= g_y(W_{hy} \cdot h_t + b_y)
\end{align*}
$$


## 初始状态

- 对于第一个时间步 \( t=1 \), 需要初始化隐藏状态 \( h_0 \)
- 通常初始化为全零向量:\( h_0 = 0 \)


## 矩阵维度示例

假设:
- 输入 \( x_t \) 维度:\( [d_x] \)（特征维度）
- 隐藏状态 \( h_t \) 维度:\( [d_h] \)（隐藏层大小）
- 输出 \( y_t \) 维度:\( [d_y] \)（输出维度）

则权重矩阵维度:
- \( W_{xh} \): \( [d_h, d_x] \)
- \( W_{hh} \): \( [d_h, d_h] \)
- \( W_{hy} \): \( [d_y, d_h] \)

# BRNN
-  Bidirectional RNN
-  同时考虑序列的前向和后向信息
-  适用于需要考虑上下文的任务, 如情感分析、机器翻译

# DRNN
-  Deep RNN
-  多个隐藏层堆叠, 每个隐藏层都是一个RNN
-  适用于处理长序列, 如文本分类、语音识别
