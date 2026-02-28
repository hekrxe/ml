# 微积分
## 导数
导数描述了自变量的微小变化导致因变量微小变化的关系.即函数在某一点函数值的增量与自变量的增量的比值.

当 x 在 a 点附近变化一个很小的量 Δx时, 函数值的变化可以用线性函数近似:

$$
f(a+\Delta x) = f(a) + k⋅\Delta x
$$

k⋅Δx 即线性变化的部分, k即线性变化的斜率即导数.

数学定义:

$$
k = f'(a) = \lim_{\Delta x \to 0} \frac{f(a+\Delta x) - f(a)}{\Delta x}
$$


证明:

$$
\boxed{\frac{d}{dx}x^n = nx^{n-1}}
$$

- 步骤 1:利用因式分解公式

$$
a^n - b^n = (a-b)(a^{n-1} + a^{n-2}b + a^{n-3}b^2 + \cdots + ab^{n-2} + b^{n-1})
$$

- 步骤 2:应用于 $(x+h)^n - x^n$
    令 $a = x+h$,  $b = x$ , 则:

$$
(x+h)^n - x^n = h \cdot \left[ (x+h)^{n-1} + (x+h)^{n-2}x + (x+h)^{n-3}x^2 + \cdots + (x+h)x^{n-2} + x^{n-1} \right]
$$

- 步骤 3:代入导数定义

$$
\frac{d}{dx}x^n = \lim_{h \to 0} \frac{(x+h)^n - x^n}{h}
$$

$$
= \lim_{h \to 0} \frac{h \cdot \left[ (x+h)^{n-1} + (x+h)^{n-2}x + \cdots + x^{n-1} \right]}{h}
$$

$$
= \lim_{h \to 0} \left[ (x+h)^{n-1} + (x+h)^{n-2}x + \cdots + x^{n-1} \right]
$$

- 步骤 4:取极限（令 $h \to 0$ ）

$$
= \underbrace{x^{n-1} + x^{n-1} + \cdots + x^{n-1}}_{n \text{ 项}}
$$
$$
= nx^{n-1}
$$

---

- 三角函数

$$
d(sinx)/dx = cosx
$$

- 加法法则: 基本函数相加形成的复合函数导数等于基本函数导数之和

$$
d(u+v)/dx = d(u)/dx + d(v)/dx
$$

- 乘法法则: 前导后不导加上后导前不导

$$
d(uv)/dx = u\cdot d(v)/dx + v\cdot d(u)/dx
$$

- 链式法则:外层导数与内层导数依次相乘

$$
\frac{d}{dx}u(v(x)) = u'(v(x)) \cdot v'(x)
$$

- 指数函数

$$
d(e^x)/dx = e^x
$$

## 偏导
保持其他变量固定而关注一个变量的微小变化带来的函数值变化情况, 这种变化的比值就是偏导数.
数学定义:

$$
\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1,  x_2,  \ldots,  x_i + \Delta x_i,  \ldots,  x_n) - f(x_1,  x_2,  \ldots,  x_i,  \ldots,  x_n)}{\Delta x_i}
$$

## 梯度
梯度是一个向量, 其方向指向函数值增加最快的方向, 长度表示在该方向上的变化率.
数学定义:

$$
\nabla f = \left[ \frac{\partial f}{\partial x_1},  \frac{\partial f}{\partial x_2},  \ldots,  \frac{\partial f}{\partial x_n} \right]
$$

## 微积分
表示所有的微小量累加起来的结果.

如果函数 $f(x)$ 在区间 $[a, b]$ 上连续, 并且存在原函数 $F(x)$（即 $F'(x) = f(x)$）, 则:

$$
\int_a^b f(x)dx = F(b) - F(a)
$$
等价于
$$
\int_a^b F'(x)dx = F(b) - F(a)
$$

## 泰勒公式
泰勒公式是一种将函数表示为多项式的方法.
它的基本思想是, 在一个点 $x=a$ 附近, 函数可以用一个多项式来近似.

### 泰勒公式的微积分基本定理推导
泰勒公式本质上是微积分基本定理连续累加的结果.
#### 微积分基本定理
微积分基本定理采用定积分来展示函数 $F(x)$ 与它的导数之间的关系, 即:

$$
\int_a^b F'(x)dx = F(b) - F(a)
$$

也就是说, 已知 $F(x)$ 可以求解 $F'(x)$ 的定积分.

#### 变量代换与一阶展开

假设 $a$ 为定值, 且 $b - a = h$, 则上面的微积分基本定理可以写成:

$$
F(a+h) = F(a) + \int_a^{a+h} F'(x)dx
$$

意义是：函数在$a + h$处的值等于其在$a$处的值加上导数从$a$ 到$a + h$的积分 。

- 令 $x = a + t$, 其中 $t$ 是新的积分变量, $a$是常数
- 当 $x = a$ 时, $t = 0$ (积分下限)
- 当 $x = a + h$ 时, $t = h$ (积分上限)
- 微分: $dx = dt$ (对新变量 $t$ 的微分)

代入后得:

$$
F(a+h) = F(a) + \int_0^h F'(a+t)dt
$$

#### 二阶展开

如果 $F'(x)$ 是连续可导函数, 那么对 $F'(a+t)$ 应用微积分基本定理:

$$
F'(a+t) = F'(a) + \int_0^t F''(a+t_1)dt_1
$$

将其代入一阶展开式:

$$
\begin{aligned}
F(a+h) &= F(a) + \int_0^h \left[ F'(a) + \int_0^t F''(a+t_1)dt_1 \right] dt \\
&= F(a) + F'(a)\int_0^h dt + \int_0^h \int_0^t F''(a+t_1)dt_1 dt \\
&= F(a) + F'(a)h + \int_0^h \int_0^t F''(a+t_1)dt_1 dt
\end{aligned}
$$

### n阶展开（泰勒公式）
重复上述过程, 对 $F''(a+t_1)$ 再次应用微积分基本定理:

$$
F''(a+t_1) = F''(a) + \int_0^{t_1} F'''(a+t_2)dt_2
$$

代入后得到三阶项, 以此类推, 经过 $n$ 次迭代后, 可得:

$$
F(a+h) = \sum_{k=0}^n \frac{F^{(k)}(a)}{k!}h^k + R_n(h)
$$

其中余项为:

$$
R_n(h) = \int_0^h \int_0^{t_{n-1}} \cdots \int_0^{t_1} F^{(n+1)}(a+t_n)dt_n \cdots dt_1
$$

### 结论
通过微积分基本定理的多次连用, 我们成功推导出了泰勒公式.这说明泰勒公式本质上是微积分基本定理的连续累加结果, 两者在数学上具有统一性.

$$
\boxed{F(a+h) = \sum_{k=0}^n \frac{F^{(k)}(a)}{k!}h^k + \int_0^h \int_0^{t_{n-1}} \cdots \int_0^{t_1} F^{(n+1)}(a+t_n)dt_n \cdots dt_1}
$$

### 意义
- **理论统一**:泰勒公式与微积分基本定理本质上是一致的, 前者是后者的多次应用
- **数值计算**:通过多项式近似和积分余项, 可以精确计算函数值
- **误差分析**:积分余项提供了误差的定量估计, 为数值方法奠定基础

# 线性代数
TODO