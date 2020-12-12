# 参数化投资组合

#### Mean-Variance efficient portfolio
这个理论很好理解，市场上的所有投资者都想：
- 赚取更高的收益同时尽可能的降低风险；
- 统一的看待公司的投资回报和投资所面临的风险，就比如说期望回报( \mu  )和风险( \sigma  )应该被一起评估，而不是分别评估；
- 尽可能的从公司的有用信息来获得更高的收益。

所以有这两个参数 ($\mu$, $\sigma$)，就不难建立一个关于投资组合权重的方程，plus一些对于权重的constraints，就可以利用Lagrangian Multiplier来求出这个权重。假设所有的可投资资产池有 $i(i=1,2,3,...,n)$  个资产，我们有 $x_{0}$  的投资金额， $x_{0,i}$ 为我们分配到第 i  个资产的金额且 $x_{0,i} = \omega_{i} x_{0}$  ，其中 $\omega_{i}$ 为第 $i$ 个资产的权重。我们第一个constraint就是权重之和等于1， $\iota^{'} \omega = \sum_{i=1}^{n} = 1$ ；然后，假设 $r_{i}$ 为第 $i$ 个资产的投资收益率(是一个随机变量)， $z = (r_{1}, r_{2}, ..., r_{n})^{'}$ 为这个收益率的随机向量，所以我们记:

$$r = \sum_{i=1}^{n}r_{i}\omega_{i} = z^{'}\omega$$  ,  $$\mu_{i} = \mathbf{E}(r_{i})$$  , 为了计算方便起见，写成向量的形式：
$$\mu = \mathbf{E}(z) = (\mu_{1}, \mu_{2}, ..., \mu_{n})^{'}$$, $$Cov(z) = \mathbf{E}[(z-\mu)(z-\mu)^{'}] = \Sigma$$ ，我们通过一个二次规划问题来得到一个最佳投资组合：
$$\min_{\omega} \frac{1}{2}\omega^{'}\Sigma\omega$$ $$s.t.$$ $$\begin{cases} \mu^{'}\omega \geq \mu_{b}\\ \iota^{'}\omega = 1 \end{cases}$$  
综上，我们的目标就是通过最小化投资组合的方差，同时满足投资组合的期望收益率大于 $\mu_{b}$  , 来得到我们的最优投资组合的权重向量 $\omega = (\omega_{1}, \omega_{2}, ..., \omega_{n})^{'}$。
开始计算： $$L(\omega, \lambda, \theta) = \frac{1}{2}\omega^{'}\Sigma\omega - \lambda(\mu^{'}\omega - \mu_{b}) - \theta(\iota^{'}\omega - 1)$$ ， 我们对拉格朗日方程中的三个参数求偏导，可得我们权重向量： $$\bar{w} = \frac{\Sigma^{-1}\iota}{(\iota^{'}\Sigma^{-1}\iota)}$$。

这次，我们假设投资者的目的是通过选择投资组合权重 $\omega_{i,t}$ 来最大化投资者预期收益(其实是条件预期收益) $r_{p, t+1}$ 的效用:

$$\max_{\omega_{n_{t}}_{i=1}}}\mathbf{E}u(r_{p,t+1}) = \mathbf{E_{t}}u(\sum_{i=1}^{n_{t}}\omega_{i,t}r_{p,t+1})$$，
我们将每个资产对应的权重 $\omega_{i,t}$ 看做是每个资产对应‘’特征‘’和对应的投资记为一个函数形式： $$\omega_{i,t} = f(x_{i}, \theta)$$ ，这个‘’特征‘’ $\theta$ 也就是我们要基于用来做决策的因素, 我们将这个函数式的权重带入上边的最大化问题可得：
$$\max_{\theta} \frac{1}{T} \sum_{t=0}^{T-1}u(r_{p,t+1}) \equiv \frac{1}{T}\sum_{t=0}^{T-1}u(\sum_{i=1}^{n_{t}}f(x_{i,t};\theta)r_{i,t+1})$$ .
如果我们把 $f(x_{i}, \theta)$ 假设成一个线性关系(至少在我给的例子中)，其中 $\mathbf{\theta}$  是代表我们考虑因子的系数向量，即 $\theta = (\alpha, \beta)^{'}$ ，在栗子中，我们取一个市值因子 Market Capitalization(size)和一个动量因子m12(past 12 month average return)，其对应的系数分别为 \alpha 和 \beta 。现在这个线性方程可以表示为：
$$\omega_{i,t} = \bar{\omega}_{i,t} + \frac{1}{n_{t}}\theta^{'}\bar{x}_{i,t}$$ ，
其中， $\bar\omega_{i,t}$ 为某个资产 $i$  在时间 $t$ 时在一个benchmark portfolio的权重； 
$\theta$ 为因子的估计量； 
$\bar{x}_{i,t}$ 是截面标准化(standardized cross-sectionally)对应因子值。
所以，这个问题转化成为：
$$\max_{\theta} \frac{1}{T} \sum_{t=0}^{T-1}u(r_{p,t+1}) \equiv \frac{1}{T}\sum_{t=0}^{T-1}u(\sum_{i=1}^{n_{t}}(\bar{\omega}_{i,t} + \frac{1}{n_{t}}\theta^{'}\bar{x}_{i,t})r_{i,t+1})$$ ，
接下来我们就来实现这个问题，得出我们的 $\bar{\omega}_{i,t}$!
