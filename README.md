# 参数化投资组合


#### Mean-Variance efficient portfolio
这个理论很好理解，市场上的所有投资者都想：
- 赚取更高的收益同时尽可能的降低风险；
- 统一的看待公司的投资回报和投资所面临的风险，就比如说期望回报( \mu  )和风险( \sigma  )应该被一起评估，而不是分别评估；
- 尽可能的从公司的有用信息来获得更高的收益。

所以有这两个参数$$( \mu, \sigma  )$$，就不难建立一个关于投资组合权重的方程，plus一些对于权重的constraints，就可以利用Lagrangian Multiplier来求出这个权重。假设所有的可投资资产池有 $i(i=1,2,3,...,n)$  个资产，我们有 x_{0}  的投资金额， x_{0,i} 为我们分配到第 i  个资产的金额且 x_{0,i} = \omega_{i} x_{0}  ，其中 \omega_{i} 为第 i 个资产的权重。我们第一个constraint就是权重之和等于1， \iota^{'} \omega = \sum_{i=1}^{n} = 1 ；然后，假设 r_{i} 为第 i 个资产的投资收益率(是一个随机变量)， z = (r_{1}, r_{2}, ..., r_{n})^{'} 为这个收益率的随机向量，所以我们记：
r = \sum_{i=1}^{n}r_{i}\omega_{i} = z^{'}\omega  ,  \mu_{i} = \mathbf{E}(r_{i})  , 为了计算方便起见，写成向量的形式：
\mu = \mathbf{E}(z) = (\mu_{1}, \mu_{2}, ..., \mu_{n})^{'}, Cov(z) = \mathbf{E}[(z-\mu)(z-\mu)^{'}] = \Sigma ，我们通过一个二次规划问题来得到一个最佳投资组合：
\min_{\omega} \frac{1}{2}\omega^{'}\Sigma\omega\\ s.t. \begin{cases} \mu^{'}\omega \geq \mu_{b}\\ \iota^{'}\omega = 1 \end{cases}  
综上，我们的目标就是通过最小化投资组合的方差，同时满足投资组合的期望收益率大于 \mu_{b}  , 来得到我们的最优投资组合的权重向量 \omega = (\omega_{1}, \omega_{2}, ..., \omega_{n})^{'} 。
开始计算： L(\omega, \lambda, \theta) = \frac{1}{2}\omega^{'}\Sigma\omega - \lambda(\mu^{'}\omega - \mu_{b}) - \theta(\iota^{'}\omega - 1) ， 我们对拉格朗日方程中的三个参数求偏导，可得我们权重向量： \bar{w} = \frac{\Sigma^{-1}\iota}{[\iota^{'}\Sigma^{-1}\iota]} 。
