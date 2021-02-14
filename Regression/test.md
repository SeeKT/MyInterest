# test
```python {cmd="/home/toda/.local/share/virtualenvs/MyInterest-csCBqNJG/bin/python" matplotlib=true}
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-10, 10, 0.01)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.show()
```

#### エントロピー
確率密度関数$p(\boldsymbol{x})$に対して，
$$
H[p(\boldsymbol{x})] = - \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} = -\mathbb{E}[\ln p(\boldsymbol{x})]
$$
をエントロピーと呼ぶ．エントロピーは確率分布の乱雑さを表す指標として知られている．

#### KLダイバージェンス
2つの確率密度関数$p(\boldsymbol{x})$および$q(\boldsymbol{x})$に対して，
$$
\mathrm{KL}[q(\boldsymbol{x}) \| p(\boldsymbol{x})] = -\int q(\boldsymbol{x}) \ln \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})} d\boldsymbol{x} = \mathbb{E}_q[\ln q(\boldsymbol{x})] - \mathbb{E}_q[\ln p(\boldsymbol{x})]
$$
をKLダイバージェンスと呼ぶ．任意の確率分布の組に対して$\mathrm{KL}[q(\boldsymbol{x}) \| p(\boldsymbol{x})] \geq 0$である．等号が成り立つのは，2つの分布が完全に一致する場合，i.e., $p(\boldsymbol{x}) = q(\boldsymbol{x})$に限られる．KLダイバージェンスは，2つの確率分布の距離を表していると解釈できるが，距離の公理は満たしていない．

#### サンプリングによる期待値の近似計算
ある確率分布$p(\boldsymbol{x})$と関数$\alpha(\boldsymbol{x})$に対して，期待値の定義式による解析的な計算が行えない場合がある．
このような場合は，$p(\boldsymbol{x})$からのサンプル点$\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(L)} \sim p(\boldsymbol{x})$を得ることで，期待値の近似値を以下のように求められる．
$$
\mathbb{E}[\alpha(\boldsymbol{x})] \approx \frac{1}{L} \sum_{l = 1}^L \alpha(\boldsymbol{x}^{(l)})
$$
複雑な確率分布のように積分計算が困難な場合や天文学的な組合せ数の足し算が必要な場合は，サンプリング手法を用いて現実的な時間内で近似解を得ることがよく行われる．

#### 正規分布のエントロピーとKLダイバージェンス
確率密度関数に対数をとると，
$$
\ln f(x) = - \frac{1}{2} \left(\frac{(x - \mu)^2}{\sigma^2} + \ln \sigma^2 + \ln 2\pi \right)
$$
となる．これより，正規分布のエントロピーは，
$$
H[f(\boldsymbol{x})] = -\mathbb{E}[\ln f(\boldsymbol{x})] = \frac{1}{2} \left(\frac{\mathbb{E}[(x - \mu)^2]}{\sigma^2} + \ln \sigma^2 + \ln 2\pi \right) = \frac{1}{2}(1 + \ln \sigma^2 + \ln 2\pi)
$$
となる．

$g(x)$を，$X \sim N(\hat{\mu}, \hat{\sigma}^2)$の確率密度関数とする．KLダイバージェンスを求める．
$$
\mathrm{KL}[g(x) \| f(x)] = -H[g(x)] - \mathbb{E}_{g(x)}[\ln f(x)]
$$
であり，
$
\mathbb{E}_{g(x)}[\ln f(x)] = - \frac{1}{2} \left(\frac{\mathbb{E}_{g(x)}[x^2] - 2 \mathbb{E}_{g(x)}[x] \mu + \mu^2}{\sigma^2} + \ln \sigma^2 + \ln 2\pi \right)
$

$
\hspace{20.5mm}= - \frac{1}{2} \left(\frac{\hat{\mu}^2 + \hat{\sigma}^2 - 2 \hat{\mu}\mu + \mu^2}{\sigma^2} + \ln \sigma^2 + \ln 2\pi\right)
$
より，
$$
\mathrm{KL}[g(x) \| f(x)] = \frac{1}{2} \left(\frac{(\mu - \hat{\mu})^2 + \hat{\sigma}^2}{\mu^2} + \ln \frac{\sigma^2}{\hat{\sigma}^2} - 1 \right)
$$
となる．