# 分布論
- Kosuke Toda (@SeeKT)
### 参考
- 佐和，回帰分析 新装版，朝倉書店，2020．
- 須山，ベイズ推論による機械学習 入門，講談社，2017．
- 狩野，2021年度多変量解析 講義資料 (大阪大学 大学院基礎工学研究科)

## 確率分布の基本的事項
#### 期待値 (平均)
$p$次元確率変数$\boldsymbol{X} = [X_1, \ldots, X_p]^{\mathrm{T}}$が，確率密度関数$f(\boldsymbol{x})$を持つとする．$\boldsymbol{X}$のある関数$\alpha(\boldsymbol{X})$の期待値は，
$$
\mathbb{E}[\alpha(\boldsymbol{X})] \coloneqq \int \alpha(\boldsymbol{x}) f(\boldsymbol{x}) d \boldsymbol{x}
$$
で定義される．
##### 期待値の性質
- $\mathbb{E}[a \alpha(\boldsymbol{X}) + b \beta(\boldsymbol{X})] = a \mathbb{E}[\alpha(\boldsymbol{X})] + b\mathbb{E}[\beta(\boldsymbol{X})]$
- $\mathbb{E}[A \boldsymbol{X} + \boldsymbol{b}] = A \mathbb{E}[\boldsymbol{X}] + \boldsymbol{b}$
#### 分散
$\boldsymbol{X}$の平均ベクトルを$\mathbb{E}[\boldsymbol{X}]$と表す．
$$
\mathrm{Var}(\boldsymbol{X}) \coloneqq \mathbb{E}\left[(\boldsymbol{X} - \mathbb{E}[\boldsymbol{X}])(\boldsymbol{X} - \mathbb{E}[\boldsymbol{X}])^{\mathrm{T}} \right]
$$
を$\boldsymbol{X}$の分散共分散行列という．2つの確率ベクトル$\boldsymbol{X}, \boldsymbol{Y}$の共分散行列を
$$
\mathrm{Cov}(\boldsymbol{X}, \boldsymbol{Y}) \coloneqq \mathbb{E}\left[(\boldsymbol{X} - \mathbb{E}[\boldsymbol{X}])(\boldsymbol{Y} - \mathbb{E}[\boldsymbol{Y}])^{\mathrm{T}} \right]
$$
で定義する．特に，$\mathrm{Cov}(\boldsymbol{X}, \boldsymbol{X}) = \mathrm{Var}(\boldsymbol{X})$．
##### 分散の性質
- $\mathrm{Cov}(A\boldsymbol{X} + \boldsymbol{b}, C\boldsymbol{Y} + \boldsymbol{d}) = A \mathrm{Cov}(\boldsymbol{X}, \boldsymbol{Y}) C^{\mathrm{T}}$
- $\mathrm{Var}(A\boldsymbol{X} + \boldsymbol{b}) = A \mathrm{Var}(\boldsymbol{X})A^{\mathrm{T}}$．
- $\mathrm{Var}(\boldsymbol{X}) \geq O$
- $\mathrm{Var}(\boldsymbol{X}) = O \Rightarrow \boldsymbol{X} = \mathbb{E}[\boldsymbol{X}] \; \; (\mathrm{w.p. \;}1)$

#### 確率変数の独立性
$\boldsymbol{X}$: $p$次元確率変数，$\boldsymbol{Y}$: $q$次元確率変数．$\forall B_1 \subset \mathbb{R}^p, \forall B_2 \subset \mathbb{R}^q$に対して，
$
\mathbb{P}(\boldsymbol{X} \in B_1, \boldsymbol{Y} \in B_2) = \mathbb{P}(\boldsymbol{X} \in B_1) \mathbb{P}(\boldsymbol{Y} \in B_2)
$
が成立するとき，$\boldsymbol{X}$と$\boldsymbol{Y}$は互いに独立であるといい，$\boldsymbol{X} \mathop{\perp \!\!\!\!\perp} \boldsymbol{Y}$と表す．

$
\boldsymbol{X} \mathop{\perp \!\!\!\!\perp} \boldsymbol{Y} \Longleftrightarrow 任意の有界連続関数\alpha(\cdot), \beta(\cdot)に対して
$
$
\hspace{20mm} \mathbb{E}[\alpha(\boldsymbol{X})\beta(\boldsymbol{Y})] = \mathbb{E}[\alpha(\boldsymbol{X})]\mathbb{E}[\beta(\boldsymbol{Y})]
$

$(\boldsymbol{X}, \boldsymbol{Y}), \boldsymbol{X}, \boldsymbol{Y}$の分布がそれぞれ確率密度関数$h(\boldsymbol{x}, \boldsymbol{y}), f(\boldsymbol{x}), g(\boldsymbol{y})$を持つとき，
$$
\boldsymbol{X} \mathop{\perp \!\!\!\!\perp} \boldsymbol{Y} \Longleftrightarrow h(\boldsymbol{x}, \boldsymbol{y}) = f(\boldsymbol{x})g(\boldsymbol{y})
$$

#### 条件付き分布
$p$次元確率変数$\boldsymbol{X}$を分割し，$\boldsymbol{X} = [\boldsymbol{X}_1^{\mathrm{T}}, \boldsymbol{X}_2^{\mathrm{T}}]^{\mathrm{T}}$とする($\boldsymbol{X}_1$: $p_1 \times 1$, $\boldsymbol{X}_2$: $p_2 \times 1$)．$\boldsymbol{X}, \boldsymbol{X}_1, \boldsymbol{X}_2$の確率密度関数をそれぞれ$f(\boldsymbol{x}_1, \boldsymbol{x}_2), f_1(\boldsymbol{x}_1), f_2(\boldsymbol{x}_2)$とする．
$\boldsymbol{X}_2 = \boldsymbol{x}_2$を与えた下での$\boldsymbol{X}_1$の条件付分布の確率密度関数は，
$$
f(\boldsymbol{x}_1 | \boldsymbol{x}_2) \coloneqq \frac{f(\boldsymbol{x}_1, \boldsymbol{x}_2)}{f_2(\boldsymbol{x}_2)} \; \; (f_2(\boldsymbol{x}_2) > 0)
$$
で定義される．

このとき，
$$
f_1(\boldsymbol{x}_1) = \int f(\boldsymbol{x}_1, \boldsymbol{x}_2) d\boldsymbol{x}_2, \; \; f_2(\boldsymbol{x}_2) = \int f(\boldsymbol{x}_1, \boldsymbol{x}_2) d\boldsymbol{x}_1
$$
が成立する．

$h(\boldsymbol{x}_1)$を$\boldsymbol{X}_1$の値域を含む領域で定義された可測関数で，$h(\boldsymbol{X}_1)$は可積分であるとする．
$$
\mathbb{E}[h(\boldsymbol{X}_1) | \boldsymbol{X}_2 = \boldsymbol{x}_2] \coloneqq \int h(\boldsymbol{x}_1) f(\boldsymbol{x}_1 | \boldsymbol{x}_2) d\boldsymbol{x}_1
$$
を，$\boldsymbol{X}_2 = \boldsymbol{x}_2$が与えられた下での$h(\boldsymbol{X}_1)$の条件付期待値という．これは$\boldsymbol{x}_2$の関数である．$\boldsymbol{x}_2$に確率ベクトル$\boldsymbol{X}_2$を代入した確率ベクトルを$\mathbb{E}[h(\boldsymbol{X}_1)|\boldsymbol{X}_2]$で表す．このとき，
$$
\mathbb{E}\left[\mathbb{E}[h(\boldsymbol{X}_1 | \boldsymbol{X}_2)]\right] = \mathbb{E}[h(\boldsymbol{X}_1)]
$$
が成立する．つまり，
$$
\int \left(\int h(\boldsymbol{x}_1) f(\boldsymbol{x}_1 | \boldsymbol{x}_2) d\boldsymbol{x}_1 \right) f_2(\boldsymbol{x}_2) d\boldsymbol{x}_2 = \int h(\boldsymbol{x}_1) f_1(\boldsymbol{x}_1) d\boldsymbol{x}_1
$$
である．
$g(\boldsymbol{x}_2)$を$\boldsymbol{X}_2$の値域を含む領域で定義された可測写像とする．このとき
$$
\mathbb{E}[h(\boldsymbol{X}_1)g(\boldsymbol{X}_2)^{\mathrm{T}} | \boldsymbol{X}_2] = \mathbb{E}[h(\boldsymbol{X}_1) | \boldsymbol{X}_2] g(\boldsymbol{X}_2)^{\mathrm{T}}
$$
が成り立つ．つまり，
$$
\int h(\boldsymbol{x}_1) g(\boldsymbol{x}_2)^{\mathrm{T}} f(\boldsymbol{x}_1 | \boldsymbol{x}_2) d\boldsymbol{x}_1 = \left(\int h(\boldsymbol{x}_1) f(\boldsymbol{x}_1 | \boldsymbol{x}_2) d\boldsymbol{x}_1 \right) g(\boldsymbol{x}_2)^{\mathrm{T}}
$$

さらに，$\mathbb{E}[\|\boldsymbol{X}\|^2] < \infty$として，
$$
\mathcal{H} \coloneqq \{h : D (\subset \mathbb{R}^{p_2}) \to \mathbb{R}^{p_1} \, | \, \mathbb{E}[\|h(\boldsymbol{X}_2) \|^2] < \infty \}
$$
を定義する．このとき，
$$
\argmin_{h \in \mathcal{H}} \mathbb{E}[\|\boldsymbol{X}_1 - h(\boldsymbol{X}_2) \|^2] = \mathbb{E}[\boldsymbol{X}_1 | \boldsymbol{X}_2]
$$
が成立する．

## 正規分布
### 1次元正規分布
平均$\mu$，分散$\sigma^2$の正規分布に従う確率変数$X$を考える．つまり，$X \sim N(\mu, \sigma^2)$とする．確率密度関数は，
$$
f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(- \frac{(x - \mu)^2}{2\sigma^2} \right)
$$
である．

#### 標準正規分布
$\mu = 0, \sigma^2 = 1$のときの正規分布$N(0, 1)$を標準正規分布という．その密度関数は，
$$
\varphi(z) = \frac{1}{\sqrt{2\pi}} \exp\left(- \frac{z^2}{2} \right)
$$
である．ここで，確率密度関数$\varphi(z)$を図示する．
```python {cmd="/home/toda/.local/share/virtualenvs/MyInterest-csCBqNJG/bin/python" matplotlib=true hide}
import numpy as np
import matplotlib.pyplot as plt
def phi(Z):
    return np.exp((-1)*Z**2/2)/np.sqrt(2*np.pi)
z = np.linspace(-3, 3, 601)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([-3,3]); ax.grid(True)
ax.set_xlabel(r"$z$"); ax.set_ylabel("density")
ax.plot(z, phi(z), c='k')
plt.show()
```

$N(\mu, \sigma^2)$の確率密度関数$f(x)$は，
$$
f(x) = \frac{1}{\sigma^2} \varphi\left(\frac{x - \mu}{\sigma} \right)
$$
のように，$\varphi(z)$の線形変換として記述できる．つまり，確率変数$X \sim N(\mu, \sigma^2)$に対して，標準化
$$
Z = \frac{X - \mu}{\sigma}, \; X = \mu + \sigma Z
$$
によって，$Z \sim N(0, 1)$の性質に変換できる．逆に$Z \sim N(0, 1)$から$X \sim N(\mu, \sigma^2)$を構成できる．

一般の正規分布$N(\mu, \sigma^2)$のモーメントは，$N(0, 1)$のモーメントで表現できる．
$
\mathbb{E}[X] = \mathbb{E}[\mu + \sigma Z] = \mu + \sigma \mathbb{E}[Z] = \mu
$
$
\mathrm{Var}[X] = \mathbb{V}[\mu + \sigma Z] = \sigma^2 \mathrm{Var}[Z] = \sigma^2
$



##### 標準正規分布の特性値
$\varphi(z)$が偶関数なので，奇数次のモーメントは$0$．よって，$\mathbb{E}[Z^{2k - 1}] = 0$．
偶数次のモーメントについて，
$$
J_k \coloneqq \mathbb{E}[Z^{2k}] = \int_{-\infty}^{\infty} z^{2k} \varphi(z) dz = \cdots = (2k - 1) \mathbb{E}[Z^{2(k - 1)}] = (2k - 1)J_{k - 1}
$$
より，
$$
J_k = (2k - 1) (2k - 3) \cdots 3 \cdot 1 = (2k - 1)!!
$$
ただし，$J_0 = \int_{-\infty}^{\infty} \varphi(z) dz = 1$．

また，$N(0, 1)$の積率母関数は，
$$
M_Z(t) = \mathbb{E}[e^{tZ}] = \cdots = e^{\frac{t^2}{2}}
$$
である．これより，$N(\mu, \sigma^2)$の積率母関数は，
$$
M_X(t) = \mathbb{E}[e^{tX}] = \mathbb{E}[e^{t(\mu + \sigma Z)}] = \exp\left(\mu t + \frac{\sigma^2 t^2}{2} \right)
$$
である．



### 多変量正規分布
$\boldsymbol{0} \coloneqq (0, \ldots, 0)^{\mathrm{T}}, I \coloneqq \mathrm{diag}\{1, \ldots, 1\}$とする．

$Z_1, Z_2, \ldots, Z_p \overset{\text{i.i.d.}}{\sim} N(0, 1)$とする．このとき，$\boldsymbol{Z} \coloneqq (Z_1, \ldots, Z_p) \sim N(\boldsymbol{0}, I)$であり，確率密度関数は，
$$
\varphi_p(\boldsymbol{Z}) = \frac{1}{(2 \pi)^{p/2}} \exp\left(- \frac{1}{2} \boldsymbol{Z}^{\mathrm{T}} \boldsymbol{Z} \right) = \prod_{i = 1}^p \frac{1}{\sqrt{2\pi}} \exp\left( - \frac{z_i^2}{2} \right)
$$
となる．$\mathbb{E}[\boldsymbol{Z}] = \boldsymbol{0}, \mathrm{Var}[\boldsymbol{Z}] = I$である．

つぎに，$\boldsymbol{Z}$を線形変換した確率変数$\boldsymbol{X} = A \boldsymbol{Z} + \boldsymbol{\mu}$を考える．ただし，$A \in GL_p(\mathbb{R})$, $\boldsymbol{\mu} \in \mathbb{R}^p$とする．さらに，$A, \boldsymbol{\mu}$の要素は確率変数ではないと仮定する．
$\boldsymbol{X}$の密度関数は，
$$
\varphi_n(A^{-1}(\boldsymbol{x} - \boldsymbol{\mu})) \left\| \frac{d \boldsymbol{z}}{d \boldsymbol{x}^{\mathrm{T}}} \right\| = \varphi(A^{-1} (\boldsymbol{x} - \boldsymbol{\mu})) | |A| |^{-1}
$$
として求まる．ただし，$||A||$は$A$の行列式の絶対値である．
よって，$\boldsymbol{X}$の密度関数は，
$$
f_p(\boldsymbol{x}) = \frac{||A||^{-1}}{(2\pi)^{p/2}} \exp\left(- \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^{\mathrm{T}} (A^{-1})^{\mathrm{T}} A^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
$$
となる．$A \in GL_p(\mathbb{R})$であることから，$AA^{\mathrm{T}}$は正定値であり，$AA^{\mathrm{T}} = \Sigma$とおけば，$||A|| = |\Sigma|^{1/2}, (A^{-1})^{\mathrm{T}}A^{-1} = (AA^{\mathrm{T}})^{-1} = \Sigma^{-1}$．したがって，
$$
f_p(\boldsymbol{x}) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\left(- \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^{\mathrm{T}} \Sigma^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
$$
となる．密度関数が上式で与えられるような確率分布を多変量正規分布といい，$N_p(\boldsymbol{\mu}, \Sigma)$と表す．

#### 多変量正規分布の性質
$\boldsymbol{X} \sim N_p(\boldsymbol{\mu}, \Sigma)$とする．
1. $\mathbb{E}[\boldsymbol{X}] = \boldsymbol{\mu}, \; \mathrm{Var}[\boldsymbol{X}] = \Sigma, \; M_X(\boldsymbol{t}) \coloneqq \mathbb{E}[e^{\boldsymbol{t}^{\mathrm{T}}\boldsymbol{X}}] = \exp\left(\boldsymbol{\mu}\boldsymbol{t} + \frac{1}{2}\boldsymbol{t}^{\mathrm{T}}\Sigma\boldsymbol{t} \right)$
2. $A: p \times q$, $\boldsymbol{b}$: $q$-ベクトル．このとき，
$$
A\boldsymbol{X} + \boldsymbol{b} \sim N_q(A \boldsymbol{\mu} + \boldsymbol{b}, A \Sigma A^{\mathrm{T}})
$$

$\boldsymbol{X} \sim N_p(\boldsymbol{\mu}, \Sigma)$とし，$p_1 + p_2 = p$とする．
$$
\boldsymbol{X} = \left[
        \begin{array}{c}
            \boldsymbol{X}_1 \\
            \boldsymbol{X}_2
        \end{array}
    \right], \; \boldsymbol{\mu} = \left[
        \begin{array}{c}
            \boldsymbol{\mu}_1 \\
            \boldsymbol{\mu}_2
        \end{array}
    \right], \; \Sigma = \left[
        \begin{array}{cc}
            \Sigma_{11} & \Sigma_{12} \\
            \Sigma_{21} & \Sigma_{22}
        \end{array}
    \right]
$$
と書くと($\boldsymbol{X}_i$: $p_i$次確率ベクトル)，
- $\mathrm{Cov}(\boldsymbol{X}_1, \boldsymbol{X}_2) = \Sigma_{12}$
- $\boldsymbol{X}_1 \sim N_{p_1}(\boldsymbol{\mu}_1, \Sigma_{11}), \; \boldsymbol{X}_2 \sim N_{p_2}(\boldsymbol{\mu}_2, \Sigma_{22})$
- $\boldsymbol{X}_{1} \mathop{\perp \!\!\!\!\perp} \boldsymbol{X}_2 \Longleftrightarrow \Sigma_{12} = O$
- $A_1, A_2$: 定数行列．$A_1 \boldsymbol{X}_1 \mathop{\perp \!\!\!\!\perp} A_2 \boldsymbol{X}_2 \Longleftrightarrow A_1 \Sigma A_2^{\mathrm{T}} = O$

このように分割すると，$\boldsymbol{X}_1 = \boldsymbol{x}_1$が与えられた下での$\boldsymbol{X}_2$の条件付分布は以下で与えられる．
$
\frac{1}{(2\pi)^{p_2/2}|\Sigma_{22.1}|^{1/2}} \exp\left( - \frac{1}{2} \left(\boldsymbol{x}_2 - \boldsymbol{\mu}_2 - \Sigma_{21}\Sigma_{11}^{-1} (\boldsymbol{x}_1 - \boldsymbol{\mu}_1)^{\mathrm{T}}\right) \times \Sigma_{22.1}(\boldsymbol{x}_2 - \boldsymbol{\mu}_2 - \Sigma_{21}\Sigma_{11}^{-1}(\boldsymbol{x}_1 - \boldsymbol{\mu}_1)) \right)
$
すなわち，$\boldsymbol{X}_2 |_{\boldsymbol{X}_1 = \boldsymbol{x}_1} \sim N_{p_2}(\boldsymbol{\mu}_2 + \Sigma_{21}\Sigma_{11}^{-1}(\boldsymbol{x}_1 - \boldsymbol{\mu}_1), \Sigma_{22.1})$であり，特に，
$$
\mathbb{E}[\boldsymbol{X}_2 | \boldsymbol{X}_1 = \boldsymbol{x}_1] = \boldsymbol{\mu}_2 + \Sigma_{21}\Sigma_{11}^{-1}(\boldsymbol{x}_1 - \boldsymbol{\mu}_1), \; \mathrm{Var}[\boldsymbol{X}_2 | \boldsymbol{X}_1 = \boldsymbol{x}_1] = \Sigma_{22.1}
$$
である．

ここで，$p = 2$を考える．このとき，
$$
\Sigma = \left[
    \begin{array}{cc}
        \sigma_1^2 & \rho \sigma_1 \sigma_2 \\
        \rho \sigma_1 \sigma_2 & \sigma_2^2
    \end{array}
    \right], \; |\rho| < 1, \sigma_1 > 0, \sigma_2 > 0
$$
と書くことにする．このとき，$|\Sigma| = \sigma_1^2 \sigma_2^2 (1 - \rho^2)$であり，
$$
\Sigma^{-1} = \frac{1}{\sigma_1^2 \sigma_2^2} \left[
    \begin{array}{cc}
        \sigma_2^2 & -\rho \sigma_1 \sigma_2 \\
        -\rho \sigma_1 \sigma_2 & \sigma_1^2
    \end{array}
    \right]
$$
である．よって，密度関数$f_2(\boldsymbol{x})$は，
$$
f_2(\boldsymbol{x}) = \frac{1}{2 \pi \sigma_1^2 \sigma_2^2 (1 - \rho^2)} \exp\left( - \frac{1}{2(1 - \rho^2)} h(x_1, x_2) \right)
$$
ただし，
$$
h(x_1, x_2) = \frac{(x_1 - \mu_1)^2}{\sigma_1^2} - 2 \rho \frac{x_1 - \mu_1}{\sigma_1} \frac{x_2 - \mu_2}{\sigma_2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2}
$$
である．2変量正規分布の等高線を以下に示す．ただし，$\mu_1 = \mu_2 = 5, \sigma_1 = \sigma_2 = 1, \rho = 1/3$としている．

```python {cmd="/home/toda/.local/share/virtualenvs/MyInterest-csCBqNJG/bin/python" matplotlib=true hide}
import numpy as np
import matplotlib.pyplot as plt
def dist_norm(x1, x2, mu1, mu2, s1, s2, rho):
    norm_term = 1/(2*np.pi*s1*s2*np.sqrt(1 - rho**2))
    t1 = (x1 - mu1)**2/s1**2
    t2 = -2*rho*(x1 - mu1)*(x2 - mu2)/(s1*s2)
    t3 = (x2 - mu2)**2/s2**2
    dist_term = np.exp(-(t1+t2+t3)/(2*(1 - rho**2)))
    return norm_term*dist_term
def exp_conditioned(x1, x2, mu1, mu2, s1, s2, rho):
    s11 = s1**2; s12 = rho*s1*s2; s22 = s2**2
    ex1_x2 = mu1 + s12*s22**(-1)*(x2 - mu2)
    ex2_x1 = mu2 + s12*s11**(-1)*(x1 - mu1)
    return [ex1_x2, ex2_x1]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
x1 = np.linspace(0, 8, 51); x2 = np.linspace(0, 8, 51)
X1_grid, X2_grid = np.meshgrid(x1, x2)
value = dist_norm(x1=X1_grid, x2=X2_grid, mu1=5, mu2=5, s1=1, s2=1, rho=1/3)
cont = ax.contour(X1_grid, X2_grid, value); cont.clabel(fmt='%1.2f')
exp_val = exp_conditioned(x1=x1, x2=x2, mu1=5, mu2=5, s1=1, s2=1, rho=1/3)
ax.plot(x1, exp_val[0], c = 'r', label = r"$E[x_2 | x_1]$")
ax.plot(exp_val[1], x2, c = 'b', label = r"$E[x_1 | x_2]$")
ax.legend(loc = 'best')
plt.show()
```
このとき，
$$
\mathbb{E}[x_2 | x_1] = \frac{10}{3} + \frac{1}{3}x_1, \; \mathbb{E}[x_1 | x_2] = \frac{10}{3} + \frac{1}{3}x_2
$$
である．

### 正規分布の2次形式の分布
#### カイ2乗分布
$X_1, \ldots, X_n  \overset{\text{i.i.d.}}{\sim} N(0, 1)$とする．$\sum_{i = 1}^n X_i^2$の分布を自由度$n$のカイ二乗分布といい，$\chi_n^2$で表す．$\chi_n^2$の確率密度関数は，
$$
f_{\chi_n^2}(x) = \frac{1}{2^{n/2}\Gamma(n/2)} x^{n/2 - 1} e^{-x/2} \; (x > 0)
$$
であり，積率母関数は，
$$
M_{\chi_n^2}(x) = \mathbb{E}[e^{t\chi_n^2}] = (1 - 2t)^{-n/2} \; (t < 1/2>)
$$
である．ただし，$\Gamma(\cdot)$はガンマ関数である．
$\chi_n^2$の確率密度関数$(n = 1, 2, 3, 4, 5)$を以下に示す．
```python {cmd="/home/toda/.local/share/virtualenvs/MyInterest-csCBqNJG/bin/python" matplotlib=true hide}
import numpy as np
from math import gamma
import matplotlib
import matplotlib.pyplot as plt
colormap = list(matplotlib.colors.cnames)
def density_chi(x, n):
    val = x**(n/2 - 1)*np.exp(-x/2)
    pos = 2**(n/2)*gamma(n/2)
    return val/pos
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel(r"$x$"); ax.set_ylabel("density")
ax.set_xlim([0, 5]); ax.set_ylim([0, 1])
x = np.linspace(0.01, 5, 500)
for n in range(1, 6):
    y = density_chi(x, n)
    ax.plot(x, y, c = colormap[n + 10], label = r"n = {}".format(n))
ax.legend(loc = 'best'); ax.grid(True)
plt.show()
```

#### Wishart分布
$\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n  \overset{\text{i.i.d.}}{\sim} N(\boldsymbol{0}, \Sigma)$とする．$X = \sum_{i = 1}^n \boldsymbol{X}_i \boldsymbol{X}_i^{\mathrm{T}}$の分布をWishart分布といい，$W_p(n, \Sigma)$で表す．$\Sigma > 0$, $n \geq p$のとき$W_p(n, \Sigma)$の確率密度関数が存在し，次式で与えられる．
$$
c_{np}^{-1}|\Sigma|^{-\frac{n}{2}} |X|^{\frac{1}{2}(n - p - 1)} \exp\left(- \frac{1}{2}\mathrm{tr}[\Sigma^{-1}X]\right) \; (X > 0)
$$
ここで$c_{np}$は正規か定数で，
$$
c_{np} = 2^{np/2}\pi^{p(p - 1)/4}\prod_{i = 1}^p \Gamma\left[\frac{1}{2}(n + 1 - i) \right]
$$
である．
Wishart分布は，ベイジアンモデリングなどで多変量正規分布の共分散行列の逆行列 (精度行列) を生成するための共役事前分布として使われる．

#### $\chi^2$分布に従うための必要十分条件
$\boldsymbol{X} \sim N_p(\boldsymbol{0}, \Sigma)$, $A \in \mathbb{R}^{p \times p}$, 対称．
$$
\boldsymbol{X}^{\mathrm{T}} A \boldsymbol{X} \sim \chi_q^2 \Longleftrightarrow \Sigma A \Sigma A \Sigma = \Sigma A \Sigma, \; q = \mathrm{tr}[A \Sigma]
$$
この必要十分条件は，$\Sigma$が正則のとき$A\Sigma A = A, \Sigma = I_p$のとき$A^2 = A$となる．

#### Cochran's Theorem
$A_i \in \mathbb{R}^{p \times p}$, 対称，$I_p = \sum_{i = 1}^n A_i$, $\boldsymbol{X} \sim N_p(\boldsymbol{0}, I_p), Q_i \coloneqq \boldsymbol{X}^{\mathrm{T}}A_i \boldsymbol{X} \; (i = 1, \ldots, n)$とする．
$$
Q_i \sim \chi^2 \; (i = 1, \ldots, n) \Longleftrightarrow \sum_{i = 1}^n \mathrm{rank}(A_i) = p
$$
このとき，$Q_i \; (i = 1, \ldots, n)$は互いに独立である．

## 確率変数の収束
$\left\{\boldsymbol{X}_n = [X_1^{(n)}, \ldots, X_p^{(n)}]^{\mathrm{T}} \right\}$: $p$次元確率変数の列
1. $\boldsymbol{a} \in \mathbb{R}^p$．$\forall \varepsilon > 0$に対して
$$
\lim_{n \to \infty} \mathbb{P}[\|\boldsymbol{X}_n - \boldsymbol{a}\| < \varepsilon] = 1
$$
が成り立つとき，$\boldsymbol{X}_n$は$\boldsymbol{a}$に確率収束するといい，$\boldsymbol{X}_n \overset{P}{\longrightarrow} \boldsymbol{a}$ $(n \to \infty)$と表す ($\|\boldsymbol{X}\| = \sqrt{\boldsymbol{X}^{\mathrm{T}}\boldsymbol{X}}$)．

2. $\boldsymbol{X} = [X_1, \ldots, X_p]^{\mathrm{T}}$: $p$次元確率変数．
$\mathbb{P}[X_k = x_k] = 0$なる任意の実数$x_k (k = 1, \ldots, p)$に対して，
$$
\lim_{n \to \infty} \mathbb{P}[X_1^{(n)} \leq x_1, \ldots, X_p^{(n)} \leq x_p] = \mathbb{P}[X_1 \leq x_1, \ldots, X_p \leq x_p]
$$
が成り立つとき，$\boldsymbol{X}_n$は$\boldsymbol{X}$に分布収束するといい，$\boldsymbol{X}_n \overset{d}{\longrightarrow} \boldsymbol{X}$ $(n \to \infty)$と表す．$F(\boldsymbol{x})$を$\boldsymbol{X}$の分布関数とするとき，分布収束を$\boldsymbol{X}_n \overset{d}{\longrightarrow} F(\boldsymbol{x})$と書くこともある．$F(\boldsymbol{x})$を分布名称(e.g. $N(0, 1)$)で置き換えることもある．

- 分布関数列の収束先は分布関数とは限らない
- 分布関数列が連続な分布関数に各点収束するならば，それは一様である．
$$
\lim_{n \to \infty} \sup_{x \in \mathbb{R}} |F_n(\boldsymbol{x}) - F(\boldsymbol{x})| = 0
$$

### 大数の法則と中心極限定理
$\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n$: 独立同一分布をもつ$p$次元確率変数．$\mathbb{E}[\boldsymbol{X}_i] = \boldsymbol{\mu}, \; \mathrm{Var}(\boldsymbol{X}_i) = \Sigma$
$$
\bar{\boldsymbol{X}}_n \coloneqq \frac{1}{n} \sum_{i = 1}^n \boldsymbol{X}_i
$$
とおく．
1. $\mathbb{E}[\bar{\boldsymbol{X}}_n] = \boldsymbol{\mu}, \; \mathrm{Var}(\bar{\boldsymbol{X}}_n) = \frac{1}{n}\Sigma$
2. (大数の法則) $\bar{\boldsymbol{X}}_n \overset{P}{\longrightarrow} \boldsymbol{\mu} \; (n \to \infty)$
3. (中心極限定理)
$$
\sqrt{n} (\bar{\boldsymbol{X}}_n - \boldsymbol{\mu}) \overset{d}{\longrightarrow} N_p(\boldsymbol{0}, \Sigma) \; \; (n \to \infty)
$$