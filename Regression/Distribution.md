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
$$
\mathbb{P}(\boldsymbol{X} \in B_1, \boldsymbol{Y} \in B_2) = \mathbb{P}(\boldsymbol{X} \in B_1) \mathbb{P}(\boldsymbol{Y} \in B_2)
$$
が成立するとき，$\boldsymbol{X}$と$\boldsymbol{Y}$は互いに独立であるといい，$\boldsymbol{X} \mathop{\perp \!\!\!\!\perp} \boldsymbol{Y}$と表す．
$$
\boldsymbol{X} \mathop{\perp \!\!\!\!\perp} \boldsymbol{Y} \Longleftrightarrow 任意の有界連続関数\alpha(\cdot), \beta(\cdot)に対して \\
\hspace{12mm} \mathbb{E}[\alpha(\boldsymbol{X})\beta(\boldsymbol{Y})] = \mathbb{E}[\alpha(\boldsymbol{X})]\mathbb{E}[\beta(\boldsymbol{Y})]
$$
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

#### エントロピー
確率密度関数$p(\boldsymbol{x})$に対して，
$$
H[p(\boldsymbol{x})] = - \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} = -\mathbb{E}[\ln p(\boldsymbol{x})]
$$
をエントロピーと呼ぶ．エントロピーは確率分布の乱雑さを表す指標として知られている．

#### KLダイバージェンス
2つの確率密度関数$p(\boldsymbol{x})$および$q(\boldsymbol{x})$に対して，
$$
\mathrm{KL}[q(\boldsymbol{x}) \| p(\boldsymbol{x})] = -\int q(\boldsymbol{x}) \ln \frac{p(\boldsymbol{x})}{q(x)} d\boldsymbol{x} = \mathbb{E}_q[\ln q(\boldsymbol{x})] - \mathbb{E}_q[\ln p(\boldsymbol{x})]
$$
をKLダイバージェンスと呼ぶ．任意の確率分布の組に対して$\mathrm{KL}[q(\boldsymbol{x}) \| p(\boldsymbol{x})] \geq 0$である．等号が成り立つのは，2つの分布が完全に一致する場合，i.e., $p(\boldsymbol{x}) = q(\boldsymbol{x})$に限られる．KLダイバージェンスは，2つの確率分布の距離を表していると解釈できるが，距離の公理は満たしていない．