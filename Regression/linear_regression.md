# 線形回帰
- Kosuke Toda (@SeeKT)
### 参考
- 佐和，回帰分析 新装版，朝倉書店，2020．
- 稲垣，数理統計学 改訂版，裳華房，2003.
- 金森，Pythonで学ぶ統計的機械学習，オーム社，2018．
- 狩野，2021年度多変量解析 講義資料 (大阪大学 大学院基礎工学研究科)

## 単回帰分析
確率変数$Y$の互いに独立な$n$個の観測値$Y_1, \ldots, Y_n$を得たとする．この観測値が，ある変数の変動によって説明されると仮定し，
$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i \; (i = 1, \ldots, n)
$$
とする．ただし，$\varepsilon_1, \ldots, \varepsilon_n \overset{\text{i.i.d.,}}{\sim} \mathbb{E}[\varepsilon_i] = 0, \; \mathrm{Var}[\varepsilon_i] = \sigma^2$とする．
データ$(x_1, y_1), \ldots, (x_n, y_n)$を用いて，$\beta_0, \beta_1, \sigma^2$を推定することが目標である．

#### Notation
- $\bar{x} \coloneqq \frac{1}{n} \sum_{i = 1}^n x_i, \; \bar{y} \coloneqq \frac{1}{n} \sum_{i = 1}^n y_i$: 標本平均
- $s_x^2 \coloneqq \frac{1}{n}\sum_{i = 1}^n (x_i - \bar{x})^2 = \frac{1}{n}\sum_{i = 1}^n x_i^2 - \bar{x}^2$: $x$の標本分散
- $s_{xy} \coloneqq \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x}) (y_i - \bar{y}) = \frac{1}{n}\sum_{i = 1}^n x_i y_i - \bar{x}\bar{y}$: $x, y$の標本共分散
- $r = \frac{s_{xy}}{s_x s_y}$: 標本相関係数

### 最小二乗法
$$
\Delta^2(\beta_0, \beta_1) = \sum_{i = 1}^n \varepsilon_i^2 = \sum_{i = 1}^n (y_i - \beta_0 - \beta_1 x_i)^2
$$
とおく．このとき，
$$
\Delta^2(\tilde{\beta}_0, \tilde{\beta}_1) = \min_{\beta_0, \beta_1} \Delta^2(\beta_0, \beta_1)
$$
として$\beta_0, \beta_1$を求める方法を最小二乗法という．
$\Delta^2(\beta_0, \beta_1)$を最小にする$\beta_0, \beta_1$を求めるために，$\beta_0, \beta_1$それぞれについて微分して$0$とおく．
$
\frac{\partial \Delta^2}{\partial \beta_0} = -2\sum_{i = 1}^n (y_i - \beta_0 - \beta_1 x_i) = 0
$

$
\frac{\partial \Delta^2}{\partial \beta_1} = -2\sum_{i = 1}^n (y_i - \beta_0 - \beta_1 x_i)x_i = 0
$
これを$\beta_0, \beta_1$の連立方程式として整理すると，
$$
\begin{cases}
\bar{y} - \beta_0 - \beta_1 \bar{x} = 0 \\
\frac{1}{n} \sum_{i = 1}^n x_i y_i - \beta_0 \bar{x} - \beta_1 \sum_{i = 1}^n x_i^2 = 0
\end{cases}
$$
この連立方程式を正規方程式という．分散共分散公式を用いると，$\beta_0, \beta_1$の解$\hat{\beta}_0, \hat{\beta}_1$は，
$$
\hat{\beta}_1 = \frac{s_{xy}}{s_x^2}, \; \hat{\beta}_0 = \bar{y} - \frac{s_{xy}}{s_x^2}\bar{x}
$$
となる．よって，推定された回帰直線は
$$
\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x = \bar{y} + \frac{s_{xy}}{s_x^2}(x - \bar{x})
$$
となる．これを回帰直線という．観測値$y$に対して推定値$\hat{y}$を予測値ということもある．観測値と予測値の差を残差 (residual) といい，$e$で表す．残差$e$は誤差$\varepsilon$の推定値である．
$$
e = (\beta_0 - \hat{\beta}_0) + (\beta_1 - \hat{\beta}_1)x + \varepsilon \approx \varepsilon
$$

#### 回帰直線の性質
1. $x$と$e$の標本共分散を
$$
s_{xe} \coloneqq \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x}) (e_i - \bar{e})
$$
とすると，$s_{xe} = 0$．つまり，説明変数$x$と残差$e$は直交する．
2. $\hat{y}$と$e$の標本共分散を$s_{\hat{y}e}$とすると，$s_{\hat{y}e} = 0$．つまり，予測値$\hat{y}$と残差$e$は直交する．
3. $s_y^2 = s_{\hat{y}}^2 + s_e^2$

3について補足する．観測値偏差の二乗和$ns_y^2$を観測変動または総変動 (total sum of squares; TSS)，予測値偏差の二乗和$ns_{\hat{y}}^2$を予測変動または回帰変動，残差の二乗和$ns_e^2 = \Delta_0^2$を残差平方和 (residual sum of square; RSS) または残差変動という．3は，総変動が回帰変動と残差変動の和に分解されることを意味する．
$
(総変動) = (回帰変動) + (残差変動)
$

$
\sum_{i = 1}^n (y_i - \hat{y})^2 = \sum_{i = 1}^n (\hat{y}_i - \bar{\hat{y}})^2 + \sum_{i = 1}^n (e_i - \bar{e})^2
$

$
ns_y^2 = ns_y^2 r^2 + ns_y^2(1 - r^2)
$

ここで，標本相関係数$r$について，
- $|r| \fallingdotseq 1$のとき，(残差変動) $\approx 0$ (当てはまりが良い)
- $|r| \fallingdotseq 0$のとき，(回帰変動) $\approx 0$

#### 最小二乗推定量の分布の性質
モデルを
$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i \; (i = 1, \ldots, n),
$$
ただし，$Y_1, \ldots, Y_n$は独立な確率変数とし，$\varepsilon_1, \ldots, \varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathbb{E}[\varepsilon_i] = 0, \; \mathrm{Var}[\varepsilon_i] = \sigma^2$ (未知) とする．
このとき，$\beta_0$と$\beta_1$の最小二乗推定量は，
$$
\tilde{\beta}_1 = \frac{s_{xY}}{s_x^2} \eqqcolon \hat{\beta}_1, \; \; \tilde{\beta}_0 = \bar{Y} - \frac{s_{xY}}{s_x^2}\bar{x} \eqqcolon \hat{\beta}_0
$$
で，推定回帰直線は
$$
\hat{Y} = \bar{Y} + \frac{s_{xY}}{s_x^2} (x - \bar{x})
$$
である．ただし，
$$
s_{xY} = \frac{1}{n}\sum_{i = 1}^n (x_i - \bar{x})(Y_i - \bar{Y}) = \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})Y_i
$$
である．

1. - $\mathbb{E}[\hat{\beta}_0] = \beta_0, \; \mathbb{E}[\hat{\beta}_1] = \beta_1$ (不偏推定量)
    - $\mathrm{Var}[\hat{\beta}_0] = \frac{\sigma^2}{n} \left(1 + \frac{(\bar{x})^2}{s_x^2} \right), \; \mathrm{Var}[\hat{\beta}_1] = \frac{\sigma^2}{n}\frac{1}{s_x^2}, \; \mathrm{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\frac{\sigma^2}{n} \frac{\bar{x}}{s_x^2}$
2. - $\mathrm{E}[\hat{Y}] = a + bx ( = y)$ (不偏推定量)
    - $\mathrm{Var}[\hat{Y}] = \frac{\sigma^2}{n} \left(1 + \left(\frac{x - \bar{x}}{s_x} \right)^2 \right)$

ここで，誤差の分散$\sigma^2$の不偏推定量は残差平方和$\Delta_0^2 = \Delta^2(\hat{\beta}_0, \hat{\beta}_1)$から次のように求められる．
$$
\mathbb{E}\left[\frac{1}{n - 2} \Delta_0^2 \right] = \sigma^2
$$
つまり，$\Delta_0^2/(n - 2)$が$\sigma^2$の不偏推定量である，ということである．これは直感的には，残差平方和を求めるとき，2個の推定量$\hat{\beta}_0, \hat{\beta}_1$にデータを使っているので，$n - 2$で割っていると考えることができる．


#### 正規誤差の場合
これまでは，線形回帰モデル
$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \; (i = 1, \ldots, n)
$$
に対して，$\varepsilon_1, \ldots, \varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathbb{E}[\varepsilon_i] = 0, \; \mathrm{Var}[\varepsilon_i] = \sigma^2$を仮定していた．以降，$\varepsilon_1, \ldots, \varepsilon_n \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$を仮定する．

このとき，線形回帰モデルは次の性質を持つ．
1. $Y_i \sim N(\beta_0 + \beta_1 x_i, \sigma^2)$, $Y_1, \ldots, Y_n$は互いに独立．
2. $Y_n = \frac{1}{n} \sum_{i = 1}^n Y_i \sim N\left(\beta_0 + \beta_1 x_i, \frac{\sigma^2}{n} \right)$
3. $s_{xY} \sim N\left(\beta_1 s_x^2, \frac{\sigma^2}{n}s_x^2 \right)$
4. $\bar{Y}$と$s_{xY}$は独立

また，正規誤差を仮定すると，次の性質が成り立つ．
1. $(\hat{\beta}_0, \hat{\beta}_1)$と$\Delta_0^2$は独立
2. $\hat{\beta}_0 \sim N\left(\beta_0, \frac{\sigma^2}{n} (1 + \frac{\bar{x}^2}{s_x^2}) \right), \; \hat{\beta}_1 \sim N\left(\beta_1, \frac{\sigma^2}{n s_x^2} \right), \; \mathrm{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\frac{\sigma^2 \bar{x}}{n s_x^2}$
3. $\frac{\Delta_0^2}{\sigma^2} \sim \chi_{n - 2}^2$

3つ目の性質から，回帰係数の最小二乗推定量の$t$-変換$T_{\beta_0}, T_{\beta_1}$について，以下が成り立つ．
$$
T_{\beta_0} = \frac{\hat{\beta}_0 - \beta_0}{\sqrt{\frac{\Delta_0^2}{n(n - 2)} \left(1 + \frac{\bar{x}^2}{s_x^2}\right)^2 }} \sim t_{n - 2}, \; T_{\beta_1} = \frac{\hat{\beta}_1 - \beta_1}{\sqrt{\frac{\Delta_0^2}{n(n - 2)s_x^2}}} \sim t_{n - 2}
$$
これを用いて回帰係数の推定値の検定を行うことができる．

#### 単回帰の実装
Bostonの住宅価格に関するsklearnのデータセットを用いて単回帰分析を行う．
(url: https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset)
ここでは，説明変数を平均部屋数(RM)，目的変数を住宅価格(price)として単回帰分析を行う．つまり，説明変数をRM，目的変数をpriceとしたときの回帰係数$\hat{\beta}_0, \hat{\beta}_1$を求める．

statsmodels.apiの線形回帰モデルを用いると，回帰係数の推定値の検定まで行ってくれる．
(url: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
今回は，これを用いて実装した．


```python {cmd="/home/toda/.local/share/virtualenvs/MyInterest-csCBqNJG/bin/python" matplotlib=true}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston

######### Regression using statsmodels.api #########
def execute_regression(exp_variable, obj_variable):
    # add constants
    X = sm.add_constant(exp_variable)
    model = sm.OLS(obj_variable, X)
    # execute regression
    results = model.fit()
    # output results
    return results
####################################################

# read data from sample
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
obj_y = boston.target
expression_label = df.columns.values
data_label = ['price']; data_list = [obj_y]
for i in range(len(expression_label)):
    data_label.append(expression_label[i])
    data_list.append(df.iloc[:, i].values)
data_dict = dict(zip(data_label, data_list))
new_df = pd.DataFrame(data_dict)

# objective variable and expression variable
y = new_df.iloc[:, 0]; x = new_df.iloc[:, 6]
# correlation matrix
correlation_matrix = np.corrcoef(x.values, y.values)
# execute regression analysis
regression_result = execute_regression(exp_variable=x, obj_variable=y)
print(regression_result.tvalues)
b0 = regression_result.params[0]; b1 = regression_result.params[1]
# figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x.values, y.values, c = 'k', s = 5, label = 'corr = %.2f' % (correlation_matrix[0][1]))
ax.plot(x.values, b0 + b1 * x.values, c = 'r', label = r'$y = %.2f + %.2fx$' % (b0, b1))
ax.set_xlabel(new_df.columns.values[6])
ax.set_ylabel(new_df.columns.values[0])
ax.grid(True); ax.legend(loc = 'best')
plt.show()
```
$\hat{\beta}_0 = -34.67, \hat{\beta}_1 = 9.10$と求まり，それぞれの$t$-値は$-13.08, 21.72$である．つまり，係数の推定値$\hat{\beta}_0, \hat{\beta}_1$が棄却域に入るので，どの係数も有意に0より大きく，0とみなすことはできない．


## 重回帰分析
確率変数$Y$の互いに独立な$n$個の観測値$y_1, \ldots, y_n$を得たとする．この観測値が，比較的少数個のある変数の変動によって説明されると仮定し，
$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i \; \; (i = 1, \ldots, n)
$$
とする．ただし，$\varepsilon_1, \ldots, \varepsilon_n \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$とする．
これをベクトル表記で書き直す．
- $\boldsymbol{y} = X \boldsymbol{\beta} + \boldsymbol{\varepsilon}, \; \; \boldsymbol{\varepsilon} \sim N_n(\boldsymbol{0}, \sigma^2I_n)$
    $$
    X = \left[
        \begin{array}{cccc}
        1 & x_{11} & \cdots & x_{1p} \\
        \vdots & \vdots & \ddots & \vdots \\
        1 & x_{n1} & \cdots & x_{np}
        \end{array}
        \right]
    $$
- $\boldsymbol{y} \sim N_n(X\boldsymbol{\beta}, \sigma^2 I_n)$

ここで，次の仮定をおく．