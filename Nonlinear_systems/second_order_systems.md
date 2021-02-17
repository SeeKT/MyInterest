# 非線形システム
- Kosuke Toda (@SeeKT)
### 参考
- H. K. Khalil, Nonlinear systems third edition, 2002.

## 2. Second-Order Systems
2次元のシステムを考える．
$$
\begin{cases}
\dot{x}_1 = f_1(x_1, x_2) = f_1(\boldsymbol{x}) \\
\dot{x}_2 = f_2(x_1, x_2) = f_2(\boldsymbol{x})
\end{cases}
$$
$\boldsymbol{x}(t) = (x_1(t), x_2(t))$を初期状態$\boldsymbol{x}_0 = (x_{10}, x_{20})$からの解とする．このとき，$x_1-x_2$平面における$\boldsymbol{x}(t)$の解の軌跡は，$\forall t \geq 0$で$\boldsymbol{x}_0$からの曲線になる．この曲線のことをtrajectoryまたはorbitという．
さらに，$x_1-x_2$平面のことをstate planeやphase planeという．
軌跡の族のことをphase portraitという．

ベクトル場$f(\boldsymbol{x}) = (f_1(\boldsymbol{x}), f_2(\boldsymbol{x}))$を考える．
任意の$\boldsymbol{x} = (x_1, x_2)$において，$f(\boldsymbol{x}) = (f_1(\boldsymbol{x}), f_2(\boldsymbol{x}))$は$\boldsymbol{x}$におけるtrajectoryの接戦になっている．
$$
\frac{dx_2}{dx_1} = \frac{f_2(\boldsymbol{x})}{f_1(\boldsymbol{x})}
$$
- $x_1$方向に$f_1(\boldsymbol{x})$，$x_2$方向に$f_2(\boldsymbol{x})$だけ変化．

### 線形システムの定性的振る舞い
非線形システムにおいて，平衡点付近の安定性に着目する．
いま，$2 \times 2$のシステム
$$
\dot{\boldsymbol{x}} = A \boldsymbol{x}, \; \; A \text{ is a } 2 \times 2 \text{ real matrix}
$$
を考える．$A = M J_r M^{-1}$とし，
$$
\boldsymbol{x}(t) = M \exp(J_r t) M^{-1}\boldsymbol{x}_0
$$
と書くことができるとする．ただし，
$$
J_r = \left[
    \begin{array}{cc}
    \lambda_1 & 0 \\
    0 & \lambda_2
    \end{array}
    \right] \text{ or } \left[
    \begin{array}{cc}
    \lambda & 0 \\
    0 & \lambda
    \end{array}
    \right] \text{ or } \left[
    \begin{array}{cc}
    \lambda & 1 \\
    0 & \lambda
    \end{array}
    \right] \text{ or } \left[
    \begin{array}{cc}
    \alpha & -\beta \\
    \beta & \alpha
    \end{array}
    \right]
$$
とする$(\lambda_1, \lambda_2, \lambda, \alpha, \beta \in \mathbb{R})$．

変数変換$\boldsymbol{x}(t) = M\boldsymbol{z}(t)$により，
$$
M \dot{\boldsymbol{z}}(t) = AMz(t), \; \; \therefore \dot{\boldsymbol{z}}(t) = J_r\boldsymbol{z}(t)
$$
となる．

#### Case1. $A$の固有値が実数のとき $(\lambda_1 \neq \lambda_2 \neq 0)$
固有値$\lambda_1, \lambda_2$に属する固有ベクトルを$\boldsymbol{v}_1, \boldsymbol{v}_2$とし，$M \coloneqq [\boldsymbol{v}_1, \boldsymbol{v}_2]$とすればよい．
$\boldsymbol{z} = (z_1, z_2)^T$とすると，
$$
\dot{z}_1 = \lambda_1 z_1, \; \dot{z}_2 = \lambda_2 z_2
$$
より，
$$
z_1(t) = z_{10} e^{\lambda_1 t}, \; z_2(t) = z_{20} e^{\lambda_2 t}
$$
から，
$$
z_2 = c z_1^{\lambda_2 / \lambda_1}, \; c = z_{20}/(z_{10})^{\lambda_2/\lambda_1}
$$
よって，phase portraitの形は$\lambda_1, \lambda_2$の符号による．

##### (i) $\lambda_2 < \lambda_1 < 0$のとき
$t \to \infty$で$e^{\lambda_1 t}, e^{\lambda_2 t} \to 0$である．また，$e^{\lambda_2 t}$の方が$e^{\lambda_1 t}$よりも早く$0$に近づく．
このとき，$\lambda_2$を"fast eigenvalue"，$\lambda_1$を"slow eigenvalue"といい，固有ベクトルもfast/slow eigenvectorという．

解軌道は，$z_2 = cz^{\lambda_2 / \lambda_1}$ $(\lambda_2 / \lambda_1 > 1)$ で，原点に向かう．
$$
\frac{dz_2}{dz_1} = c\frac{\lambda_2}{\lambda_1}z_1^{\frac{\lambda_2}{\lambda_1} - 1}
$$
である．このとき，stable nodeという．いま，$z_2$方向はより早く$0$へ向かう．

![stable_node](./fig/stable_node.png)

##### (ii) $\lambda_2 > \lambda_1 > 0$のとき
解軌道は，$\infty$に発散．上の図の矢の向きが逆になる．このとき，unstable node．

##### (iii) $\lambda_2 < 0 < \lambda_1$のとき
$e^{\lambda_1 t} \to \infty, \; e^{\lambda_2 t} \to 0$ as $t \to \infty$．
このとき，$\lambda_2$をstable eigenvalue, $\lambda_1$をunstable eigenvalueといい，固有ベクトルもstable / unstable eigenvalueという．
このとき，Saddleという．
![saddle_node](./fig/saddle.png)
この図の場合は，初めは$z_2$軸方向から$0$に近づき，最終的に$z_1$方向の無限大に発散している．

#### Case2. $A$の固有値が複素数のとき $\lambda_{1, 2} = \alpha \pm j \beta$
このとき，
$$
\dot{z}_1 = \alpha z_1 - \beta z_2, \; \dot{z}_2 = \beta z_1 + \alpha z_2
$$
である．極座標変換を
$$
r = \sqrt{z_1^2 + z_2^2}, \; \theta = \tan^{-1}\left(\frac{z_2}{z_1} \right)
$$
とする．解は，
$$
r(t) = r_0 e^{\alpha t} \text{ and } \theta(t) = \theta_0 + \beta t
$$
と得られる．よって，
- $\alpha < 0$ $\Rightarrow$ $r(t) \to 0$ as $t \to \infty$
- $\alpha > 0$ $\Rightarrow$ $r(t) \to \infty$ as $t \to \infty$
- $\alpha = 0$ $\Rightarrow$ $r(t) \equiv r_0 \; \forall t$
である．それぞれStable Focus, Unstable Focus, Centerという．
![complex_eigen](./fig/complex_eigen.png)

### 摂動の影響
$$
A \to A + \delta A \; \; (\delta A \text{ arbitrarily small})
$$
とする．この行列の固有値は$\delta A$の変化によって連続的に変化する．
#### structurally stable
A node (with distinct eigenvalues), a saddle or a focusは，structurally stable．$\delta A$の微小変動によって定性的な振る舞いは変化しない

A stable node (重複度2以上) は微小変動により，
- Stable node
- Stable focus
になる可能性がある．

例: $A$の固有値が$-1$ (重複度2)のとき，固有多項式は，
$$
\Delta (\lambda) = \lambda^2 + 2 \lambda + 1
$$
である(重解)．このとき，$A$の微小変化により，$\Delta (\lambda) = 0$は，
- 2実解をもつ
- 共役複素解を持つ

ことの2つの場合が考えられる．

微小変動において，固有値の実部の正負は変化しない．よって，Stable NodeまたはStable Focus．

また，固有値が純虚数のCenterはstructually stableにならない．
$$
\left(
    \begin{array}{cc}
    0 & 1 \\
    -1 & 0
    \end{array}
    \right) \rightarrow \left(
    \begin{array}{cc}
    \mu & 1 \\
    -1 & \mu
    \end{array}
    \right)
$$
に微小変動したとする．固有値は$\mu \pm j$となる．
- $\mu < 0$: Stable Focus
- $\mu > 0$: Unstable Focus