# 非線形システム
- Kosuke Toda (@SeeKT)
### 参考
- H. K. Khalil, Nonlinear systems third edition, 2002.

## 1. Introduction
### 非線形モデルと非線形現象
#### 非線形システム
- $x_1, \ldots, x_n$: 状態変数
- $u_1, \ldots, u_p$: 入力変数

次の微分方程式で表されるダイナミカルシステムを扱う．
$ 
\dot{x}_1 = f_1(t, x_1, \ldots, x_n, u_1, \ldots, u_p)
$
$
\dot{x}_2 = f_2(t, x_1, \ldots, x_n, u_1, \ldots, u_p)
$
$
\hspace{20mm}\vdots
$
$ 
\dot{x}_n = f_n(t, x_1, \ldots, x_n, u_1, \ldots, u_p) 
$
次の記法を用いる．
- $\boldsymbol{x} = (x_1, \ldots, x_n)^T$
- $\boldsymbol{u} = (u_1, \ldots, u_p)^T$
- $f(t, \boldsymbol{x}, \boldsymbol{u}) = (f_1(t, \boldsymbol{x}, \boldsymbol{u}), \ldots, f_n(t, \boldsymbol{x}, \boldsymbol{u}))$

このとき，ダイナミクスは
$$
\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}, \boldsymbol{u})
$$
というcompact formで表される．

また，観測を$\boldsymbol{y} = h(t, \boldsymbol{x}, \boldsymbol{u})$とすると，システムのダイナミクスと出力は，
$
\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}, \boldsymbol{u})
$
$
\boldsymbol{y} = h(t, \boldsymbol{x}, \boldsymbol{u})
$
である．ただし，$\boldsymbol{y} \in \mathbb{R}^q$であり，$q \leq n$である．

特別な場合のダイナミクスと出力を示す．
##### 線形システム (linear system)
$
\dot{\boldsymbol{x}} = A(t)\boldsymbol{x} + B(t)\boldsymbol{u}
$
$
\boldsymbol{y} = C(t)\boldsymbol{x} + D(t)\boldsymbol{u}
$
##### 入力なし (unforced state equation)
$$
\dot{\boldsymbol{x}} = f(t, \boldsymbol{x})
$$
これは，入力$\boldsymbol{u}$が時間$t$および状態$\boldsymbol{x}$の関数として表されているものである．
- $\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}, \boldsymbol{u}), \; \boldsymbol{u} = \gamma(t, \boldsymbol{x})$
##### Autonomous system
$$
\dot{\boldsymbol{x}} = f(\boldsymbol{x})
$$
システムが時間および入力に依存しない．
##### Time-Invariant system
$
\dot{\boldsymbol{x}} = f(\boldsymbol{x}, \boldsymbol{u})
$
$
\boldsymbol{y} = h(\boldsymbol{x}, \boldsymbol{u})
$
初期時刻が$t_0$から$\tau_0 \neq t_0$に変わるが，ダイナミクスは時間シフトのみ．

![time_invariant](./fig/time_invariant_shift.svg)

#### 解の存在性と一意性
非線形システム$\dot{\boldsymbol{x}} = f(t, \boldsymbol{x})$においては，
- piecewise continuous in $t$
- locally Lipschitz in $\boldsymbol{x}$ over the domain

が重要．

##### piecewise continuous
$f(t, \boldsymbol{x})$がpiecewise continuous in $t$ on an interval $J \subset \mathbb{R}$
$\overset{\text{def}}{\Longleftrightarrow}$ 任意のboundedな部分区間$J_0 \subset J$において，$f$は有限個の点における有限ジャンプによる不連続を除き，全ての$t \in J_0$で連続である．
つまり，$x$を固定して$t$を動かしたとき，下図のような場合にpiecewise continuous．

![piecewise_continuous](./fig/piecewise_continuous_path.svg)

##### locally Lipschitz
$f(t, \boldsymbol{x})$が点$\boldsymbol{x}_0$において$\boldsymbol{x}$についてlocally Lipschitzである
$\overset{\text{def}}{\Longleftrightarrow}$ $\exists N(\boldsymbol{x}_0, r) = \{\boldsymbol{x} \in \mathbb{R}^n \, | \, \|\boldsymbol{x} - \boldsymbol{x}_0\| < r\}$, where $f(t, \boldsymbol{x})$ satisfies the Lipschitz condition
$$
\| f(t, \boldsymbol{x}) - f(t, \boldsymbol{y}) \| \leq \|\boldsymbol{x} - \boldsymbol{y} \|, \; L > 0.
$$
つまり，$t$を固定して$x$を動かしたとき，下図のような場合にlocally Lipschitz．
![locally_lipschitz](./fig/locally_lipschitz_cond.svg)
ここで，$L$は$\boldsymbol{x}_0$に依存することに注意．

$f(t, \boldsymbol{x})$がdomain (開かつ連結集合) $D \subset \mathbb{R}^n$において，$\boldsymbol{x}$についてlocally Lipschitzである
$\overset{\text{def}}{\Longleftrightarrow}$ $\forall \boldsymbol{x}_0 \in D$でlocally Lipschitzである．

$n = 1$で$f$が$x$のみに依存するときは，
$$
\frac{|f(y) - f(x)|}{y - x} \leq L
$$
である．これは，$f(x)$に任意の2点を結ぶ直線の傾きが$L$以下であるということである．

このことから，次の2つのことが分かる．
- ある点において接点の傾きが$\infty$となるような関数$f(x)$はその点においてlocally Lipschitzではない．
- 不連続な関数はその不連続点においてlocally Lipschitzではない．

例えば，$f(x) = x^{1/3}$は，$f^{\prime}(x) = x^{-2/3}/3$より，$x \to 0$で$f^{\prime}(x) \to \infty$であるから，原点でlocally Lipschitzではない．

一方，$f^{\prime}(x)$が点$x_0$で連続であれば$x_0$でlocally Lipshitzである．これは，$f^{\prime}(x)$の連続性より，$x_0$の近傍において$|f^{\prime}(x)|$が定数$k$でおさえられるので，$L = k$とすると条件を満たすからである．

より一般的に述べると，$t \in J \subset \mathbb{R}, \; \boldsymbol{x} \in D \in \mathbb{R}^n$において，偏微分$\partial f_i / \partial x_j$が連続ならば，$f(t, \boldsymbol{x})$は$D$において$\boldsymbol{x}$についてlocally Lipschitzである．

##### 解の存在性と一意性に関する補題
$f(t, \boldsymbol{x})$は$\forall t \in [t_0, t_1]$においてpiecewise continuousで，$\boldsymbol{x}_0$において$\boldsymbol{x}$についてlocally Lipschitzであるとする．このとき，状態方程式$\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}), \; \boldsymbol{x}(t_0) = \boldsymbol{x}_0$の$[t_0, t_0 + \delta]$における解が一意に存在するような$\delta > 0$が存在する．

例: local Lipschitz conditionを満たさないもの
$\dot{x} = x^{1/3}$に対し，初期状態$x(0) = 0$のときの解は，
$x(t) = (2t/3)^{3/2}, \; x(t) \equiv 0$であり，2つの異なる解を持つ．

この補題はlocal resultである．
- 区間$[t_0, t_0 + \delta]$における解の存在性と一意性を保証するが，一般に$[t_0, t_0 + \delta] \subset [t_0, t_1]$であるので，一定時間経過後に解が存在しなくなる可能性がある．

例: 
$$
\dot{x} = -x^2
$$
$f(x) = -x^2$は$\forall x$でlocally Lipschitzである．
$$
x(0) = -1 \Rightarrow x(t) = \frac{1}{t - 1}
$$
である．このとき，この解は$t = 1$でfinite escape timeを持つという．このような性質は線形システムには見られない．

一般に，$f(t, x)$がdomain $D$でlocally Lipschitzであり，$\dot{\boldsymbol{x}} = f(t, \boldsymbol{x})$の解がfinite escape time $t_e$を持つならば，この解$\boldsymbol{x}(t)$は$t \to t_e$としたときの$D$の全てのcompact (有界閉) な部分集合から離れなければならない．

#### Globally Existance and Uniqueness
$f(t, \boldsymbol{x})$が$\boldsymbol{x}$についてglobally Lipschitzである
$\overset{\text{def}}{\Longleftrightarrow}$ $\forall \boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$, $\exists L$ s.t.
$$
\| f(t, \boldsymbol{x}) - f(t, \boldsymbol{y}) \| \leq L \| \boldsymbol{x} - \boldsymbol{y} \|.
$$

ここで，次の命題が成り立つ．
$\forall \boldsymbol{x} \in \mathbb{R}^n$で偏微分$\partial f_i / \partial x_j$が連続な$f(t, \boldsymbol{x})$が$\boldsymbol{x}$についてglobally Lipschitzであるための必要十分条件は，偏微分$\partial f_i / \partial x_j$が$t$について一様にglobally boundedであることである．

例えば，$f(x) = -x^2$は任意の$x$でlocally Lipschitzだが，globally Lipschitzではない．なぜなら，$f^{\prime}(x) = -2x$がglobally boundedではないからである．

##### 解の存在性と一意性に関する補題
$f(t, \boldsymbol{x})$が$\forall t \in [t_0, t_1]$においてpiecewise continuousで，$\boldsymbol{x}$についてglobally Lipschitzであるとする．このとき，状態方程式$\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}), \; \boldsymbol{x}(t_0) = \boldsymbol{x}_0$の$[t_0, t_1]$における解が一意に存在する．

(Remark) 
線形システム$\dot{\boldsymbol{x}} = A(t)\boldsymbol{x} + g(t)$についてはglobal Lipschitzの条件を常に満たす．一般の非線形システムについては制約的な条件になっている．

さらに，次の補題が成り立つ．
$\forall t \geq t_0$, $\forall \boldsymbol{x} \in D \subset \mathbb{R}^n$について，$f(t, \boldsymbol{x})$が$t$においてpiecewise continuousであり，$\boldsymbol{x}$についてlocally Lipschitzであるとする．$W \subset D$をコンパクトとしたとき，
$$
\dot{\boldsymbol{x}} = f(t, \boldsymbol{x}), \; \boldsymbol{x}(t_0) = \boldsymbol{x}_0, \; \boldsymbol{x} \in W
$$
の全ての解を考える．このとき，$\forall t \geq t_0$において定義される解が一意に存在する．

例:
$$
\dot{x} = -x^3 = f(x)
$$
$f(x)$は$\mathbb{R}$においてlocally Lipschitzであるが，$f^{\prime}(x) = -3x^2$はglobally boundedではないため，globally Lipschitzではない．

任意の時刻$t$において，$x(t)$が正ならば$\dot{x}(t)$は負，$x(t)$が負ならば$\dot{x}(t)$は正である．よって，任意の初期条件$x(0) = a$に対して，解はコンパクト集合$\{ x \in \mathbb{R} \, | \, |x| \leq |a| \}$から出ない．つまり，この方程式は任意の$t \geq 0$で解を一意に持つ．

#### 平衡点
状態空間上の点$\boldsymbol{x} = \boldsymbol{x}^{*}$が$\dot{\boldsymbol{x}} = f(t, \boldsymbol{x})$の平衡点である
$\overset{\text{def}}{\Longleftrightarrow}$ $\boldsymbol{x}(t_0) = \boldsymbol{x}^{*} \Rightarrow \boldsymbol{x}(t) \equiv \boldsymbol{x}^{*}, \; \forall t \geq t_0$

Autonomous system $\dot{\boldsymbol{x}} = f(\boldsymbol{x})$においては，平衡点は方程式
$$
f(\boldsymbol{x}) = \boldsymbol{0}
$$
の実数解である．

平衡点は孤立している (近傍に他の平衡点が1つもない) ことも，continuumであることもある．
例: $\dot{x} = -x(1 - x)$の平衡点は，$x = 0, 1$

例:
$$
\left[
    \begin{array}{c}
    \dot{x}_1 \\
    \dot{x}_2
    \end{array}
    \right] = \left[
        \begin{array}{cc}
        0 & 1 \\
        0 & 0
        \end{array}
        \right] \left[
            \begin{array}{c}
            x_1 \\
            x_2
            \end{array}
            \right]
$$
の平衡点は，
$$
\left[
        \begin{array}{cc}
        0 & 1 \\
        0 & 0
        \end{array}
        \right] \left[
            \begin{array}{c}
            x_1 \\
            x_2
            \end{array}
            \right] = \left[
            \begin{array}{c}
            0 \\
            0
            \end{array}
            \right]
$$
つまり，$x_2 = 0$, $x_1$は任意．これはcontinuumである．

線形システム$\dot{\boldsymbol{x}} = A\boldsymbol{x}$については，
- $A$の固有値が$0$ $\Rightarrow$ continuum (上の例)
- $A$の固有値が非$0$ ($A^{-1}$が存在) $\Rightarrow$ isolated equilibrium point at $\boldsymbol{x} = \boldsymbol{0}$

また，複数のisolated equilibrium pointsは存在しない．$\boldsymbol{x}_a, \boldsymbol{x}_b$が平衡点であるとすると，2点を通る直線上の任意の点$\boldsymbol{x} = \alpha \boldsymbol{x}_a + (1 - \alpha)\boldsymbol{x}_b$は平衡点になる．

非線形システムにおいては複数の(無限の)isolated equilibrium pointsが存在することもある．
例: $\dot{x}_1 = x_2, \; \dot{x}_2 = -a \sin x_1 - bx_2$
平衡点は，$x_2 = 0, \; -a \sin x_1 - b x_2 = 0$．よって，$(x_1 = n \pi, x_2 = 0)$ for $n = 0, \pm 1, \pm 2, \ldots$なので，可算無限個の平衡点が存在する．

#### 線形化
非線形システムをある点のまわりで線形化して解析することもある．
$\dot{\boldsymbol{x}} = f(\boldsymbol{x})$を$\boldsymbol{x} = \boldsymbol{x}^{*}$のまわりでTaylor展開すると，
$$
\dot{\boldsymbol{x}} = f(\boldsymbol{x}) \simeq f(\boldsymbol{x}^{*}) + \nabla f(\boldsymbol{x}^{*})(\boldsymbol{x} - \boldsymbol{x}^{*}) + \mathrm{H.O.T.}
$$
($\mathrm{H.O.T.}$はhigh order term)

$\boldsymbol{x}^{*}$が平衡点のとき，$f(\boldsymbol{x}^{*}) = \boldsymbol{0}$．このとき，
$$
\dot{\boldsymbol{x}} \simeq \nabla f(\boldsymbol{x}^{*})(\boldsymbol{x} - \boldsymbol{x}^{*}) + \mathrm{H.O.T.}
$$
である．$\bar{\boldsymbol{x}} \coloneqq \boldsymbol{x} - \boldsymbol{x}^{*}$とすると，
$$
\dot{\bar{\boldsymbol{x}}} = \dot{\boldsymbol{x}} \simeq \nabla f(\boldsymbol{x}^{*})(\boldsymbol{x} - \boldsymbol{x}^{*}) + \mathrm{H.O.T.} \simeq \nabla f(\boldsymbol{x}^{*})\bar{\boldsymbol{x}}
$$
より，この$\dot{\bar{\boldsymbol{x}}} = \nabla f(\boldsymbol{x}^{*})\bar{\boldsymbol{x}}$が非線形システム$\dot{\boldsymbol{x}} = f(\boldsymbol{x})$の平衡点$\boldsymbol{x}^{*}$における線形化である．

##### 線形化の限界
- 線形化はある点の近傍の近似．
    - その点の近傍のlocalなふるまいしか予測できない (globalなふるまいを予測できない)．
- 非線形のときにのみ見られる重要な現象を見ることができない．
    - Finite escape time
    - 複数のisolated equilibrium points
    - Limit cycles
    - Subharmonic, harmonic, almost-periodic oscillations
    - Chaos
    - Multiple modes of behavior

### 非線形システムの例
#### 振り子
$$
mg \ddot{\theta} = -mg \sin \theta - kl\dot{\theta}
$$
$x_1 = \theta, x_2 = \dot{\theta}$とすると，この運動方程式は，
$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = - \frac{g}{l} \sin x_1 - \frac{k}{m} x_2
\end{cases}
$$
である．
平衡点は，$(x_1, x_2) = (n \pi, 0)$ for $n = 0, \pm 1, \pm 2, \ldots$である．
この平衡点は数学的には無限に存在するが，物理的には2点$(0, 0), (\pi, 0)$である．

$-kx_2/m$は摩擦の項とみなすことができる．これを取り除いた
$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = - \frac{g}{l} \sin x_1
\end{cases}
$$
はPendulum without frictionの方程式である．

さらに，振り子の中心にトルクをつけたと考えると，方程式は
$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = - \frac{g}{l} \sin x_1 - \frac{k}{m} x_2 + \frac{1}{ml^2}T
\end{cases}
$$
となる．

#### Tunnel-Diode Circuit
電源電圧を$E$，抵抗を$R$とする．コンデンサに流れる電流，加わる電圧を$i_C, v_C$，コイルに流れる電流，加わる電圧を$i_L, v_L$，ダイオードに流れる電流，加わる電圧を$i_R, v_R$とする．また，ダイオードの特性が$i = h(v)$で表されるとする．
このとき，
$$
i_C = C \frac{d v_C}{dt}, \; \; v_L = L \frac{d i_L}{dt}
$$
を満たす．$x_1 = v_C, x_2 = i_L, u = E$とする．キルヒホッフの法則より，
$
i_C + i_R - i_L = 0 \Rightarrow i_C = -h(x_1) + x_2
$
$
v_C - E + Ri_L + v_L = 0 \Rightarrow v_L = -x_1 - Rx_2 + u
$
である．よって，システムの方程式は
$$
\begin{cases}
\dot{x}_1 = \frac{1}{C}[-h(x_1) + x_2] \\
\dot{x}_2 = \frac{1}{L}[-x_1 - Rx_2 + u]
\end{cases}
$$
平衡点は，
$$
h(x_1) = \frac{E}{R} - \frac{1}{R}x_1
$$
である．平衡点の個数は電圧$E$と抵抗$R$によって変わる．

#### Mass-Spring System
Massの質量を$m$，変位を$y$，物体に加える力を$F$，物体にかかる摩擦力を$F_f$，ばねに加わる力を$F_{sp}$とする．運動方程式より，
$$
m \ddot{y} + F_f + F_{sp} = F
$$
非線形の要素として，
- 非線形ばね
    - $F_{sp} = g(y)$
- 静止 (もしくはクーロン力による) 摩擦力
    - $F_{sp} = g(y)$
ただし，
$
g(y) = k(1 - a^2 y^2)y, \; |ay| < 1 \; \; (\text{softening spring})
$
$
g(y) = k(1 + a^2 y^2)y \; \; (\text{hardening spring})
$

- $F_f$: static / Coulomb / viscous frictionによる要素

Massが静止しているとき，
- 静止摩擦力$F_s$が表面に平行に作用．
- $F_s \in [-\mu_s mg, \mu_s mg]$のとき，Massは静止を続ける

動き始めると，$F_f$は速度$v = \dot{y}$の関数として表される．

#### Negative-Resistance Oscillator
発振器の特性関数$i = h(v)$が以下の性質を持つ．
- $h(0) = 0, h^{\prime}(0) < 0$
- $v \to \infty$で$h(v) \to \infty$，$v \to -\infty$で$h(v) \to -\infty$．

キルヒホッフの法則より，
$$
i_C + i_L + i = 0
$$
なので，
$$
C \frac{dv}{dt} + \frac{1}{L} \int_{-\infty}^t v(s) ds + h(v) = 0
$$
これを$t$について微分し，$L$を掛けると，
$$
CL \frac{d^2v}{dt^2} + v + L h^{\prime}(v) \frac{dv}{dt} = 0
$$
$\tau \coloneqq t / \sqrt{CL}$とすると，
$$
\frac{dv}{d\tau} = \sqrt{CL} \frac{dv}{dt}, \; \; \frac{d^2 v}{d \tau^2} = CL \frac{d^2 v}{dt^2}
$$
よって，
$$
\ddot{v} + \varepsilon h^{\prime}(v) \dot{v} + v = 0, \; \varepsilon = \sqrt{\frac{L}{C}}
$$
である．特に，$h(v) = -v + v^3/3$のとき，
$$
\ddot{v} - \varepsilon(1 - v^2)\dot{v} + v = 0
$$
これをVan der Pol方程式という．

$x_1 = v, x_2 = \dot{v}$とする．
$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = -x_1 - \varepsilon h^{\prime}(x_1)x_2
\end{cases}
$$
という状態空間表現を得る．