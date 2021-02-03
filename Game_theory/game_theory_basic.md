---
marp: true
theme:
footer: "Kosuke Toda (@SeeKT)"
---
<!-- paginate: true -->
# ゲーム理論 (基本)
- 参考文献
    - J. W. Weibull, Evolutionary game theory, MIT press, 1997.
---
## ゲーム理論の考え方と応用
- ゲーム理論は合理的な意思決定主体の間の戦略的な相互作用を記述する数学的モデル
    - 応用範囲が広い (e.g. 経済，コンピュータサイエンス)

- 非協力ゲーム理論では，各主体は自分の利益のみを考え，利己的に行動するときについて解析する

- 非協力ゲームにおける古典的な仮定
    - 完全情報
        - エージェントがゲームの情報を完全に分かっている
    - 完全合理性
        - 自身の保有する情報全てを用いて，自身の利益の最大化を実現する

---
## 戦略と利得関数 (Notation)
分析を標準型の有限ゲームに限定する．次のNotationを用いる．
- $n \in \mathbb{N}$: エージェントの総数，$n \geq 2$．
- $\mathcal{N} = \{1, 2, \ldots, n\}$: エージェントの集合
- $S_i, \; \forall i \in I$: エージェント$i$の純粋戦略集合．$S_i = \{1, 2, \ldots, m_i\}$とする．
- $s = (s_1, \ldots, s_n), \; s_i \in S_i \; \forall i \in I$: 純粋戦略プロファイル
- $s_{-i} = (s_1, \ldots, s_{i - 1}, s_{i + 1}, \ldots, s_n)$: $i$以外の戦略プロファイル
- $S = \times_{i \in I} S_i$: 純粋戦略空間
- $U_i: S \to \mathbb{R}$: エージェント$i$の (純粋戦略) 利得関数
- $U: S \to \mathbb{R}^n$: ゲームの結合純粋戦略利得関数
    - $U(s) = (U_1(s), \ldots, U_n(s))$

---
## ゲームの定義
純粋戦略の観点から，標準型ゲームはタプル$G = (\mathcal{N}, S, U)$で定義される．ただし，$\mathcal{N}, S, U$は3ページで定義したものである．

特に，$\mathcal{N} = \{1, 2\}$のときは，利得関数$U_1, U_2$を$m_1 \times m_2$行列として表現できる．
- $A = (a_{hk})_{h \in S_1, k \in S_2}$: エージェント$1$の利得行列．$a_{hk} = U_1(h, k)$．
- $B = (b_{hk})_{h \in S_1, k \in S_2}$: エージェント$2$の利得行列．$b_{hk} = U_2(h, k)$．

例: 囚人のジレンマ
$$
A = \left(\begin{array}{cc}
    4 & 0 \\
    5 & 3
\end{array}\right), \; \; 
B = \left(\begin{array}{cc}
    4 & 5 \\
    0 & 3
\end{array}\right)
$$

---
## 混合戦略
エージェント$i$の混合戦略 = 純粋戦略集合$S_i$上の確率分布

- $x_i = (x_i^1, \ldots, x_i^{m_i})^{\mathrm{T}} \in \mathbb{R}^{m_i}$: エージェント$i$の混合戦略．エージェント$i$が純粋戦略$j \in S_i$をとる確率を並べたベクトル．
- 純粋戦略$j$を混合戦略として考えると，単位ベクトル$e_{m_i}^j$である．

$$
\Delta_i = \left\{x_i  \, \left| \,  \forall j \in S_i, \; x_i^j \geq 0, \;  \sum_{k \in S_i} x_i^k = 1  \right. \right\}
$$
はエージェント$i$の混合戦略集合．
- $\Delta_i$は単位ベクトル$e_{m_i}^j$を頂点とする$m_i - 1$次元単位単体．

---

![bg 40%](./fig/graph.svg)

$
S_i = \{1, 2\}
$
$
\Delta_i = \{(x_i^1, 1 - x_i^1)^{\mathrm{T}}\}
$
$
(0 \leq x_i^1 \leq 1)
$

---
## 混合戦略
すべての混合戦略$x_i \in \Delta_i$は，すべての単位ベクトル (純粋戦略) $e_{m_i}^j$ の凸結合として表される:
$$
x_i = \sum_{j \in S_i} x_i^j e_{m_i}^j
$$

- $C(x_i) = \{j \in S_i \, | \, x_i^j > 0\}$: $x_i \in \Delta_i$のキャリア (台)．
    - 実際に使われる可能性のある戦略の集合．

- $\mathrm{int}(\Delta_i) = \{x_i \in \Delta_i \, | \, \forall j \in S_i, x_i^j > 0\}$: $\Delta_i$の内部．

- $\mathrm{bd}(\Delta_i) = \{x_i \in \Delta_i \, | \, x_i \notin \mathrm{int}(\Delta_i)\}$: $\Delta_i$の境界．

---
## 混合戦略空間
- $\Theta = \times_{i \in \mathcal{N}} \Delta_i$: 混合戦略空間
- $x = (x_1, \ldots, x_n) \in \Theta$: 混合戦略プロファイル

非協力ゲームの標準的なアプローチでは，すべてのエージェントのランダムな行動は統計的に独立であるとする．
$\Rightarrow$ 混合戦略プロファイル$x \in \Theta$がプレーされるとき，純粋戦略プロファイル$s = (s_1, \ldots, s_n) \in S$が起こる確率は，
$$
x(s) = \prod_{i \in \mathcal{N}} x_i^{s_i}
$$
である．

---
## 期待利得
期待利得 = エージェント$i$の利得の期待値

期待利得関数$u_i: \Theta \to \mathbb{R}$
$$
u_i(x) := x(s)U_i(s) = \sum_{j \in S_i} x_i^j u_i(e_{m_i}^j, x_{-i})
$$
- 各エージェントが個別にとる混合戦略の線形関数
- エージェント$i$の純粋戦略に対して，他の全てのエージェントの混合戦略を固定して，エージェント$i$が得る利得に関して重み付けした合計．

- $u: \Theta \to \mathbb{R}^n$: ゲームの結合混合戦略利得関数
    - $u(x) = (u_1(x), \ldots, u_n(x))$

---
## ゲームの混合拡大
混合戦略の観点から，$G = (\mathcal{N}, \Theta, u)$としてゲームを定義する．$\Theta, u$はそれぞれ8, 9ページで定義．

特に2人ゲームのときは，任意の混合戦略$x_1 \in \Delta_1,  x_2 \in \Delta_2$に対して，
$$
u_1(x) = \sum_{j \in S_1} x_1^j u_1(e_{m_1}^2, x_2) = \sum_{j \in S_1} \sum_{k \in S_2} x_1^j x_2^k a_{jk} = (x_1, Ax_2) \\
u_2(x) = \sum_{j \in S_1} \sum_{k \in S_2} x_1^j x_2^k b_{jk} = (x_1, Bx_2) = (x_2, B^{\mathrm{T}}x_1)
$$
と書ける．ただし，$A, B$はそれぞれエージェント1, 2の利得行列．

---
## 支配関係