---
marp: true
theme:
footer: "Kosuke Toda (@SeeKT)"
---
<!-- paginate: true -->
# 進化的安定性
- 参考文献
    - J. W. Weibull, Evolutionary game theory, MIT press, 1997.
---
## Notation
対称2人ゲームに限定して分析を行う．

---
## 議論のための設定
- 設定
    - 全員が同一の既存戦略$x \in \Delta$をとるようにプログラムされている主体からなる大集団に，突然変異戦略$y \in \Delta$をとる突然変異体の集団がシェア$\varepsilon$で現れた
    - 2要素集団のなかから，主体のペアがゲームをプレーするたびに繰り返しランダムに (等確率で) 抜き出される
    $\Rightarrow$ ひとつの主体がゲームをプレーするために抜き出されてたとき，対戦相手が戦略$y$を確率$\varepsilon$でとり，戦略$x$を確率$1 - \varepsilon$でとる
    $\Rightarrow$ 自分の戦略を$z \in \{x, y\}$とすると，相手との対戦による (期待) 利得は$u(z, w)$である．ただし，$w = \varepsilon y + (1 - \varepsilon)x \in \Delta$ とする．

---
## 生物学的な直観と定義
進化が突然変異戦略を淘汰する = 突然変異戦略から得られる利得 (適合度) が既存戦略のそれよりも小さい，つまり，
$$
u(x, \varepsilon y + (1 - \varepsilon)x) > u(y, \varepsilon y + (1 - \varepsilon)x)
$$
が成り立つこと．

$x \in \Delta$ が進化的安定戦略 (ESS) であるとは，
$$
\forall y \neq x, \; \exists \bar{\varepsilon}_y \in (0, 1), \; \forall \varepsilon \in (0, \bar{\varepsilon}_y), \\
u(x, \varepsilon y + (1 - \varepsilon)x) > u(y, \varepsilon y + (1 - \varepsilon)x)
$$
が成り立つことをいう．

---
## 進化的安定戦略とナッシュ均衡
$\Delta^{ESS} \subset \Delta$を進化的安定戦略の集合とする．
$x \in \Delta^{ESS}$ならば$x \in \beta^{*}(x)$である．つまり，$\Delta^{ESS} \subset \Delta^{NE}$である．

また，$\Delta^{ESS}$について，
$$
\Delta^{ESS} = \{x \in \Delta^{NE} \, | \, u(y, y) < u(x, y), \; \forall y \in \beta^{*}(x), \; y \geq x \}
$$
が成り立つ．