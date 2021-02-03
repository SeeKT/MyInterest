---
marp: true
theme:
footer: "Kosuke Toda (@SeeKT)"
---
<!-- paginate: true -->
# ゲーム理論とサイバーセキュリティ

- 参考文献
    [1] A. Iqbal, et al. "Game theoretical modelling of network/cybersecurity," IEEE Access, vol. 7, pp. 154167-154179, 2019.
    [2] A. Sokri, "Optimal resource allocation in cyber-security: A game theoretic approach," Procedia computer science, vol. 134, pp. 283-288, 2018.
---
## 背景 (ゲーム理論の応用)
- ゲーム理論は合理的な意思決定主体の間の戦略的な相互作用を記述する数学的モデル
    - 応用範囲が広い (e.g. 経済，コンピュータサイエンス)

- セキュリティの文脈においては，攻撃をする主体と対策を行う主体は戦略的に振る舞う合理的なエージェントであるとみなせる
    - 攻撃者と対策を行う者の間のゲームを考えることができる

---
## 背景 (サイバーセキュリティ)
- サイバー空間を保護する上でのサイバーセキュリティの重要性
    - サイバー空間がより複雑になっていること
    - 攻撃が複雑かつ高度になっていること

- サイバーセキュリティの目的
    - 有効でスケーラブルなセキュリティメカニズムの提供
    - CPSの信頼性の向上
        - サイバー攻撃が物理的なシステムや人命に影響を及ぼしうる

---
## 背景 (ゲーム理論とセキュリティ)
- 物理的なセキュリティにゲーム理論が応用されていたこと

- ゲーム理論とサイバーセキュリティは相性が良い
    - 攻撃の種類によって攻撃者が得る利得が異なる
    - サイバーセキュリティの成功は実装された対策だけではなく攻撃にも依存する

- 応用例
    - 侵入検知，プライバシーの保護，ジャミング，盗聴

---
## 応用例: セキュリティのリソース配分 [2]
- サイバー空間におけるリソース配分問題をゲーム理論的に定式化する．

- Cyberinfrastructure system における，attacker $a$ と defender $d$ の間のセキュリティゲームを考える．
    - 全てが高速ネットワーク (インターネット) でリンクされた，高度なコンピューティングシステム，データ，ソフトウェア，人間から構成されている動的なエコシステム

---
## Notation
- $T = \{t_1, t_2, \ldots, t_n\}$: 攻撃を受けるリスクがあるターゲットの集合
    - e.g. インターネットに接続された制御システム
- $S = \{s_1, s_2, \ldots, s_m\}$: ターゲットをカバーするリソースの集合
    - e.g. ファイアウォール
- $\langle a_t \rangle \in \mathbb{R}^n$: attackerの混合戦略．$a_t$: ターゲット$t$を攻撃する確率
- $\langle p_t \rangle \in \mathbb{R}^n$: defenderの混合戦略．$p_t$: ターゲット$t$を保護する周辺確率
- $\langle a, p \rangle$: 戦略プロファイル．attackerとdefenderがとる戦略の組．
- $r_d(t)$: ターゲット$t$がカバーされたときのdefenderの報酬
- $c_d(t)$: ターゲット$t$がカバーされなかったときのdefenderのコスト
- $r_a(t)$: ターゲット$t$がカバーされなかったときのattackerの報酬
- $c_a(t)$: ターゲット$t$がカバーされたときのattackerのコスト

---
## 利得
戦略プロファイル$\langle a, p \rangle$がプレーされたときの2人のプレイヤーの期待利得
$$
U_d(a, p) = \sum_{t \in T} a_t (p_t r_d(t) - (1 - p_t)c_d(t)) \; \; (1)
$$
$$
U_a(a, p) = \sum_{t \in T} a_t ((1 - p_t) r_a(t) - p_tc_a(t)) \; \; (2)
$$

- 攻撃されたターゲットとその利得にのみ依存する．
- 攻撃されていないターゲットには依存しない．
---
## ゲームについて
- Nash equilibrium
    - プレイヤー (defender, attacker) が同時に動いたときのゲームの解
- Stackelberg equilibrium
    - ゲームがsequentialで，leader-follower, i.e., defenderが先に動いて戦略をとり，attackerがそれに反応する
    - Leader-follower interactionのstandard solution

$\Rightarrow$ ここではStackelberg gameの文脈．

- Stackelberg game
    - Leaderが自分自身のpayoffとfollowerのpayoffを知っているという仮定
        - 現実のサイバーセキュリティでは利得は正確には分からない

---
## [2]のアプローチ
Stochastic simulationを用いて利得をrandomizeする

- 固定値を値の範囲に変更することで，報酬とコストに不確実性を与える

- 報酬とコストの変動の評価のためにのアプローチの使用
    - Optimistic, most likely, pessimistic
---
## Followerの最適化問題
Leaderのポリシー$p$が与えられた下で，followerの最適化問題は以下．
$$
\mathrm{Max}_a \;  \sum_{t \in T} a_t ((1 - p_t) r_a(t) - p_tc_a(t)) \; \; (3) \\
\mathrm{subject \; to} \; \sum_{t \in T} a_t = 1 \; \; (4) \\
a_t \geq 0, \; \; \forall t \; \; (5)
$$

この問題は，$p$が与えられた下でのfollowerの期待利得に関する線形計画問題．
$$
U_a(t, p) := (1 - p_t) r_a(t) - p_tc_a(t), \; \; \forall t \in T \; \; (6)
$$
が最大となるような$t$で$a_t = 1$とするものが最適解である．

---
## ベクトル表現
問題(3) - (5)をベクトルを用いて記述し直す．
$$
U_a := (U_a(t_1, p), \ldots, U_a(t_n, p))^{\mathrm{T}} \in \mathbb{R}^n \\
\boldsymbol{1}_n := (1, \ldots, 1)^{\mathrm{T}} \in \mathbb{R}^n
$$
とする．このとき，問題(3) - (5)は，
$$
\mathrm{Max}_a \;  U_a^{\mathrm{T}} \langle a_t \rangle \\
\mathrm{subject \; to} \; \boldsymbol{1}_n^{\mathrm{T}} \langle a_t \rangle = 1 \\
\langle a_t \rangle \geq \boldsymbol{0}
$$
と書き直される．

---
## 双対問題
この線形計画問題の双対問題は，
$$
\mathrm{Min} \; u \\
\mathrm{subject \; to} \; \boldsymbol{1}_n u \geq U_a
$$
である．これを書き下すと，
$$
\mathrm{Min} \; u \; \; (7) \\
u \geq U_a(t, p), \; \; \forall t \in T \; \; (8)
$$
となる．相補スラック条件は，
$$
a_t (u - U_a(t, p)) = 0, \; \; \forall t \in T \; \; (9)
$$
である．

---
## Leaderの最適化問題について
Followerの最適性条件を入れた下での最適化問題は混合整数二次計画問題となる．

$\mathrm{Max}_p \; \sum_{t \in T} a_t (p_t r_d(t) - (1 - p_t)c_d(t)) \; \; (10)$
$\sum_{t \in T} p_t \leq m \; \; (11)$
$\sum_{t \in T} a_t = 1 \; \; (12)$
$0 \leq u - U_a(t, p) \leq (1 - a_t)M, \; \; \forall t \in T \; \; (13)$
$p_t \in [0, 1], \; \; \forall t \in T \; \; (14)$
$a_t \geq 0, \; \; \forall t \; \; (15)$
$u \in \mathbb{R} \; \; (16)$

(13)式における$M$は巨大数．これは，$a_t > 0$となるすべての純粋戦略に対してfollowerのpayoff $u$が最適であるということを示している．

