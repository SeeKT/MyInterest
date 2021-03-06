# 確率システムの制御
- Kosuke Toda (@SeeKT)
### 参考
- 大住，確率システム入門，朝倉書店，2002．

### はじめに
#### 確率システムの理論研究
- 1940年代，Wiener
    - 不規則雑音に埋もれた信号を抽出するフィルタリング問題の考察
    - ウィーナ・ホップ積分方程式を解く問題に帰着
- 1960年 - 61年
    - カルマンフィルタ
        - 信号が動的システムによって生成されるという仮定の下，システムの状態を推定
        - Kalman, Bucy
##### ウィーナフィルタとカルマンフィルタ
- ウィーナフィルタ
    - 信号過程の単一性，定常性，無限時間観測という仮定
- カルマンフィルタ
    - 信号の多次元，非定常性，有限時間観測という設定
    - 推定値を与えるベクトル微分 (差分) 方程式と，それを計算するために必要なリッカチ微分 (差分) 方程式の2本を解くことで実現
##### 確率システムの制御
- 不規則雑音に乱された観測データに基づいて，確率システムをある目的を達成するように制御するにはどのようにすればよいか？という問題
    - 確率微分方程式によるモデル化，それに基づいた非線形フィルタの設計，確率的最適制御，確率的安定性の研究
        - Kushner, Wonham
- 背景あるのは
    - 確率過程論
    - システム理論

### 確定システムの制御
確率システムの制御については，外乱が介入しないダイナミカルシステムに対して確立されている制御理論がもとになっている．オブザーバのような，確率システムについて展開された理論が確定システムに影響を与えたものもある．

システムおよび観測過程が，次の微分方程式によって記述されているとする．
$$
\dot{x}(t) = A x(t) + C u(t), \; t_0 \leq t \leq T \\
y(t) = H x(t)
$$

- $x(t)$: 状態ベクトル, $x \in \mathbb{R}^n$
- $y(t)$: 観測ベクトル, $y \in \mathbb{R}^m \; (m \leq n)$
- $u(t)$: 制御量ベクトル, $x \in \mathbb{R}^{\ell}$

行列$H$が単位行列$H = I$であるならば，システムの状態は独立に観測できる．よって，最適制御は評価コスト汎関数を2次形式
$$
J(u) = x^T(T) F x(T) + \int_{t_0}^T [x^T(s) M x(s) + u^T(s) N u(s)]ds
$$
- $F, M \geq 0$: 対称
- $N > 0$: 対称

と設定すると，
$$
u^o(t) = -N^{-1}C^T \Pi(t)x(t)
$$
で与えられる．ここで，$\Pi(t)$は$n \times n$次元リッカチ微分方程式
$$
\dot{\Pi}(t) + \Pi(t) A + A^T \Pi(t) + M - \Pi(t) C N^{-1} C^T \Pi(t) = 0 \\
\Pi(T) = F
$$
の正定値解である．

観測過程の次元が$m < n$ ($H \neq I$) ならばシステム状態量のすべてが独立に観測されるわけではない．この場合は状態推定器としてのオブザーバが必要．
システム状態量と同一次元をもつオブザーバは，
$$
\dot{\hat{x}} = A \hat{x}(t) + Cu(t) + K(y(t) - H \hat{x}(t))
$$
によって，最小次元オブザーバは
$$
\hat{x}(t) = Dz(t) + Ey(t) \\
\dot{z}(t) = \hat{A}z(t) + \hat{C}u(t) + Ky(t)
$$
によってシステム状態量$x(t)$の推定値$\hat{x}(t)$が生成される．

オブザーバゲイン行列$K$は推定誤差
$$
e(t) = x(t) - \hat{x}(t)
$$
あるいは
$$
e(t) = x(t) - Mz(t)
$$
($M$は次元をそろえるための行列)が$e(t) \to 0 \; (t \to \infty)$となるように定められるが，決め方は一意的ではない．同一次元オブザーバに関しては，行列$A - KH$が安定であればよく，$n$個の固有値は設計者が任意に指定できる．
この式では，オブザーバは$y(t) - H\hat{x}(t)$という修正項を付与することで構成されている．これは歴史的に見てカルマンフィルタの構造をもとにして構成されているといえる．

オブザーバによって得られる推定値を用いると，最適制御値は
$$
u^o(t) = -N^{-1} C^T\Pi(t)\hat{x}(t)
$$
によって与えられる．制御ゲインマトリクスは同じであり，状態量を推定値で置換しただけであることがわかる．つまり，制御と推定を分離して設計できる．
(確率システムにおける分離定理が確定システムに対しても同様に成り立つこと)


$\Rightarrow$ システムに不規則雑音 (外乱) が介入する場合の状態推定や最適制御はどうなる？

### 確率システムの制御問題
確率システムの表現として，
$$
\dot{x}(t) = A x(t) + Cu(t) + G\gamma(t) \\
y(t) = Hx(t) + R \theta(t)
$$
のように，不規則外乱$\gamma(t), \theta(t)$を付与したモデルが通常よく用いられる．
##### 確率システムを考察する際の問題点
1. 何故不規則外乱項をもつモデル？
    - システムには環境外乱などの付加雑音が存在する
    - 確定システムにもモデルを構築した際の近似によるモデルの誤差が含まれる
    - 確定システムのモデル中のパラメータが正確でないことから，誤差分として考慮する必要がある
    - 制御器が誤差を持ち込む
    - 状態を観測する際には計測機器に雑音が介入したり，計測誤差が含まれる

$\Rightarrow$ 確定モデルに含まれる誤差の性質を正確に特定できずに，それを付加雑音として一括したものとみることができる．

確率システム理論においては，雑音を白色雑音過程と考えて取り扱う．

2. なぜ白色雑音が採用されるのか？
    - システムに介入する雑音は，実際にはどのような周波数成分も含んでいるのかわからない場合が多い．いかなる周波数成分をもつ雑音にも対処できるようにするには，白色雑音が最適．
    - 白色雑音は同時刻における相関 (分散) が無限大で与えられる．システムの設計にあたっては雑音の最悪のケースを考えているので，制御系としてはある種のロバスト性を考慮している．
    - どのような周波数成分をもつ雑音過程も，適当なフィルタを導入することで白色雑音から生成できる．

しかし，非現実的な白色雑音をそのまま設定するわけにはいかないので，それに代わってブラウン運動過程 (ウィーナ過程) と呼ばれる確率過程を導入する．これは，$\gamma(t) = d w(t) / dt$によって得られる過程であり，これの導入により上記の方程式は
$$
dx(t) = Ax(t) dt + Cu(t) dt + Gdw(t)
$$
という差分方程式に変換できる．これを確率微分方程式という．
確率システム理論はこの確率微分方程式をシステムのダイナミクスとして展開される．

3. なぜ確率微分方程式でモデル化するのか？
    - 確率微分方程式の解過程は直前の時刻の状態のみに依存して決定される (マルコフ過程) となり，確率過程論として数学的に厳密に裏付けされた理論展開が可能．
    - 実際の制御系の出力は定常ではなく本質的に非定常過程として取り扱わなければならない．
    - 理論展開が時間領域のみ．得られる結果も時間領域内で処理できる．差分系のダイナミクスで与えられることはコンピュータにとって好都合．