# 回帰分析に必要な線形代数の知識
- Kosuke Toda (@SeeKT)
### 参考
- 佐和，回帰分析 新装版，朝倉書店，2020．
- 狩野，2021年度多変量解析 講義資料 (大阪大学 大学院基礎工学研究科)

## Notation
- $\mathbb{R}^{p \times q}$: $p \times q$の実行列全体
- $\mathbb{R}^p$: $p$次元実縦ベクトル全体
- $GL_p(\mathbb{R})$: $p$次の正則行列全体
- $\mathcal{O}(p)$: $p$次直交行列全体
- $\mathcal{O}(p \times q)$: $p \times q$列直交行列全体
    - $V \in \mathbb{R}^{p\times q}, V^{\mathrm{T}}V = I_q$
## 線形代数の基本
### 行列の階数
#### 階数の性質
1. - $A \in \mathbb{R}^{p \times q}, B \in \mathbb{R}^{q \times r} \Longrightarrow \mathrm{rank}(AB) \leq \min\{\mathrm{rank}(A), \mathrm{rank}(B) \} $
    - $A, B \in \mathbb{R}^{p \times q} \Longrightarrow \mathrm{rank}(A + B) \leq \mathrm{rank}(A) + \mathrm{rank}(B) $
    - $ A \in \mathbb{R}^{p \times q}, B \in GL_q(\mathbb{R}) \Longrightarrow \mathrm{rank}(AB) = \mathrm{rank}(A) $
    - $ A \in \mathbb{R}^{p \times q} \Longrightarrow \mathrm{rank}(A) = \mathrm{rank}(A^{\mathrm{T}}) = \mathrm{rank}(A^{\mathrm{T}}A) = \mathrm{rank}(AA^{\mathrm{T}}) $
2. $A \in \mathbb{R}^{p \times p}$, 対称．$A$の固有値は実数で，異なる固有値に対応する固有ベクトルは互いに直交する．

### スペクトル分解，階数分解，特異値分解
#### スペクトル分解
$A \in \mathbb{R}^{p \times p}$, 対称．$\exists V = [\boldsymbol{v}_1, \ldots, \boldsymbol{v}_p] \in \mathcal{O}(p)$, $\exists \Lambda = \mathrm{diag}\{\lambda_1, \ldots, \lambda_p\}$ s.t. 
$$
A = V \Lambda V^{\mathrm{T}} = \sum_{k = 1}^p \lambda_k \boldsymbol{v}_k \boldsymbol{v}_k^{\mathrm{T}}
$$
を$A$のスペクトル分解という．また，$V^{\mathrm{T}}AV = \Lambda$を$A$の対角化という．
#### 階数分解と特異値分解
$A \in \mathbb{R}^{p \times q}, \mathrm{rank}(A) = r$
1. $\exists B \in \mathbb{R}^{p \times r}, \exists C \in \mathbb{R}^{q \times r}$ s.t. $A = B C^{\mathrm{T}}$.
2. $\exists U_1 \in \mathcal{O}(p \times r), \exists V_1 \in \mathcal{O}(q \times r), \exists \rho$: 対角成分がすべて正の$r$次対角行列 s.t. $A = U_1 \rho V_1^{\mathrm{T}}$.
3. 2より直ちに，$\exists U \in \mathcal{O}(p), \exists V \in \mathcal{O}(q)$ s.t. 
$$
A = U \left[ \begin{array}{cc}
    \rho & O \\
    O & O
\end{array}
\right] V^{\mathrm{T}}
$$

1を$A$の階数分解といい，2,3を$A$の特異値分解という．$\rho$の対角要素を$A$の特異値という．

### 行列のトレース
$A = (a_{ij}) \in \mathbb{R}^{p \times p}$. $\mathrm{tr}(A) = \sum_{k = 1}^p a_{kk}$を$A$のトレースという．
#### トレースの性質
$A, B$: 行列，$c, d$: スカラー．
1. $\mathrm{tr}(A^{\mathrm{T}}) = \mathrm{tr}(A), \; \mathrm{tr}(cA + dB) = c \mathrm{tr}(A) + d \mathrm{tr}(B)$.
2. $\mathrm{tr}(AB) = \mathrm{tr}(BA) = \sum_{i,j} a_{ij} b_{ji}$
3. $\mathrm{tr}(A) = \sum_k \lambda_k (A)$, ただし$\lambda_k(A)$は$A$の固有値．

### 正定値行列
$A, B \in \mathbb{R}^{p \times p}$, 対称．
1. $\boldsymbol{x}^{\mathrm{T}} A \boldsymbol{x} \geq 0$ for $\forall \boldsymbol{x} \in \mathbb{R}^p$のとき，$A$は非負定値行列といわれ，$A \geq 0$と書く．$\boldsymbol{x}^{\mathrm{T}} A \boldsymbol{x} > 0$ for $\forall \boldsymbol{x} \in \mathbb{R}^p, \boldsymbol{x} \neq \boldsymbol{0}$のとき，$A$は正定値行列といわれ，$A > 0$と書く．
2. $A - B \geq 0$のとき，$A \geq B$，また，$A - B > 0$のとき，$A > B$と表す．

#### 正定値行列の性質
$A \in \mathbb{R}^{p \times p}$, 対称．
1. $A > 0 \Longleftrightarrow \lambda_k(A) > 0 \; (k = 1, \ldots, p)$
    $A > 0 \Longleftrightarrow A \geq 0$ and $\mathrm{det}(A) \neq 0$
2. $A \geq 0 \Longleftrightarrow \lambda_k(A) \geq 0 \; (k = 1, \ldots, p)$
    $A \geq 0 \Longleftrightarrow \exists B \in \mathbb{R}^{p \times r}, \; r = \mathrm{rank}(A)$ s.t. $A = BB^{\mathrm{T}}$
    $A \geq 0 \Longleftrightarrow \exists C \in \mathbb{R}^{p \times p}$, 対称, s.t. $A = C^2$

行列$C$で非負定値であるものを$A$の平方根といい，しばしば$A^{\frac{1}{2}}$で表す (その他の性質については適宜別資料参照)．

### 分割行列
正方行列$A$の分割を$A = \left[ 
    \begin{array}{cc}
    A_{11} & A_{12} \\
    A_{21} & A_{22}
    \end{array}
\right]$とする．ここで，$A_{11}$と$A_{22}$は正方行列である．
$A^{-1} = \left[ 
    \begin{array}{cc}
    A^{11} & A^{12} \\
    A^{21} & A^{22}
    \end{array}
\right]$と書く．ここで登場する逆行列はすべてその存在を仮定する．

#### 分割行列の性質
以下，必要な逆行列の存在を仮定する．$A_{11}, A_{22}$: 正方行列．
$$
A_{11.2} := A_{11} - A_{12}A_{22}^{-1}A_{21}, \; A_{22.1} = A_{22} - A_{21}A_{11}^{-1}A_{12}
$$
とおく．
1. 
$
\mathrm{det}\left[ 
    \begin{array}{cc}
    A_{11} & A_{12} \\
    A_{21} & A_{22}
    \end{array}
\right] = \mathrm{det}(A_{22})\mathrm{det}(A_{11.2}) = \mathrm{det}(A_{11}) \mathrm{det}(A_{22.1})
$
2. 
$
(A^{11})^{-1}A^{12} = -A_{12}A_{22}^{-1}, \; \; A^{21} (A^{11})^{-1} = - A_{22}^{-1} A_{21}
$
$
A^{12}(A^{22})^{-1} = -A_{11}^{-1}A_{12}, \; \; (A^{22})^{-1}A^{21} = -A_{21}A_{11}^{-1} 
$
3. 
$
\left[ 
    \begin{array}{cc}
    A_{11} & A_{12} \\
    A_{21} & A_{22}
    \end{array}
\right]^{-1} = \left[ 
    \begin{array}{cc}
    A_{11.2}^{-1} & -A_{11.2}^{-1}A_{12}A_{22}^{-1} \\
    A_{22}^{-1}A_{21}A_{11.2}^{-1} & A_{22}^{-1} + A_{22}^{-1}A_{21}A_{11.2}^{-1}A_{12}A_{22}^{-1}
    \end{array}
\right]
$

$
\hspace{26mm}= \left[ 
    \begin{array}{cc}
    O & O \\
    O & A_{22}^{-1}
    \end{array}
\right] + \left[ 
    \begin{array}{c}
    -I  \\
    A_{22}^{-1}A_{21}
    \end{array}
\right] A_{11.2}^{-1} \left[
    \begin{array}{cc}
        -I & A_{12}A_{22}^{-1}
    \end{array}
\right]
$

$
\hspace{26mm}= \left[ 
    \begin{array}{cc}
    A_{11}^{-1} + A_{11}^{-1}A_{12}A_{22.1}^{-1}A_{21}A_{11}^{-1} & -A_{11}^{-1}A_{12}A_{22.1}^{-1} \\
    A_{22.1}^{-1}A_{21}A_{11}^{-1} & A_{22.1}^{-1}
    \end{array}
\right]
$

$
\hspace{26mm}= \left[ 
    \begin{array}{cc}
    A_{11}^{-1} & O \\
    O & O
    \end{array}
\right] + \left[ 
    \begin{array}{c}
    A_{11}^{-1}A_{12}  \\
    -I
    \end{array}
\right] A_{22.1}^{-1} \left[
    \begin{array}{cc}
        A_{21}A_{11}^{-1} & -I
    \end{array}
\right]
$
4. $A> 0$のとき，$A_{11.2} > 0, A_{22.1} > 0, A^{11} \geq A_{11}^{-1}$.

#### Woodbury's identities
分割行列$\left[ 
    \begin{array}{cc}
    A_{11} & A_{12} \\
    A_{21} & A_{22}
    \end{array}
\right]$に対して次の公式が成り立つ．
$
(A_{11} + A_{12}A_{22}^{-1}A_{21})^{-1} = A_{11}^{-1} - A_{11}^{-1}A_{12}(A_{22} + A_{21}A_{11}^{-1}A_{12})^{-1}A_{21}A_{11}^{-1}
$
$
(A_{11} + A_{12}A_{22}^{-1}A_{21})^{-1}A_{12} = A_{11}^{-1}A_{12}(A_{22} + A_{21}A_{11}^{-1}A_{12})^{-1}A_{22}
$
$
A_{21}(A_{11} + A_{12}A_{22}^{-1}A_{21})^{-1} = A_{22}(A_{22} + A_{21}A_{11}^{-1}A_{12})^{-1}A_{21}A_{11}^{-1}
$
$
A_{21}(A_{11} + A_{12}A_{22}^{-1}A_{21})^{-1}A_{12} = A_{22}(A_{22} + A_{21}A_{11}^{-1}A_{12})^{-1}A_{22}
$
$
[A_{21}(A_{11} + A_{12}A_{22}^{-1}A_{21})^{-1}A_{12}]^{-1} = (A_{21}A_{11}^{-1}A_{12})^{-1} + A_{22}^{-1}
$

#### Khatri's Lemma
$[A, B]$を正則行列，$A^{\mathrm{T}}B = O, M > 0$とする．このとき，次式が成立する．
$$
MA(A^{\mathrm{T}}MA)^{-1}A^{\mathrm{T}}M + B(B^{\mathrm{T}}MB)^{-1}B^{\mathrm{T}} = M
$$

## ベクトルと行列の微分
行列$A$の要素$a_{ij}$が$x$の関数$a_{ij}(x)$になっているとき，$A$の$x$に関する微分を
$$
\frac{\partial A(x)}{\partial x} = \left(\frac{\partial a_{ij}(x)}{\partial x}\right)
$$
と定義する．また，行列$X$の関数$f(X)$の$X$に関する微分を
$$
\frac{\partial f(X)}{\partial X} = \left(\frac{\partial f}{\partial x_{ij}}\right)
$$
と定義する．ベクトル$\boldsymbol{x}$の関数$f(\boldsymbol{x})$の微分も，行列に関する微分の特例とみなせる．また，ベクトル$\boldsymbol{x}$の関数$f$の2階微分は，
$$
\frac{\partial^2 f(\boldsymbol{x})}{\partial \boldsymbol{x} \boldsymbol{x}^{\mathrm{T}}} = \left(\frac{\partial^2 f}{\partial x_i \partial x_j} \right)
$$
と定義される．

#### ベクトルと行列に関する微分の公式
1. 
$
\frac{\partial \boldsymbol{x}^{\mathrm{T}} \boldsymbol{a}}{\partial \boldsymbol{x}} = \boldsymbol{a}, \; \; \frac{\partial^2 \boldsymbol{x}^{\mathrm{T}} \boldsymbol{a}}{\partial \boldsymbol{x} \boldsymbol{x}^{\mathrm{T}}} = \boldsymbol{0}
$
2. 
$
\frac{\partial \boldsymbol{x}^{\mathrm{T}} A\boldsymbol{x}}{\partial \boldsymbol{x}} = 2A\boldsymbol{x}, \; \; \frac{\partial^2 \boldsymbol{x}^{\mathrm{T}} A\boldsymbol{x}}{\partial \boldsymbol{x} \boldsymbol{x}^{\mathrm{T}}} = 2A
$
3. 
$
\frac{\partial |X|}{\partial X} = (X_{ij}) = |X|(X^{-1})^{\mathrm{T}}
$
ただし，$X_{ij}$は行列$X$の要素$x_{ij}$の余因子である．
4. $X$が対称行列であり，$x_{ij} = x_{ji}$を同一の変数とみなすならば，
$$
\frac{\partial |X|}{\partial X} = 2(X_{ij}) - \mathrm{diag}[(X_{ij})]
$$
ただし，$\mathrm{diag}[(X_{ij})]$は対角要素を$X_{ii}$，非対角要素をゼロとする対角行列である．
5. 
$
\frac{\partial \log |X|}{\partial |X|} = \begin{cases}
(X^{-1})^{\mathrm{T}} & \mathrm{if} \; \; X \mathrm{ \; is \; not \; symmetry} \\
2X^{-1} - \mathrm{diag}(X^{-1}) & \mathrm{otherwise}
\end{cases}
$
6. 
$
\frac{\partial \mathrm{tr}(XA)}{\partial X} = \begin{cases}
A^{\mathrm{T}} & \mathrm{if} \; \; X \mathrm{ \; is \; not \; symmetry} \\
A + A^{\mathrm{T}} - \mathrm{diag}(\mathrm{A}) & \mathrm{otherwise}
\end{cases}
$
7. 
$
\frac{\partial x^{pq}}{\partial X} = -(x^{pi}x^{jq})
$
ただし，$x^{ij}$は$X^{-1}$の$(i, j)$要素である．