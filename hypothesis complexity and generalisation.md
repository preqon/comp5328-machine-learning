# hypothesis complexity and generalisation

What kind of hypothesis class $H$ do we choose before searching for $h$?
$$ h \in H $$
We want the target function $c$ in universal function space to be in $H$.

$$c = \arg \min_h R(h) $$
**risk**:

empirical risk is the estimated loss

$$ R_S(h) = \frac {1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h) $$
expected risk is the expected loss
$$R(h) = \mathbb{E} [ R_S(h)]
= \mathbb{E}[\ell (X, Y, h)] $$
$$ = \int_{(X,Y)} \ell(X, Y, h) p(X,Y) \, dXdY $$
**notation**:

while target function $c$ (in universal function space) is given above, the optimal hypothesis function inside $H$ is denoted $h^*$ 
$$ h^* = \arg \min_{h \in H} R(h) $$

the hypothesis learned from data $S$ is denoted
$$h_S = \arg \min_{h \in H} R_S(h)$$
which we learn using estimated loss, as above.

**approximation error**:
- arises from difference between h* and c
$$R(h^*) - R(c)$$
- if target $c$ is in $H$, this will be zero

**estimation error**:
- arises from difference between $h^*$ and $h_S$
$$R(h_S) - R(h^*) $$
## PAC learning framework

'Probably approximately correct' learning framework.
Arises because if $H$ is too large and complex, the estimation error becomes large.
Explains how many training samples are needed to learn the best $h$ in $H$.

A hypothesis class $H$ is PAC-learnable if there is a learning algorithm $\mathcal{A}$ and a polynomial function $\text{poly}(\cdot , \cdot)$ s.t for any $\epsilon > 0$ and $\delta > 0$, for the whole distribution $D$ on $X \times Y$, the following probability holds for 
- any sample size $n > \text{poly} ( 1 / \delta , 1 / \epsilon)$ 
- $h_S$ learned by $\mathcal{A}$
$$p \biggl( R(h_S) - R(h^*) \le \epsilon  \biggr) \ge 1 - \delta $$
This shows that when sample size is large enough, the learned hypothesis $h_S$ is an approximation of the best $h^*$ in $H$. 

If $H$ is too complex, we need more training samples, e.g. size given by
$$ n> \exp (1/\delta , 1/\epsilon) $$
and so $H$ is not PAC-learnable in this case.

### how to check if $H$ is PAC-learnable

The below proof uses Empirical Risk Minimisation to show that $H$ must have a finite size to be PAC-learnable.

notice $R_S(h_S)$ $\le$ $R_S(h^*)$. so
$$R(h_S) - R(h^*) $$
$$ = R(h_S) - R_S(h_S) + R_S(h_S) - R_S(h^*) + R_S(h^*) - R(h^*)   $$
$$ \le R(h_S) - R_S(h_S) + R_S(h^*) - R(h^*) $$
$$ \le \bigl|  R(h_S) - R_S(h_S) \bigr| + 
\bigl| R_S(h^*) - R(h^*) \bigr| $$
$$\le \sup_{h \in H} \bigl|  R(h) - R_S(h) \bigr| 
+ \sup_{h \in H} \bigl|  R(h) - R_S(h) \bigr|$$
$$ = 2 \sup_{h \in H} \bigl| R(h) - R_S (h) \bigr| $$
We now have this inequality for estimation error
$$R(h_S) - R(h^*) \le 2\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr| $$
and a notion of **generalisation error**
$$R(h_S) - R_S(h_S) \le \sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|  \quad (*)$$
- (difference between expected risk and empirical risk for found $h_S$)
- (not sure if generalisation error is just $R(h_S)$ or above)

We now want to upper bound either $R(h_S) - R_S(h_S)$ or $\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|$.
- notice with large enough sample size, $R_S(h)$ will converge to $R(h)$, asymptotically.
- We will use a non-asymptotical measurement between the two

**Hoeffding's inequality**:
Let $X_1 \dots X_n$ be independent random variables, such that $X_i \in [a_i, b_i]$ with probability 1. Let $S_n = \frac{1}{n} \sum_{i=1}^n X_i$ . Then probability that difference between $S_n$ and its expected value will be higher than some $\epsilon$ is upper bounded as follows
$$p \bigl( |S_n - \mathbb{E}[S_n]| > \epsilon   \bigr)
\le 2 \exp \biggl( 
\frac {-2n^2 \epsilon^2} {\sum_{i=1}^n (b_i - a_i)^2}
\biggr)$$
for any $\epsilon > 0$.

Now consider that loss functions on varying training sets are independent random variables. We can assume that each $\ell (X_i, Y_i, h) \in [0, M]$. So we have
$$p \bigl( | R_S(h) - R(h)| > \epsilon   \bigr)
\le 2 \exp \biggl( 
\frac {-2n^2 \epsilon^2} {M^2}
\biggr) \quad (**)$$
for any $\epsilon > 0$. 
We now show how to use $(**)$ to upper bound $\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|$ 

**union bound**:
For any events $A_1 \dots A_n$  we have
$$p(\cup_{i=1}^n A_i) \le \sum_{i=1}^n p(A_i) $$
**when A implies B**:
$$p(A) \le p(B) $$
Using implication we have
$$p \bigl( \sup_{h \in H}|R(h) - R_S(h)| \ge \epsilon   \bigr)
\le p \bigl( \cup_{h \in H} |R(h) - R_S(h) | \ge \epsilon \bigr)
$$
Using union bound we have
$$ p \bigl( \cup_{h \in H} |R(h) - R_S(h) | \ge \epsilon \bigr)
\le \sum_{h \in H} p \bigl( |R(h) - R_S(h)| \ge \epsilon \bigr)
$$
And finally from $(**)$ we have
$$\sum_{h \in H} p \bigl( |R(h) - R_S(h)| \ge \epsilon \bigr)
\le 2|H| \exp \frac{-2n\epsilon^2} {M^2}$$
Let $\delta = 2|H| \exp \frac{-2n\epsilon^2} {M^2}$
Then
$$\epsilon = M \sqrt{\frac{\log{H} + \log{2/\delta}}
{2n}} $$
So with probability $\ge 1 - \delta$ we have
$$\sup_{h \in H}|R(h) - R_S(h)| \le 
M \sqrt{\frac{\log{H} + \log{2/\delta}} {2n}} $$
Note from $(*)$ we, have a **generalisation error bound**:
$$R(h_S) \le R_S(h_S) +
\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr| $$
Now, a hypothesis class $H$ is PAC-learnable if it is of finite hypotheses.
Because
$$ p \Biggl\{ 
R(h_S) - R(h^*) \le 2\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|
\le 2M \sqrt{\frac{\log{H} + \log{2/\delta}} {2n}}
\Biggr\} \ge 
1 - \delta
$$
And from $\delta$ , the size of training data is given by $n$:
$$n = \frac {M^2}{\epsilon} \log \frac {2|H|} {\delta}$$
i.e., $n > \text{poly}(1/\delta, 1 / \epsilon)$

This shows that as the size of $H$ increases, we need more training samples, so it is in our interest to keep $H$ small.

### can an infinitely sized $H$ be PAC-learnable?

If $H$ has infinitely many hypotheses, how can we upper bound
$$\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|$$
?

We will use VC dimension to show if the classifier is binary, then an infinitely sized $H$ can still be PAC-learnable.

Consider a binary classifier such that
$$H = \{ (h_1^1, \dots , h_{n_1}^1 ),  \dots , (h_1^G, \dots , h_{n_G}^G ) \}  $$
Although $H$ here has infinitely many hypotheses, we can group hypotheses into finite groups, where each group is given by the same hypotheses
$$h(X_1), \dots , h(X_n) $$
Let $h^1 \dots h^G$ represent each group, so we can write
$$ H' = \{ h^1,  \dots , h^G \} $$
**how to find $H'$**:

**growth function**:
The growth function $\prod_H : \mathbb{N} \mapsto \mathbb{N}$ , for a hypothesis class $H$ is defined by
$$\prod_H (n) = \max_{X_1 \dots X_n} 
| \{ h(X_1), \dots , h(X_n) : h \in H \}| \quad , 
\forall n \in \mathbb{N}$$
i.e. the growth function is the max group of hypotheses with the same predictions as the others.

We can use this to upper bound $\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|$.

$$ p \biggl( \sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|
\ge \epsilon \biggr) $$
$$ \le 2p \biggl( \sup_{h \in H}\bigl| R_{S'}(h) - R_S (h) \bigr|
\ge \epsilon /2 \biggr)$$
$$ \le 4p \biggl( \sup_{h \in H} 
\frac {1}{n} \biggl| 
\sum_{i=1}^n \sigma_i \ell(X_i, Y_i, h) \biggr|
\ge \epsilon / 4
\biggr)$$
$$\le 4 \prod_H (n) p \biggl( \sup_{h \in H} 
\frac {1}{n} \biggl| 
\sum_{i=1}^n \sigma_i \ell(X_i, Y_i, h) \biggr|
\ge \epsilon / 4
\biggr) $$
$$\le 4 \prod_H (n) \exp( - n \epsilon^2 / 32 M^2) $$
**shattering**:
The data points $\{ X_1, \dots X_n \}$ is shattered by $H$ when $H$ realises all possible binary predictions. That is, $\prod_H(n) = 2^n$.

**VC dimension**:
The VC dimension of $H$, is the size of the largest set that can be fully shattered by $H$.
$$\text{VC dimension} (H) = \max_n \{n : \prod_H (n) = 2^n \} $$
If the $H$ is of finite VC dimension, it is PAC learnable as follows.

Let $H$ be a hypothesis class with $\text{VC dimension} (H) = d$, 
then for all $n \ge d$ 
$$\prod_H(n) \le \Bigl( \frac {en} {d} \Bigr)^d $$
So we continue to upper bound $\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr|$ as follows.

$$4 \prod_H (n) \exp( - n \epsilon^2 / 32 M^2)
\le 8 \Bigl( \frac {en} {d} \Bigr)^d 
\exp( - n \epsilon^2 / 32 M^2)$$
Let $\delta = 8 \Bigl( \frac {en} {d} \Bigr)^d \exp( - n \epsilon^2 / 32 M^2)$ 
We have
$$\epsilon =  M \sqrt{\frac{32 \bigl( d \log {\frac {en} {d}}
+ \log{8/\delta} \bigr)}
{n}}  $$
With probability at least $1 - \delta$ we have
$$\sup_{h \in H}\bigl| R(h) - R_S (h) \bigr| =
M \sqrt{\frac{32 \bigl( d \log {\frac {en} {d}}
+ \log{(8/\delta)} \bigr)}
{n}}  $$
From $\delta$ we have
$$n = \frac{32 M^2} {\epsilon^2}
\bigl( d \log {\frac {en} {d}} + \log{(8/\delta)} \bigr)$$
and so $n > \text{poly} ( 1/\delta, 1/\epsilon)$ and $H$ is PAC-learnable.