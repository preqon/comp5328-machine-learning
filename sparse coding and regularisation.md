# sparse coding and regularisation

See [[dictionary learning and NMF]].

For given set of samples, we learn $DR$ such that
$$ X \approx D R $$
We want
- dictionary to be overcomplete
- representation to be sparse (i.e. few non-zero values, most elements are zero)

*i think this is the idea:*
We want dictionary $D$ to be overcomplete, to have the best chance of containing bases for whole domain of samples. 
However, there are naturally infinitely many solutions for the representation $\alpha_i \in R$ for each sample $x_i \in X$. How do we choose the best one? We want the *sparse* representation, which is often unique.

benefits of sparsity:
- parsimony: sparse representations are short and highlight the essential features, making them easy to describe.
- denoising: if a signal is a mixture of noise and true signal, the sparse representation can be considered an approximation of true signal.
- data compression: sparse representation allows data compression.
- compressed sensing: we can reduce the amount of types of measurements needed to construct a signal, if we know its sparse basis. basis can be learned via dictionary learning.

**$\ell_p$ norm**:

$$ || \alpha ||_p = \biggl( \sum_{j=1}^k |\alpha_j|^p \biggr) ^ {1/p}$$
As $p \to 0$ we get a count of non-zeros in the vector, so we use $\ell_0$ to measure sparsity.

## ensuring sparsity in representations

We can use $\ell_0$ norm to ensure sparsity is below threshold during dictionary learning
1. $$\min_{\alpha} || X - D \alpha ||_F^2 \quad \text{s.t. } 
\forall i, ||\alpha||_0 < L$$
or alternatively optimise sparsity, such that fitting $X$ is below threshold
2.  $$ \min_{\alpha} || \alpha||_0 \quad \text{s.t. }
|| X - D \alpha ||_F^2 \le \epsilon$$
However $\ell_0$ is non-differentiable. We can use $\ell_1$ as a surrogate, and optimise the following, such that fitting $X$ is below threshold.
1. $$ \min_{\alpha} || \alpha||_1 \quad \text{s.t. }
|| X - D \alpha ||_F^2 \le \epsilon $$
or alternatively, use a regularisation approach. Namely, we add a regularisation term to objective during dictionary learning to make $R$ sparse.
$$ \arg \min_{D, R} = || X - DR ||_F^2 + \lambda \psi (R)$$

and use $\ell_1$ for regularisation function $\psi$.
2. $$\min_{\alpha} || X - D \alpha ||_F^2 +
\lambda || \alpha ||_1$$
## our approaches to find sparse representations are not stable

**algorithmic stability**:

Suppose there are two sets of training data, $S$ and $S_i$ that differ only by one example.
An algorithm is uniformly stable if, for any example $(X, Y)$:
$$| \ell (X, Y, h_S) - \ell(X, Y, h_{S_i})| < \epsilon(n) $$

i.e., if the difference between prediction is negligible, after small changes in training data, the algorithm is stable. 
- note that $\epsilon(\cdot)$ is a function of number of training samples, and $\epsilon(n)$ will vanish as  $n$ goes to infinity.

A good stable algorithm also has low **generalisation error** (see [[hypothesis complexity and generalisation]]) as follows.
$$ \mathbb{E} [ R(h_S) - R_S(h_S)]$$
$$ = \mathbb{E}_S \biggl[
	\mathbb{E}_{X,Y} \Bigl[ \ell(X, Y, h_S) \Bigr] -
	\frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h_S)
\biggr] $$

Now $\mathbb{E}_{X,Y}$ should not change if we make small change to training set $S$.
$$ = \mathbb{E}_S \biggl[
	\mathbb{E}_{X, Y} \Bigl[ \ell(X', Y', h_S) \Bigr] -
	\frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h_S)
\biggr]  $$
$$ = \mathbb{E}_S \biggl[
	\mathbb{E}_{S'} \Bigl[ 
	\frac{1}{n} \sum_{i=1}^n \ell(X_i', Y_i', h_S)
	\Bigr] -
	\frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h_S)
\biggr]  $$
$$ = \mathbb{E}_{S, S'} \biggl[
	\frac{1}{n} \sum_{i=1}^n \Bigl( \ell(X_i', Y_i', h_S)
	 - \ell(X_i, Y_i, h_S) \Bigr)
\biggr]  $$
Since the difference in $h_S$ and $h_{S'}$ is small in a stable algorithm we have
$$= \mathbb{E}_{S, S'} \biggl[
	\frac{1}{n} \sum_{i=1}^n \Bigl( \ell(X_i', Y_i', h_S)
	 - \ell(X_i', Y_i', h_{S'}) \Bigr)
\biggr]  $$
$$ \le \epsilon ' (n) $$

Finding a sparse $R$, by using $\ell_p$ norms with $p \to 0$ to measure sparsity, is **not stable**. This makes the above approaches listed not stable.

Generally speaking, we can use $\ell_2$ norm regularisation to make a learning algorithm stable.
- if the employed surrogate loss function is convex
$$h_S = \arg \min_{h \in H}
	\frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h) +
	\lambda ||h||_2^2$$

- When the convex surrogate is Lipschitz continuous with respect to $h$, for $||X||_2 < B$, you can prove:
$$ | \ell(X, Y, h_S) - \ell(X, Y, h_{S^i}) | < 
\frac{2L^2B^2} {\lambda n}$$
note that this does not mean $\ell_2$ will be a good measure of sparsity and that we should add a $\ell_2$ regularisation to dictionary learning to make our representation sparse. 
Firstly, from above we have a better count of zeros if $p \to 0$, but also $\ell_2$ is smooth, and the solution that minimises a smooth function can be dense (not sure why). $\ell_1$ is convex but is non-smooth, so still promotes sparsity.

minimising smooth function disadvantage: something to do with possible solutions to the minimisation being closer together, for closer points in $X$.
non-smooth functions (or 'pointed'), mean possible solutions to the minimisation have a higher chance of being further apart, for closer points in $X$?
