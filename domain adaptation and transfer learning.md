# domain adaptation and transfer learning

*maybe watch this lecture if time*

see notation in [[hypothesis complexity and generalisation]].

the assumption in machine learning is that the underlying distribution of training data will be the same for test data.

if the underlying distribution between datasets for two tasks are also the same, we should use the same machine learning algorithm.

recall that we call the underlying distribution of input data its domain.

**domain adaptation**:
reduce the difference between source and target domains, i.e. match the two underlying distributions

**transfer learning**:
exploit training samples ('extract knowledge') from some source domain, to improve performance in a target domain
i.e. train and test on different domains.

suppose we have source domain with samples

$$ \{ (X_1^S, Y_1^S) \dots (X_n^S, Y_n^S) \} $$
and target domain with samples
$$ \{ (X_1^T, Y_1^T) \dots (X_n^T, Y_n^T) \} $$
let's say the target domain is such that most $Y^T$ are not observed/difficult to find.

To use a classifier trained on $S$ we must have a knowledge of how the underlying distribution $p(X, Y)$ is different to that of $T$.

## importance reweighting (transfer learning)

consider expected risk in target domain
$$R^T(h) = \mathbb{E}_{(X,Y) \sim p_t(X,Y)}
[\ell (X, Y, h)]$$
$$ = \int_{(X,Y)} \ell(X, Y, h)p_t(X,Y) \thinspace dXdY $$
now add in term for probability distribution for source domain
$$ = \int_{(X,Y)} \ell(X,Y,h)
\frac{p_t(X,Y)} {p_s(X,Y)} p_s(X,Y) \thinspace dXdY$$
$$ = \mathbb{E}_{(X,Y) \sim p_s(X,Y)} 
[\ell(X,Y,h) \frac{p_t(X,Y)} {p_s(X,Y)} ]$$
which we will denote as
$$  = \mathbb{E}_{(X,Y) \sim p_s(X,Y)} 
[ \beta(X,Y) \ell(X,Y,h) ]$$
where
- $\beta(X,Y)$ = $p_t(X,Y) / p_s(X,Y)$ 

so we have expected risk in the target domain equal to the expected loss in the source domain, multiplied by this beta distribution, which measures the difference in two distributions.

so knowing $\beta(X,Y)$ lets us approximate the expected risk in the target domain, using the source domain.

i.e. we approximate $R^T(h)$ using

$$ \frac{1}{n_S} \sum_{i=1}^{n_S} 
\beta(x_i^S, y_i^S) \ell(x_i^S, y_i^S)$$
by minimising above risk, we optimise for target domain, by training in source domain.

### covariate shift model of transfer learning

**product rule of joint probability**:
$$p(X,Y) = p(Y | X)p(X) = p(X |Y)p(Y) $$

in this model, assume $p_t(Y|X) = p_s(Y|X)$ and that $p_t(X) \ne p_s(X)$.

we then have 
$$\beta(X,Y) = \frac{p_t(X,Y)}  {p_s(X,Y)}  $$
$$  = \frac{p_t(Y|X)p_t(X)} {p_s(Y|X)p_s(X)} = \frac{p_t(X)}{p_s(X)} $$
$$ = \beta(X) $$
Note $\beta(X)$ can be learned by [[domain adaptation and transfer learning#kernel mean matching|kernel mean matching]].

### target shift model of transfer learning

in this model, assume $p_t(X|Y) = p_s(X|Y)$ and that $p_t(Y) \ne p_s(Y)$.

we then have
$$\beta(X,Y) = \frac{p_t(X,Y)}  {p_s(X,Y)}  $$
$$= \frac{p_t(X|Y)p_t(Y)} {p_s(X|Y)p_s(Y)} = \frac{p_t(Y)}{p_s(Y)} $$
$$ = \beta(Y) $$
However $\beta(Y)$ is not easy to learn if the target domain has no labels.
But we use kernel mean matching again.

we have from the above
$$p_t(Y) = \beta(Y)p_s(Y)$$
so from $p_t(X) = p_t(X|Y)p_t(Y)$ we have
$$p_t(X) = \int p_t(X | Y) \beta(Y) p_s(Y) \thinspace dY $$
and from our assumption
$$  = \int p_s(X|Y)\beta(Y) p_S(Y) \thinspace dY $$
Now match the distributions $p_t(X)$ and $\int p_s(X|Y)\beta(Y) p_S(Y) \thinspace dY$ 

We minimise with respect to $\beta$

$$\min_{\beta} || \space \mu(p_t(X))
- \mathbb{E}_{Y \sim p_s(Y)} 
\bigl[ \mu( p_s (X|Y)) \beta(Y) 
\bigr] \space ||^2$$
empirically written as

$$\min_{\beta} \Biggl| \Biggl| 
\space \frac{1}{n_T} \sum_{i=1}^{n_T} \phi(x_i^T)
- \frac{1}{n_S}\sum_{i=1}^{n_S} 
\beta(y_i^S) \hat\mu (p_s(X|y_i^S)) 
\space \Biggr|\Biggr|^2  $$
subject to
$$\beta(y_i^S) \ge 0 , \space \frac{1}{n_S} \sum_{i=1}^{n_S} 
\beta(y_i^S) = 1 $$

## domain adaptation

note that a domain in machine learning can be regarded as $p(X,Y)$ 
- if $Y \in \{1, 2, \dots , C\}$ we have a classification task
- if $Y \in \mathbb{R}$ we have a regression task

We have source domain $p_s(X,Y)$ and target domain $p_t(X,Y)$.

What conditions on these two distributions can we directly use a classifier trained on $S$, on $T$?
- we should reduce the difference between these distributions

### kernel mean matching

consider a function
$$ \phi : X \to \mathcal{H} $$
where $\mathcal{H}$ is a Reproducing Kernel Hilbert Space (RKHS), with kernel function
$$ K(x_1, x_2) = \langle \phi(x_1) , \phi(x_2) \rangle $$
where $\langle \cdot , \cdot \rangle$ is the inner product.

let 
$$ \mu (p(X)) = \mathbb{E}_{X \sim p(X)} [ \phi (X)] $$
where $p(X)$ is a marginal distribution on feature space.
note the expectation $\mu$ is a bijective function if $K$ is a universal kernel.

Now if we consider 
$$ \mu (p(X)) = \mathbb{E}_{X \sim p_S(X)} [ \beta(X) \phi (X)] $$
where
- $\beta(X) \ge 0$ 
- $\mathbb{E}_{X \sim p_S(X)} [ \beta(X)] = 1$ 

then we have
- $\beta(X) = p_t(X) / p_s(X)$ 
because we can write
$$\mu(\beta(X)p_s(X)) = \mu(p_t(X)) $$
now we would like to learn $\beta$.

$$\min_{\beta} || \space \mu(p_t(X))
- \mathbb{E}_{X \sim p_s(X)} 
\bigl[ \beta(X) \phi(X) 
\bigr] \space ||^2 $$
subject to 
$$\beta(X) \ge 0, \space \mathbb{E}_{X \sim p_s(X) } [\beta(X)] = 1 $$
given samples then, from each domain, we use the empirical mean to approximate $\mu$ and $\mathbb{E}_{X \sim p_s(X)}$ above.
$$\min_{\beta} \Biggl| \Biggl| 
\space \frac{1}{n_T} \sum_{i=1}^{n_T} \phi(x_i^T)
- \frac{1}{n_S}\sum_{i=1}^{n_S} 
\beta(x_i^S) \phi(x_i^S) 
\space \Biggr|\Biggr|^2  $$
subject to  $$\beta(x_i^S) \ge 0 , \space \frac{1}{n_S} \sum_{i=1}^{n_S} 
\beta(x_i^S) = 1$$
