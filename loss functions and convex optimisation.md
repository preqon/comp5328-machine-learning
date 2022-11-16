# loss functions and convex optimisation

In the convex [[optimisation method]], we use a convex surrogate $\ell$ of the 0-1 loss function.
$$\arg \min_{h \in H} \frac{1}{n} \sum_{i=1}^n l(X_i, Y_i, h) $$
This is because the 0-1 loss function may not have one minimum, but a convex function does. We also want a convex function that is smooth.
- optimisation can then exploit derivative information

Popular convex surrogates:
- hinge loss
$$\max \{ 0, 1 - Yh(X) \} $$
- Logistic loss
$$\log_2 (1 + \exp(- Yh(X) )) $$

- least square loss
$$(Y - h(X))^2$$
- exponential loss
$$\exp( - Yh(X)) $$

Non-convex surrogates
- Cauchy loss
- Correntropy loss (or 'Welsch loss')

## classification calibrated surrogate loss functions

will result in a classifier that approaches the same accuracy as when using 0-1 loss, if training data is sufficiently large

i.e. with 
$$\arg \min_{h_n} \frac{1}{n} \sum_{i=1}^n l(X_i, Y_i, h) $$
$$\arg \min_{h_c}  \mathbb{E} [ 1_{\{ Y_{i} \ne sign(h(X_i)) \}} ]  $$
then
$$\mathbb{E} [ 1_{\{ Y_{i} \ne sign(h_n(X_i)) \}} ]
\to 
\mathbb{E} [ 1_{\{ Y_{i} \ne sign(h_c(X_i)) \}} ]$$

**to show a surrogate loss function is classification calibrated**:
- Let $\phi(Yh(X)) = \ell (X, Y, h)$ .
- Given $\phi$ is convex, the loss function is classification calibrated iff $\phi$ is differentiable at 0, and $\phi'(0) < 0$.

## convex optimisation definitions

**convex set:**
- a set $C \in \mathbb{R}^d$ is convex if
	- $x,y \in C$  
	- $\theta x + (1 - \theta)y \in C$ for any $\theta \in [0, 1]$[^1]

[^1]: inclusive range

**convex function:**
A function $f : \mathbb{R}^d \mapsto \mathbb{R}$ is convex if its domain is a convex set and

$$f (\theta x + (1 - \theta)y) \le
\theta f(x) + (1 - \theta) f(y)$$
for all $x, y \in \text{domain } f$ , when $0 \le \theta \le 1$.
notice $x$ and $y$ are vectors here.

**differentiable function**:
A function is differentiable if its gradient exists, for all $x$.

**differentiable and convex**:
A differentiable function $f$, with convex domain, is convex iff
$$f(x) \ge f(y) + \nabla f(y) ^T (x - y)$$for all x, y $\in$ domain $f$.

**twice differentiable**:
A function $f$ is twice differentiable, if the Hessian matrix

$$H_{ij} = \frac{\partial^2 f(x)} 
{\partial x_i \partial x_j}, \forall x \in \text{domain} f$$
exists.

**twice differentiable and convex**:
If we assume that $f$ is twice differentiable, then $f$ is convex iff the Hessian matrix is positive semidefinite for all points in the domain.

**positive semidefinite**:
A square matrix $H \in \mathbb{R}^{d \times d}$ is positive semidefinite iff
$$x^T H x \ge 0, \forall x \in \mathbb{R}^d $$
or, if all its eigenvalues are non-negative.

**pointwise maximum and convex functions**:
If $f_1$ and $f_2$ are convex functions, then their pointwise maximum $f$ given by
$$f(x) = \max \{ f_1(x), f_2(x) \} $$
is also convex. Note domain of $f$ here is intersection of $f_1$ and $f_2$ domains.

**non-negative weighted sum:**
$$f(x) = \theta_1 f_1(x) + \theta_2 f_2(y) $$
**composition with affine mapping:**
$$g(x) = f(Ax + b) $$
**pointwise maximum:**
$$f(x) = \max_i \{ f_i (x) \} $$
**optimality criterion**:
When a point $x$ in $f$ is optimal, it must also be feasible (satisfies objective's constraints) and
$$ \nabla f (x)^T (y - x) \ge 0 $$ for all feasible $y$ $\in$ domain $f$.

**feasible set X**:
All the points $x$ in $f$ that satisfy the constraints of the objective.

## unconstrained convex optimisation

The convex optimisation problem can be written generally as follows, where $f$ is convex.
$$\arg \min_h f(h) $$
In classification, we search for a hypothesis function $h \in H$ that minimises our chosen objective, i.e.

$$\arg \min_{h \in H} \frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h)
 = \arg \min_h f(h)$$
where $\ell$ is a convex surrogate loss function of 0-1 loss.

**Taylor's Theorem:**
Let $k \ge 1$ be an integer, and $f : \mathbb{R} \mapsto \mathbb{R}$ be a function that is $k$ times differentiable at point $a \in \mathbb{R}$. Then there exists a function $h_k : \mathbb{R} \mapsto \mathbb{R}$ such that 
$$ f(x) = f(a) + f'(a)(x - a) + \dots +
\frac {f^{(k)}(a) } {k!} (x - a)^k +
h_k(x) \cdot (x - a)^k$$
and $\lim_{x \to a} h_k(x) = 0$ .

*You should be able to write out the Taylor series of a satisfying function at any point $a$, for some $k$. Then write as $x$ approaches $a$, $h_k(x)$ approaches zero faster than $(x-a)^k$ does. Write this using small-o notation. 
i.e. just turn the last term from $h_k(x)(x - a)^k$ into $o((x-a)^k)$*

**Small-o notation**:
The notation $f(x - 1) = o((x - 1)^2), x \to 1$ means that when $x$ approaches $1$,
1. $\frac{f(x -1)}{(x-1)^2} \to 0$ 
2. $f(x - 1)$ coverges to $0$ faster than $(x - 1)^2$.

### gradient descent method
Let 
$$f(h) = \frac {1} {n} \sum_{i=1}^n \ell(X_i, Y_i, h), $$
and $$ h_{k+1} = h_k +
\eta d_k . \quad (2)$$
where
- $\eta$ is learning rate
- $d_k$ is chosen direction to step towards

By Taylor's theorem we have
$$f(h_{k+1}) = f(h_k) + \eta \nabla f(h_k)^T d_k +
o(\eta). $$
For positive, but sufficiently small $\eta$, $f(h_{k+1})$ is smaller than $f(h_{k})$, if the direction $d_k$ is chosen such that 
$$\nabla f(h_k)^T d_k < 0 
\quad \text{when} \quad 
\nabla f (h_k) \ne 0 \quad \quad (4).$$
We can iteratively update $h$ using $\eta$ and $d_k$ according to (2).

##### how to find $d_k$:
set 
$$ d_k = -D^k \nabla f (h_k)$$
so our iterative rule is now
$$h_{k+1} = h_k -
\eta D^k \nabla f (h_k). $$
where
- $D^k$ is a positive definite symmetric matrix such that
$$ \nabla f (h_k) ^T D^k \nabla f (h_k) > 0$$
- $\eta$ is a positive such that
$$f(h_{k+1}) = f(h_k) - \eta \nabla f(h_k)^T D^k \nabla f(h_k)$$

In **steepest descent**, $D^k = I$.
In **Newtown's method** $D^k = [\nabla ^2 f(h)]^{-1}$.

So, for steepest descent, we have
$$ d_k = - \nabla f(h_k) $$
and our update rule becomes
$$h_{k+1} = h_k -
\eta \nabla f(h_k)$$
Notice this formulation of $d_k$ satisfies (4) above.

##### how to find $\eta$:

Naiive search ('exact line search') is expensive:
$$\eta =  \arg \min_\eta f(h_k - \nabla f(h_k)) $$
Thankfully, the Lipschitz smooth constant $L$ exists for the gradient! If we know $L$, we have
$$ h_{k+1} = h_k - \frac {1} {L} \nabla f(h_k) $$
$$f(h_{k+1}) \le f(h_k) -
\frac {1}{2L} || \nabla f(h_k) ||^2 .$$
So we choose $\eta = 1/L$. (I think second equation is just showing us that we indeed descend by using this $\eta$).

This requires $f$ to be Lipschitz continuous which is when

$$| f(x_1) - f(x_2) |
\le L || x_1 - x_2 || , \quad 
\forall x \in \text{domain} f$$

##### gradient convergence rate

how many steps do we need to find optimal solution $h_s$?

$$ h_s = \arg \min_h f(h) $$
When the objective function $f$ is strongly convex, and has Lipschitz gradient, we have a **linear convergence rate**:
$$f(h_{k+1}) - f(h_s) \le 
\Bigl( 1 - \frac {\mu} {L} \Bigr)^k
(f(h_1) - f(h_s) )$$
for some value of $\mu \in \mathbb{R}$.

A function $f$ is **$\mu$-strongly convex** when
$$ f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle +
\frac {\mu} {2} || x - y || ^2 ,
\quad \forall x,y$$
if and only if
$$\mu \mathbf{I} \preccurlyeq \nabla^2 f(x), \forall x $$
(curly inequality symbol means there is some partial ordering between the matrices)

Notice the convergence rate, in this strongly convex case, is dominated by left term, and so can be written as

$$ O\Bigl( ( 1 - \frac {\mu} {L})^k \Bigr)$$
**when objective function is convex** and has Lipschitz gradient, convergence rate is only
$$O (1/k)$$
##### using newtown's method of gradient descent

If we use Newtown's method instead, the rate is
$$\prod _{i=1} ^k \rho_k, \quad \rho \to 0$$
again when $f$ has Lipschitz gradient and is strongly convex.

Finding $D_k$, where $D^k = [\nabla ^2 f(h)]^{-1}$, as required by newton's method, is computationally expensive.

Practical alternatives:
- Modify the Hessian to be positive-definite.
- Only compute Hessian every m iterations
- Only use diagonals of the Hessian
- Use an approximation of the Hessian

## constrained optimisation

Find $$\arg \min_h f(h)$$ subject to constraints on other functions $a_i$ and $b_j$ where
- $a_i (h) = 0 \quad \forall i$ (equality constraints)
- $b_j(h) \le 0 \quad \forall j$ (inequality constraints)
where $b_j$ are convex functions and $a_i$ are affine functions.

**affine functions:**
$a(h)$ is affine if $a(h) = \alpha^T h - \beta$
- ($\alpha, \beta, h$  are vectors.)