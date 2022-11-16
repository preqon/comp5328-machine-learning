# learning with noisy data

*apparently in ### optimality criterion across surrogate loss functions, defining g is not necessary, according to people on Ed. maybe work out what they mean.*

## definitions

**likelihood**:
likelihood of $\theta$ is the probability of observing the data given $\theta$.
$$ p(S | \theta) $$
**prior**:
prior of $\theta$ is the probability distribution (from any prior knowledge) we have of $\theta$
$$p(\theta) $$
**posterior**:
posterior of $\theta$ is the probability of observing $\theta$ given the data.
$$ p(\theta | S) $$
- consider this is proportional to likelihood x prior, following Bayes' rule.

**Bayes' rule**:
$$ p(\theta | S) = 
\frac {p(S | \theta) p(\theta)} {p(S)}
$$
**maximum likelihood estimation**:
- estimating the value of $\theta$ that maximises the likelihood $p(S|\theta)$, i.e. makes the observed data most probable.
- for training samples (X, Y) we have
$$ p(S|\theta) = \prod_{i=1}^n
p( (x_i, y_i) \space | \space \theta) $$
or can be thought of as
$$p(S|\theta) = \prod_{i=1}^n
p((y_i | x_i) \space , \space \theta)$$
consider, from Bayes rule that the maximum posterior is proportional to the maximum likelihood.
$$p(\theta | S) \propto  p(S | \theta) p(\theta) $$
$$ \arg \max_\theta p(\theta | S)
= \arg \max_\theta p(S | \theta) p(\theta) $$
that is, finding the value of $\theta$ that maximises its likelihood, should also maximise its posterior.
We can turn the optimisation into a minimisation as follows
$$ \arg \min_\theta(- \log p(\theta | S))
= \arg \min_\theta (- \log p(S | \theta) - \log p(\theta) )  $$
## linear regression with Gaussian noise

- say we have a regression problem, i.e. find $y = h(x)$, but we also want to model additive noise, so we find $$ y = h(x) + \epsilon $$
where
$$ \epsilon \sim \mathcal{N} (0, \beta^{-1}) $$
so our probabilistic model is
$$ p( y | x , h, \beta) = \mathcal{N}( y| h(x), \beta^{-1}) $$
Written as an estimation of likelihood, we have
$$p (S | \theta , h, \beta) = \prod_{i=1}^n 
	\mathcal{N} (y_i | h(x_i) \space , \space \beta^{-1} ) $$
Now write the above in terms of empirical risk, so that we have an objective function to minimise.
$$ = \prod_{i=1}^n \sqrt{ \frac{\beta}{2 \pi}}
\exp \biggl(  
	- \frac {\beta(y_i - h(x_i))^2} {2}
\biggr)
$$
$$ = \biggl( \frac{\beta}{2 \pi} \biggr)^{n/2} \prod_{i=1}^n 
\exp \biggl(  
	- \frac {\beta(y_i - h(x_i))^2} {2}
\biggr)  $$
Now taking the negative natural log of both sides gives
$$ - \ln p (S | \theta , h, \beta)
= - \frac{n}{2} \ln\beta + \frac{n}{2} \ln (2\pi) 
+ \frac{\beta}{2} \sum_{i=1}^n (y_i - h(x_i))^2 $$
Notice the last sum is just $n \times \text{mean squared loss}$ i.e. an empirical risk
$$  - \ln p (S | \theta , h, \beta)
= - \frac{n}{2} \ln\beta + \frac{n}{2} \ln (2\pi) 
+ \frac{\beta}{2}n R_S(h)$$
which gives us our objective, which will give us a maximum likelihood.

What if we want to write the above optimisation problem as an estimation of posterior?
From above definitions, we know
1. maximum posterior is proportional to maximum likelihood
2. maximising posterior can be written as a minimisation 

so we have LHS posterior of $h$, RHS likelihood of $h$

$$ \arg \min_h(- \ln p(h \space | \space S, \theta, \beta^{-1}))
= \arg \min_h (- \ln p(S \space | \space \theta, h, \beta^{-1}) - \ln p(h) )   $$
$$ = \arg \min_h \biggl(
- \frac{n}{2} \ln\beta + \frac{n}{2} \ln (2\pi) 
+ \frac{\beta}{2}n R_S(h) - \ln p(h)
\biggr) $$
so we can use the objective
$$ R_S(h) - \ln p(h) $$

**example: given some prior**
- let's say we expect h(x) to be a nine-degree polynomial with weights (coefficients) $w_i$ for $i \in {0 \dots 9}$.
- we assume the prior distribution
$$p(h) = \prod_{i=1}^9 
\sqrt{\frac{\tau}{2\pi}}
\exp \biggl( - \frac{\tau w_i^2}{2} \biggr)$$
So to find maximum posterior we use the following optimisation from above
$$\arg \min_h  R_S(h) - \ln p(h)$$
$$ = \arg \min_h  R_S(h) - 5 \ln \tau + 5 \ln(2 \pi)
+ \frac{\tau}{2} n \sum_{i=0}^9 w_i^2$$
which is equivalent to minimising
$$\arg \min_h R_S(h) + \lambda \sum_{i=0}^9 w_i^2 $$
where $\lambda = \tau / \beta$.
which is equivalent to minimising empirical risk, with some regularisation term for the weights.
$$ = \arg \min_h R_S(h) + \lambda ||w ||_2^2  $$

## bias and variance

**underfitting:**
- does not learn training data well
- large empirical risk

**overfitting**:
- learns trained data well, but cannot generalise
- large difference between train and test error

small $\lambda$ (low regularisation) = low bias, high variance
 --> overfitting

large $\lambda$ (high regularisation) = high bias, low variance
--> underfitting

this is because regularisation term constrains hypothesis, ideally closer to expected risk.

**avoid overfitting**:
- reduce hypothesis complexity: won't be as sensitive to subtle variations in training data
- increase sample size: training data more likely to represent underlying distribution due to law of large numbers.

## robustness of surrogate loss functions

consider these surrogate loss functions
1. least squares loss
$$(Y - h(X))^2$$
2. absolute loss
$$| Y - h(X)| $$
3. cauchy loss
$$\log_2 \biggl( 
1 + \Bigl( \frac{Y-h(x)} {\sigma} \Bigr)^2
\biggr) $$
4. correntropy loss (welsch loss)
$$\log_2 \biggl[ 
1 - \exp \biggl( -\Bigl( \frac{Y-h(x)} {\sigma} \Bigr)^2 \biggr)
\biggr]  $$

consider these models of distributions of noise, where we assume noise is experienced as $\epsilon = Y - h(X)$.

1. Gaussian distribution
$$p(\epsilon | X, Y, h, \beta^{-1}) 
= \sqrt{\frac{\beta}{2\pi}} \exp
\biggl( - \frac{\beta \epsilon^2} {2} \biggr)$$
2. Laplacian distribution
$$p(\epsilon | X, Y, h, \sigma) = 
\frac{1} {\sqrt{2} \sigma} \exp
\biggl( - \frac{\sqrt{2}|\epsilon|} {\sigma} \biggr)$$
3. Cauchy distribution
$$ p(\epsilon | X, Y, h, \gamma) =
\frac{1}{\pi \gamma \Bigl( 1 + (\epsilon / \gamma)^2\Bigr)}$$
### laplacian regression

from above, assume the laplacian distribution of noise.
the likelihood of training samples, our hypothesis, and experienced noise is
$$p (S | X, h, b) = \biggl(
\frac{1}{\sqrt{2}\sigma} \biggr)^n \prod_{i=1}^n \exp 
\biggl( - \frac {\sqrt{2}|y_i - h(x_i)|} {\sigma} \biggr)$$
taking the negative log of both sides, to turn into a minimisation problem yields,
$$ - \ln p (S | X, h, b) =
n \ln (\sqrt{2}\sigma) + \frac{\sqrt{2}} {\sigma}
\sum_{i=1}^n |y_i - h(x_i)|
$$
Notice we only need the rightmost term to minimise with respect to $h$, which is also just the empirical risk when using absolute loss.
Therefore, we should use the absolute loss function in our objective, if we would like to model a laplacian distribution of noise.

### cauchy regression

from above, assume the cauchy distribution of noise.

the likelihood of training samples, our hypothesis, and experienced noise is

$$ p(S| X, h, \gamma) =
\biggl( \frac{1}{\pi \gamma} \biggr)^n \prod_{i=1}^n
\frac{1} {1 + \Bigl(
	\frac{y_i - h(x_i)}{\gamma} \Bigr)^2 }$$
taking the negative log of both sides, to turn into a minimisation problem yields,	
$$ -\ln p(S| X, h, \gamma) =
 n \ln (\pi \gamma) + \sum_{i=1}^n 
\ln
\biggl( 1 + \Bigl(
	\frac{y_i - h(x_i)}{\gamma} \Bigr)^2 \biggr) $$
Where we minimise right hand sum, which is empirical risk using Cauchy loss.
So we should use Cauchy loss in our objective, if we would like to model a cauchy distribution of noise.

### optimality criterion across surrogate loss functions

See [[loss functions and convex optimisation]], definitions.

Consider an objective function 
$$ f(h) = \frac{1}{n} \sum_{i=1}^n \ell(X_i, Y_i, h)$$
A point $h$ is optimal for $f$ if and only if it is feasible and
$$\nabla f(h)^T (h' - h) \ge 0 $$
for all feasible $h' \in \text{domain } f$ .
if the domain of $f$ is not bounded, optimality at $h$ will be achieved such that
$$ \nabla f(h) = \mathbf{0} $$
Now let
$$g(t) = f(th), t \in \mathbb{R}, h \in \mathbb{R}^d $$
We have
$$g'(t) = \nabla f(th)^Th $$
If $h$ is the minimiser of $f$, then $\nabla f(h) = \mathbf{0}$, so
$$g'(1) = 0 $$
So to minimise $f(h)$, we can find an $h$ such that $g'(1) = 0$.

consider this formulation for **least squares loss**:
$$\ell(X, Y, h) = (Y - h(X))^2$$
$$g'(1) = \frac{1}{n}\sum_{i=1}^n
2(y_i - h(x_i))(-h(x_i))$$
or consider this formulation for **absolute loss**:
$$\ell(X, Y, h) = | Y - h(X)| $$
$$g'(1) =  \frac{1}{n}\sum_{i=1}^n 
\frac{1} {|y_i - h(x_i)|}(y_i -h (x_i))
(-h(x_i))$$
note we define the derivative of $|x|$ to be any real value in $[-1, 1]$.

consider $g'(1)$ for **cauchy loss**:
$$ \ell(X, Y, h) = \log_2 \biggl( 
1 + \Bigl( \frac{Y-h(x)} {\sigma} \Bigr)^2
\biggr) $$
$$g'(1) = \frac{1}{n} \sum_{i=1}^n \frac{2}
{\gamma ^2 + (y_i - h(x_i))^2} 
(y_i -h (x_i)) (-h(x_i))$$

consider $g'(1)$ for **correntropy (welsch) loss**:
$$\ell(X, Y, h) = \log_2 \biggl[ 
1 - \exp \biggl( -\Bigl( \frac{Y-h(x)} {\sigma} \Bigr)^2 \biggr)
\biggr]  $$
$$g'(1) = \frac{1}{n} \sum_{i=1}^n \frac{2}
{\sigma ^2 \exp \Bigl( \frac{y_i - h(x_i)} {\sigma} \Bigr)^2} 
(y_i -h (x_i)) (-h(x_i)) $$

Notice all these $g'(1)$ are the same sums, with different terms in front of $(y_i -h (x_i)) (-h(x_i))$.

A surrogate loss function is more robust, if it assigns smaller weights to each 'loss term', given by $(y_i -h (x_i)) (-h(x_i))$, as error ($y_i - h(x_i)$) in that loss term increases. 

i.e. above, ranked robustness to noise is
correntropy > cauchy > absolute > least squares

If we are skeptical of quality of data, we should choose a surrogate loss that does not heavily penalise error in optimisation due to outliers.

## robustness in NMF

See [[dictionary learning and NMF]].

In NMF our objective is given by
$$\arg\min_{D \in \mathcal{D}, R \in \mathcal{R}} = 
\sum_{i=1}^n \ell (X_{:,i} - DR_{:,i})$$
If we use $\ell_2$ or $\ell_1$ norm, we increasingly penalise outliers. 
truncated cauchy NMF reduces weight on loss terms which will have large error; therefore is more robust than the above variants.