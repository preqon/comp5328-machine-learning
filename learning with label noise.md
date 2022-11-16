# learning with label noise

*go through to make consistent that $D$ is a domain, not distribution, which is given by $P_D$*

labelling is difficult in big data sets, and is often automated, introducing label noise.

even in small data sets
- labels can be provided by non-experts
- labels can be subjective (image captions)
- labelling can be time consuming and costly

e.g up to 20% corruption in WebVision and JFT-300M

## notation

- observation: $X \in \mathcal{X} \in R^d$
- clean but unobservable label: $Y \in \mathcal{Y}^c \in \{-1, +1 \}^c$
	- clean label, for one class: $y \in \mathcal{Y} \in \{-1, +1 \}$ 
- observable but noisy label: $\tilde{Y} \in \mathcal{Y}^c$
	- noisy label, for one class: $\tilde{y} \in \mathcal{Y}$
- clean distribution: $D(X, Y)$
- noisy distribution: $D_\rho(X, \tilde{Y})$
- flip rate: $\rho \in [0, 1]$ 

note to self, slides don't generalise to multi-class scenario until later. to be consistent, i'm going to think of $Y$ as a one-hot vector with dimension $c$, and will use $y$ to denote only one of the labels in this vector.
- in a binary classification scenario, i only need $y$ to describe a label.
$X$ is a *single* observation with dimension $d$.

**problem**:

Given the training samples 
$$\{ (X_i, \tilde{Y}_i) \}_{1 \le i \le n} \sim D_\rho(X, \tilde{Y})^n$$
We would like to find a discriminant function 
$$f_n : \mathcal{X} \mapsto \mathcal{Y}^c$$ such that the function predicts the correct label $Y$ given observation $X$.

## modelling label noise

consider the flip rate of some observation's label (the chance that this observation's label will be incorrect) probabilistically

$$\rho_{Y}(X) = P(\tilde{Y} | Y, X)$$
- note $\tilde{Y}$ is being (ab)used here to mean noisy **and incorrect**.

we denote the positive flip rate, as the probability that a label *for one class* is flipped from positive to negative.
$$\rho_{+1}(X) = P(\tilde{y} = -1 | y = 1, X)$$
we denote the negative flip rate, as the probability that a label *for one class* is flipped from negative to positive.
$$\rho_{-1}(X) = P(\tilde{y} = 1 | y = -1, X)$$
we can simplify our probabilistic consideration in two ways.

1. **random classification noise**:
$$\rho_{Y}(X) = \eta \quad \text{for } 0 \le \eta < 1/2 $$
2. **class dependent noise**:
$$\rho_Y(X) = P(\tilde{Y} | Y) $$

with no simplification, the model is called **instance and label dependent noise**.
$$\rho_{Y}(X) = P(\tilde{Y} | Y, X)$$

## learning under random classification noise

**convex potential**:
some potential loss function $$\ell(X, Y, h) \mapsto \phi(Yh(X))$$
where
- $\phi : \mathbb{R} \mapsto \mathbb{R}$ 
- $\phi$ is convex, non-increasing ($\phi(+ \infty) \to 0$), differentiable with $\phi ' (0) < 0$

**linear function class**:
$$\mathcal{F}_{lin} = 
\{ \mathbf{x} \mapsto \mathbf{\omega}^T \mathbf{x} : \mathbf{\omega} \in \mathbb{R}^d  \} $$

learning under random classification noise means that 
minimisation of any convex potential from a linear function class can result in a classification performance equivalent to random guessing.

note that when $\eta = 1/2$, the classifier only sees labels that are a result of an unbiased coin flip, and the target classifier is no longer PAC-learnable.

when $0 \le \eta < 1/2$, the target classifier is PAC-learnable, and empirical risk minimisation can be used as usual to learn.

but how do we ensure robustness to random classification noise, so that classification performance is not necessarily closer to random guessing?

**symmetric loss functions** are robust to RCN, when drawn from the universal function class (not just linear).
i.e. loss functions that satisfy,
$$ L(f(x), +1) + L(f(x), -1) = C $$
where $C$ is constant, the following is true
$$\arg \min_f R_{D,L}(f) = \arg \min_f R_{D_\rho, L}(f) $$
i.e. the expected risk will converge equivalently under a clean distribution or RCN distribution.
this is shown by the proof below.

### proof that expected risk with symmetric loss, under RCN distribution, is a linear transformation of expected risk, under clean distribution

we want to prove that expected risk under an RCN distribution, with symmetric loss $L$

$$R_{D_\rho,L}(f) =
\mathbb{E}_{(X, \tilde{Y}) \sim D_\rho} 
[ L(f(X), \tilde{Y}) ]$$
is equal to
$$ = (1 - 2\rho)R_{D,L}(f) + \rho C $$
first consider the probability of a noisy label *for one class* of some observation being +1.
$$ P(\tilde{y} = 1 | X)$$
$$= P(\tilde{y} = 1, y = 1 | X) + P(\tilde{y} = 1, y = -1 | X) $$
from probability product rule* we have
$$ = P(\tilde{y} = 1 | y = 1, X)P(y=1 | X)
+ P(\tilde{y} = 1 | y = -1, X) P(y = -1 |X)$$
from the definitions of positive and negative flip rates we have
$$ = (1 - \rho_{+1}(X)) P(y = 1|X) +
(\rho_{-1}(X)) P(y = -1 | X)$$
then using $P(y = - 1|X) = 1 - P(y=1|X)$ we have
$$ = (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=1 | X) + 
\rho_{-1}(X)$$
which is starting to look pretty epic, since we have the probability of a noisy label in terms of flip rates and the probability of a clean label.

you can follow the same line of reasoning to reach
$$P(\tilde{y} = -1 | X)$$
$$= (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=-1 | X) + 
\rho_{+1}(X)$$
i.e. equivalent expression but for the probability of a noisy label *for one class* of some observation being -1.

now, under RCN, positive and negative flip rates are the same, determined by the random rate $\eta$. we'll use the notation for flip rate in general $\rho$, to mean the random flip rate $\eta$, for this proof for RCN.
$$\rho_{+1} = \rho_{-1} = \eta = \rho $$
this gives us the two expressions
$$P(\tilde{y} = 1 | X) = (1 - 2\rho)P(y=1 | X) + \rho$$
$$P(\tilde{y} = -1 | X) = (1 - 2\rho)P(y=-1 | X) + \rho$$

We now construct our expression for the expected risk, when using a symmetric loss function, under an RCN distribution

$$R_{D_\rho,L}(f) =
\mathbb{E}_{(X, \tilde{Y}) \sim D_\rho} 
[ L(f(X), \tilde{Y}) ]$$
using definition of expected value, in a binary classification task, we have
$$ = \int \biggl( 
\Bigl( P(\tilde{y}=1,X) \Bigr) \Bigl(  L(f(X), 1) \Bigr) +
\Bigl( P(\tilde{y}=-1 ,X) \Bigr) \Bigl( L(f(x), -1) \Bigr)
\biggr)
\thinspace dX$$
then use probability product rule
$$ = \int \biggl( 
\Bigl( P(\tilde{y}=1|X)P(X) \Bigr) \Bigl(  L(f(X), 1) \Bigr) +
\Bigl( P(\tilde{y}=-1|X)P(X) \Bigr) \Bigl( L(f(x), -1) \Bigr)
\biggr)
\thinspace dX $$
Now use our respective expressions for $P(\tilde{Y}=1|X)$ and $P(\tilde{Y}=-1|X)$
$$ = \int \biggl( 
\Bigl( \bigl( (1 - 2\rho)P(y=1 | X) + \rho \bigr)
P(X) \Bigr) \Bigl(  L(f(X), 1) \Bigr)$$
$$ +
\Bigl( \bigl( (1 - 2\rho)P(y=-1 | X) + \rho \bigr)
P(X) \Bigr) \Bigl( L(f(x), -1) \Bigr)
\biggr)
\thinspace dX $$
then returning to joint probability with product rule
$$ = (1 - 2\rho) \int \biggl( P(y=1,X) L(f(X), 1) +
P(y=-1,X)L(f(X), -1) \biggr) \thinspace dX$$
$$ + \rho \int \biggl( 
P(X) \Bigl[L(f(X), 1) + L(f(X), -1) \Bigr]
\biggr) \thinspace dX$$
then using definition of expected value (but for clean distribution), and noticing that the second integral is independent of any noisy or clean label, and therefore constant when integrating with respect to $X$ we have
$$R_{D_\rho,L}(f) = 
( 1 - 2 \rho) \mathbb{E}_{(X,Y) \sim D} [L(f(X), Y)] + \rho C $$
which gives the expected risk under RCN in terms of expected risk under clean distribution.
$$ = (1 - 2 \rho)R_{D, L}(f) + \rho C $$
since this is a linear transformation, the expected risk will converge equivalently during optimisation, under a clean distribution or RCN distribution

---

\*more of an extension of the usual rule $P(a, b) = P(a|b)P(b)$
$$P(a, b | c) = p(a | b, c)p(b|c) $$

---
### sample symmetric loss functions

- 0-1 loss
- unhinged loss
$$L(f(X), Y) = 1 - Yf(X)$$
- sigmoid loss
$$L(f(X), Y) = 1 / (1 + e^{Yf(X)})$$
- ramp loss
$$L(f(X), Y) = \frac{1}{2} \max \Bigl( 0,
\min \bigl( 2, 1 - Yf(X) \bigr) \Bigr) $$

## learning under class dependent noise

we try to find a modification of a given loss function $L$ to $\tilde{L}$, to learn under class-dependent noise, such that
$$ \arg \min_{f \in \mathcal{F}} R_{D,L}(f) =
\arg \min_{f \in \mathcal{F}} R_{D_\rho,\tilde{L}}(f)$$
we use **importance reweighting**. (unbiased estimator, cost-sensitive loss, rank pruning are alternative methods)

consider two domains, $D$ produces clean data, $D_\rho$ produces noisy data

$$R_{D,L}(f) = \mathbb{E}_{(X,Y) \sim D} [L(f(X), Y)] $$
expected value is from the integral of probability distribution of this clean domain
$$ = \int P_D(X,Y) L(f(X), Y)\thinspace dXdY $$
now we add in a term for the probability distribution of the noisy domain, while maintaining equality
$$ = \int P_{D_\rho}(X,Y) \frac{P_D(X,Y)} {P_{D_\rho}(X,Y)} 
L(f(X), Y)\thinspace dXdY$$
using definition of expected value, but for noisy domain
$$ = \mathbb{E}_{(X, \tilde{Y}) \sim D_\rho}[\beta(X,Y) L(f(X), Y)] $$
where $\beta = P_D(X,Y) / P_{D_\rho}(X,Y)$ 

note $\beta$ can be written in terms of the conditional probabilities ($P(X)$ terms cancel out)
$$\beta = P_D(Y|X) / P_{D_\rho}(Y|X) $$

From previous proof, **in the binary classification scenario**, we have the expressions
$$P(\tilde{y} = 1 |X) = (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=1 | X) + 
\rho_{-1}(X) $$ 
$$ P(\tilde{y} = -1 |X) = (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=-1 | X) + 
\rho_{+1}(X)$$
we can rearrange these to express the probability of a clean label in terms of its noisy label
$$ P(y=1|X) = \frac{P(\tilde{y} = 1|X) - \rho_{-1}(X)}
{1 - \rho_{+1}(X) - \rho_{-1}(X)}$$
$$ P(y=-1|X) = \frac{P(\tilde{y} = -1|X) - \rho_{+1}(X)}
{1 - \rho_{+1}(X) - \rho_{-1}(X)}$$
this gives us these two expressions for $\beta$
$$\beta(X, y=1) = \frac{P(\tilde{y}=1|X) - \rho_{-1}(X)}
{(1 - \rho_{+1}(X) - \rho_{-1}(X))P(\tilde{y}=1|X)}$$ $$\beta(X, y=-1) = \frac{P(\tilde{y}=-1|X) - \rho_{+1}(X)}
{(1 - \rho_{+1}(X) - \rho_{-1}(X))P(\tilde{y}=-1|X)}$$$\beta$ is looking pretty epic to estimate, because its in terms of noisy posteriors. but we need a way to estimate the positive and negative flip rates.

we will assume that our class-dependent flip rates are small such that
$$\rho_{+1} + \rho_{-1} \le 1 $$
from the above two expressions for $P(\tilde{y} = 1 |X)$ and $P(\tilde{y} = -1 |X)$, we can therefore deduce
$$P(\tilde{y} = 1 |X) \ge \rho_{-1}(X)$$
$$P(\tilde{y} = -1 |X) \ge \rho_{+1}(X)$$
(when $\rho_{+1} + \rho_{-1} = 1$, the above inequalities turn into equalities)

now make another deduction from the two expressions for $P(\tilde{y} = 1 |X)$ and $P(\tilde{y} = -1 |X)$

> what if $P(y=1|X) = 0$ or $P(y=-1|X) = 0$, i.e. we become oracles for the clean unobservable label.

that would be pretty epic because it would give us expressions for positive and negative flip rates in terms of noisy posterior
$$\rho_{-1} = P(\tilde{y}=1|X)$$
$$\rho_{+1} = P(\tilde{y}= -1|X)$$
> but we're not oracles. BUT WE'RE STILL DJs!!!

so we estimate positive and negative flip rates as follows
$$ \rho_{-1} = \min_{X \in \mathcal{X}} P(\tilde{y} = 1|X) $$
$$ \rho_{+1} = \min_{X \in \mathcal{X}} P(\tilde{y} = -1|X)  $$
two ways to reason why this works
1. From the inequalties, we can see how a lower posterior approaches the value of its respective flip rate
2. Some $X$ in $\mathcal{X}$ is probably 'obviously' its ground truth, regardless of its noisy label. a classifier of reasonable complexity will find this, and return this sample in the above minimisation. if the classifier is correct with respect to ground truth, then indeed $P(y=1|X) = 0$ or $P(y=-1|X) = 0$, and we have a good estimation.

> consider that the estimation of negative flip rate is saying, 'find me the sample that is least likely to be positive'. if our bet pays off, then this is also when $P(y=1|X) = 0$. and so we have $\rho_{-1} = P(\tilde{y}=1|X_{min})$ 

we can now estimate $\beta$, and we have an expression for the expected risk, under clean label distribution, in terms of expected risk under noisy label distribution.

$$R_{D,L}(f) = \mathbb{E}_{(X, \tilde{Y}) \sim D_\rho}[\beta(X,Y) L(f(X), Y)] $$
we can use empirical risk minimisation of the latter, to approximate the former.

(the modification to loss function would be written as follows)

$$\tilde{L}(f(X), Y) = \beta(X,Y) L(f(X), Y) $$

### extending to multi-class classification

we use the short hand $Y = \{ 1, 2 \dots c \}$ to be the $c$-dimensional one hot vectors $Y = \{(1, 0 \dots, 0), (0, 1 \dots, 0) \dots (0, 0 \dots 1)\}$

from probability product and sum rules, and from the assumption in class-dependent noise that $P(\tilde{Y}|Y) = P(\tilde{Y}|Y,X)$, the following matrix product is true.

$$\left[ 
\matrix{P(\tilde{Y} = 1 |X) \\ \vdots \\ P(\tilde{Y} = C|X) }
\right] =
\left[
\matrix{
P(\tilde{Y}=1|Y=1, X) & \dots & P(\tilde{Y} = 1| Y=C, X) \\
\vdots & \ddots & \vdots \\
P(\tilde{Y}=C|Y=1, X) & \dots & P(\tilde{Y} = C| Y=C, X) 
}
\right] \left[
\matrix{P(Y=1|X) \\ \vdots \\ P(Y=C|X)}
\right]
$$
the inner matrix is called the transition matrix, $T$.

short hand for the above: 
$$P(\forall \tilde{Y}|X)^T = T P(\forall Y|X)^T \quad (\text{forward})$$  
we also have, from the inverse of the above
$$P(\forall Y|X)^T = T^{-1} P(\forall\tilde{Y}|X)^T \quad \text{(backward)}$$
($T$ is known for the following two methods)

**forward learning**:
suppose we have a classifier that is meant to estimate $P(\forall Y|X)^T$. 
however, under label noise, such a classifier predicts $P(\forall \tilde{Y} |X)^T$.
we ensure that the classifier is trained to instead target $P(\forall Y|X)^T$, by adjusting its loss function to measure loss between $TP(\forall \tilde{Y} |X)^T$ and $P(\forall \tilde{Y} |X)^T$.

**backward learning**:
here, our classifier is trained to estimate $P(\forall \tilde{Y} |X)^T$, (as it naturally would, under label noise).
After training, we apply the inverse $T$ to estimate clean labels i.e. simply perform (backward).

for proofs each of these methods are robust to label noise, see (Patrini et al., 2017)

**transition matrix estimation**:
use anchor points, i.e. similar underlying argument as seen above in the binary case, we find samples for the $i$th class, for which it is most likely that
$$P(Y=i|X) = 1 $$
The $i$th column of the transition matrix is given by $P(\forall \tilde{Y}|X_i)$ 
where $X_i$ is the anchor point for the $i$th class.

to find the sample for which it is most likely that $P(Y=i|X) = 1$, we simply do
$$\max_{X \in \mathcal{X}} P(\tilde{Y} = i|X) $$
for proof this estimation converges to the true value of $T$, see (Patrini et al., 2017)

## learning under instance- and class- dependent label noise

as previous, we have the expressions
$$P(\tilde{y} = 1 |X) = (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=1 | X) + 
\rho_{-1}(X) $$

$$ P(\tilde{y} = -1 |X) = (1 - \rho_{+1}(X) - \rho_{-1}(X)) 
P(y=-1 | X) + 
\rho_{+1}(X)$$
estimating flip rate here is ill-posed. e.g. there are infinite solutions to positive and negative flip rates for some value of $P(\tilde y = 1|X)$.

an open question is, what are some reasonable assumptions we can make about these flip rates, such that estimating them is viable?
