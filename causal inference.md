# causal inference

## overview

it's very hard to derive a causal graph (i.e., a model of causal structure) from observational data alone. this is called **causal discovery**. Consider that ML finds $P(Y | X)$, but this is entirely a measure of correlation, which may or may not contain causation.

usually we build such a graph modelling causal relationships using reasonable assumptions/heuristics.

then we can start to answer 'what if' questions about varying causal factors, i.e. produce estimations of the scale of causal effect from these 'what if' scenarios, from observational data. this is called **causal inference**.

assuming markov conditions to hold, lets us transform a graph into an expression for the joint probability distribution of all the graph variables, e.g. $P(X, Y, Z$). applying such a 'markov decomposition' risks less redundant modelling of $P(X, Y, Z)$, compared to applying probability chain rule on all variables, as you would when making no assumptions.

local markov conditions and global markov conditions are different only in terms of language. they are both just saying how markov conditions would be applied to the graph.

## correlation is not causation

a hidden driving factor can cause two of its targets to be correlated.

a driver's conditional dependence on a third factor, can cause the third factor to be correlated with driver's targets.

## causal graphs

we model causal relationships using a DAG

![[DAG example.png|250]]

**Markov condition**: if causal relationships as drawn in the causal graph are true, then statistical independence is known, satisfying

> every variable $X_i$, when conditioned on its own parents, is independent of its non descendant $Y$. $$ (X_i \perp\!\!\!\perp Y | \text{parents}(X_i) )$$

[^1] when we assume the markov condition is true, we can model the joint probability distribution of all variables, as the product of independent conditional probabilities.

in the above example we have
$$P(X_1, X_2, X_3, X_4, X_5) 
= P(X_1)P(X_2|X_1)P(X_3|X_1)P(X_4|X_2, X_3)P(X_5|X_4) $$
consider that this is a simplification compared to applying the probability chain rule, without any assumptions, as follows 

$$ P(X_1, \dots , X_5)
= P(X_1 | X_2 \dots X_5)P(X_2|X_3, X_4, X5)P(X_3| X_4, X5)P(X_4|X_5)P(X_5) $$

the above formulation of the markov condition, for a variable $X_i$, is called a **local markov condition**. we can formulate the same idea as a **global markov condition**.

> for three disjoint sets of variables $\mathbf{X, Y, S}$ the set $\mathbf{X}$ is d-separated from $\mathbf{Y}$, conditional on $\mathbf{S}$ iff all paths between any member of $\mathbf{X}$ and any member of $\mathbf{Y}$ are blocked by $\mathbf{S}$.

> a path $q$ is blocked by $\mathbf{S}$ if 
> 1. $q$ contains a chain $i \to m \to j$ or a fork $i \leftarrow m \to j$ where $m$ is in $\mathbf{S}$
> 2. $q$ contains a collider $i \to m \leftarrow j$ such that $m$ is **not** in $\mathbf{S}$, and **no descendant** of $m$ is in $\mathbf{S}$

We use this notation to say $\mathbf{S}$ d-separates $\mathbf{X}$ and $\mathbf{Y}$.
$$ (\mathbf{X} \perp\!\!\!\perp \mathbf{Y} | \mathbf{S}) $$
[^2]which again, lets us model the probability distribution of all variables, since we assume d-separation to mean that the analagous independence is statistically true.

[^1]: I find this notation a little strange. I would write $( (X_i | \text{parents}(X_i)) \perp\!\!\!\perp Y)$ 
[^2]: And in the global case I would write $( (\mathbf{X} | \mathbf{S}) \perp\!\!\!\perp \mathbf{Y})$ 

## causal faithfulness assumption

consider that the independence observed (using markov conditions) in a graph may not actually be statistically true.

we denote the statistical independence observed in a probability distribution $p$ and the independence observed in a graph $G$
$$ (\mathbf{X} \perp\!\!\!\perp \mathbf{Y} | \mathbf{S})_p \quad \text{and} \quad
(\mathbf{X} \perp\!\!\!\perp \mathbf{Y} | \mathbf{S})_G$$
respectively.

when there are no additional independent conditional probability terms in the true probability distribution, compared to the graph, i.e.

$$(\mathbf{X} \perp\!\!\!\perp \mathbf{Y} | \mathbf{S})_p \Rightarrow (\mathbf{X} \perp\!\!\!\perp \mathbf{Y} | \mathbf{S})_G$$
then the distribution $p$ is **faithful** to $G$.

the assumption that markov conditions hold, is called the **causal faithfulness assumption**.

## causal bayesian network

when we add observations to the simple DAG causal graph, we have a causal bayesian net.

e.g.
![[bayesian network example.png|500]]

some natural 'what if' questions arise as follows.

1. **conditioning**
	e.g. what is the chance the pavement will be slippery if we find the sprinkler off?
$$P(\text{Slippery} | \text{Sprinkler} = \text{OFF}) $$
2. **intervention**
	e.g. what is the chance the pavement will be slippery if we TURN the sprinkler off?
$$P(\text{Slippery} | do(\text{Sprinkler} = \text{OFF})) $$
1. **counterfactual reasoning**
	e.g. what is the chance that the pavement would be slippery, if the sprinkler had not been on, given the sprinkler IS on and the pavement IS NOT slippery.
$$P(\text{Slippery}_{Sprinkler=OFF} | \text{Sprinkler} = \text{ON}, \text{Slippery} = False) $$

how do we answer these 'what if' questions, *from observational data*? i.e. how can we estimate these probabilities from observation?

## estimating causal effects

ideally we use randomised controlled experiments.

For example we want to find the effects of treatment $T$ on recovery $R$, i.e. we want to find $P(R|T)$. To control for all other variables $S$, we find
$$ P(R|T) = \sum_S P(R|T,S)P(S|T) $$
note the above is a conditioning question. the intervention question would be
$$ P(R|do(T)) = \sum_S P(R|T,S)P(S|T) $$
but with large number of $S$ this is infeasible.

how might we use our heuristical construction of a causal graph to aid our estimation of causal effects? we can use back-door adjustment or the front-door adjustment.

### back-door adjustment

A set of variables $\mathbf Z$ is a back-door to an ordered pair of variables $(X, Y)$ in a DAG $G$ if
- no node in $\mathbf Z$ is a descendant of $X$
- $\mathbf Z$ blocks every non-causal path from $X$ to $Y$.
	- to construct such a non-causal path, flip arrows targeting $X$, then ignore arrow directions and see if you can reach $Y$.

note 'ordered pair' means $Y$ must be a descendant of $X$.

example:
![[backdoor example.png|200]]

if $\mathbf{Z} = \{ X_3, X_4 \}$ then it is a backdoor to $(X_i, X_j)$.
if $\mathbf{Z} = \{ X_4, X_5 \}$ then it is a backdoor to $(X_i, X_j)$.
if $\mathbf{Z} = \{ X_4 \}$ then it is not a backdoor to $(X_i, X_j)$.

now we consider a fixed value of $X$, $x$, and a fixed value of $Y$, $y$. 
with some abuse of notation, we also consider a fixed combination of values *for each node* in $\mathbf{Z}$, $z$.

conditioning on all possible value combinations in the back-door $\mathbf{Z}$, lets us estimate the causal effect of some $x$ on some $y$.
$$P(y | x) = \sum_{z} P(y |x, Z)P(z) $$

### front-door adjustment

A set of variables $\mathbf Z$ is a front-door to an ordered pair of variables $(X, Y)$ in a DAG $G$ if
- $\mathbf Z$ intercepts all directed paths from $X$ to $Y$
- there is no back-door path from $X$ to $\mathbf{Z}$.
- all back-door paths from $\mathbf{Z}$ to $Y$ are blocked by $X$.

now we consider a fixed value of $X$, $x$, and a fixed value of $Y$, $y$. 
with some abuse of notation, we also consider a fixed combination of values *for each node* in $\mathbf{Z}$, $z$.

if $P(x, z) > 0$ for all values of $x$ and $z$, then we can find the causal effect of $X$ on $Y$ as 

$$P(y | x) = \sum_{z} P(z | x) \sum_{x'} P(y |x', z)P(x') $$

### final remark on causal inference

is it possible to estimate causal effects from observational data?
- if we have constructed a causal graph from reasonable assumptions, then yes, we can estimate causal effects i.e. answer our 'what if' questions.

however, can we find a causal structure, i.e. our causal graph, from observational data *alone*('causal discovery')? i.e. from empirical construction of $P(Y|X)$. This is an open question - several algorithms do exist, with little resemblance to eachother, and themselves involving some heuristics.


