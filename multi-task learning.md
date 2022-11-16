# multi-task learning

we would like to train a classifier such that readouts can be performed on the same classifier, for varying tasks.

we exploit relatedness between tasks. this is useful when datasets for a particular task are limited, but that task is related to others, for which large datasets are available.

when tasks are independent, there is no advantage to multi-task learning, compared to 'single-task' learning.

Notice that in MTL, while we exploit knowledge across multiple tasks, the motivation is different to transfer learning, for which, though we may draw from many source tasks, we are ultimately interested in one target task.

## notation

- set of $m$ learning tasks: $\{ T_i \}_{i=1}^m$
- dataset for task $T_i$: $\mathcal{D} = \{X_j^i, Y_j^i\}_{j=1}^{n_i}$

## problem

Given a set of $m$ learning tasks $\{ T_i \}_{i=1}^m$, each with dataset $\mathcal{D}_i$, where all tasks, or a subset of tasks, are related, we learn hypotheses for $\{ T_i \}_{i=1}^m$, improving the hypothesis for $T_i$ by using knowledge contained across the $m$ tasks.

consider each task is learned by the linear hypothesis function such that 
$$ h(x) = w^Tx $$
Given $\{ T_i \}_{i=1}^m$, we denote $w$ for the $i$th task as $w^i$. $W = \{ w^1, \dots, w^m \}$ 

we can use empirical risk minimisation across tasks, to find the optimal set of $w$, $W^*$.
$$W^* = \arg \min_{W} \frac{1}{m} \sum_{i=1}^m 
\sum_{j=1}^{n_i} \ell(X^i_j, Y^i_j, w^i)$$
How do we decide if this performs better than learning each task individually?
$$\text {do} \quad w^{i*} = \arg \min_{w^i}
\frac{1}{n_i} \sum_{j=1}^{n_i} \ell (X^i_j, Y^i_j, w^i)
\quad \text {for } i \dots m$$
We share knowledge between tasks *when there is relatedness between the tasks*. Consider that multi-task learning requires some manual selection of a set of tasks - it is possible that one task could reduce the performance for other tasks.

We share either parameters or features between tasks, using either parameter-based MTL, or feature-based MTL.

## parameter-based multi-task learning

we study 4 models. the last one is the most cracked.

----
**model 1**
we assume tasks are related by parameters such that
$$w^i = w_0 + \Delta w^i, \forall i \in\{1, \dots, m\}$$
we denote $\{\Delta w^1, \dots \Delta w^m \}$ as $\Delta W$. i.e. hypotheses have shared parameters $w_0$ with some difference.

using empirical risk minimisation across tasks, we have
$$W^* = w_0 + \Delta W^* :w_0, \Delta W^* = \arg \min_{w_0, \Delta W} \frac{1}{m} \sum_{i=1}^m 
\sum_{j=1}^{n_i} \ell(X^i_j, Y^i_j, w_0 + \Delta w^i)$$
which we want to outperform
$$\text {do} \quad w^{i*} = \arg \min_{w^i}
\frac{1}{n_i} \sum_{j=1}^{n_i} \ell (X^i_j, Y^i_j, w^i)
\quad \text {for } i \dots m$$
---
**model 2**
We can use square Frobenius norm regularisation to improve our empirical risk minimisation, such that relatedness between tasks is improved.

$$\arg \min_{w_0, \Delta W} \frac{1}{m} \sum_{i=1}^m 
\sum_{j=1}^{n_i} \ell(X^i_j, Y^i_j, w_0 + \Delta w^i)
+ \lambda ||\Delta W||_F^2$$
consider that the above regularisation minimises large differences between each $\Delta w^i$.

---
**model 3**
We can use a rank regularisation term to improve relatedness.

**rank**: of a matrix is the maximum number of linearly independent columns.

if our regularisation term is a scaled rank of $W$, then we force the parameters across hypotheses to be more related (there will be fewer linearly independent $w^i$).
$$\arg \min_{W} \frac{1}{m} \sum_{i=1}^m \frac{1}{n_i}
\sum_{j=1}^{n_i} \ell(X^i_j, Y^i_j, w^i) + \lambda \, \text{rank}(W) $$

---
**model 4**
We consider each task to be learned by a hypothesis function 
$$h(x) = w^Tx = u^Tx + v^T \Theta x^T $$
where 
- $u$ and $v$ are vectors that store the difference of this hypothesis' weights to $\Theta$
- we assume all samples have $p$ dimensions across all tasks. $\Theta$ is a $p$-dimensional parameter subspace, uniform across all hypotheses. we assume hypotheses all have $n_w$ parameters. $\Theta$ is in the form of a matrix in $\mathbb{R}^{n_w \times p}$ and has low rank.

So $\Theta x^T$ is the projection of the sample into the parameter subspace shared across all hypotheses. We then do a further transformation via $u$ and $v$, specific to this hypothesis, to arrive at $w^Tx$.

So consider each $w^i \in W$, has $u^i \in U$ and $v^i \in U$, and is given by
$$w^i = u^i + \Theta^T v^i $$
where $\Theta$ is shared across all hypotheses.

now we find optimal $U, V, \Theta$ as follows
$$\arg \min_{U, V, \Theta} \frac{1}{m} \sum_{i=1}^m \frac{1}{n_i}
\sum_{j=1}^{n_i} \ell(X^i_j, Y^i_j, u^i + \Theta^Tv^i) + \lambda
||(U)||_F^2 $$
and use these to construct $W^*$. 
- we also make the constraint that $\Theta^T \Theta = I_{n_w \times n_w}$. This ensures parameters in the shared parameter subspace are orthonormal, and therefore non-redundant/highly descriptive.
- notice we're also regularising $U$, to reduce the difference between additive transformations to $\Theta^Tv^i$ for each hypothesis.
	- i.e. another way to ensure hypotheses' weights are closer together.

## feature-based multi-task learning

each task has dataset $\mathcal{D}$ such that
$$\mathcal{D_i} = \{X_j^i, Y_j^i\}_{j=1}^{n_i} $$
we find a projection matrix $P$
$$\mathcal{D_i} = \{P^TX_j^i, Y_j^i\}_{j=1}^{n_i} $$
such that samples are projected into a feature space, where samples across tasks are more correlated.
i.e., we improve the tasks' relatedness, by improving the relatedness of datasets' feature space.

we study two models.

---
**model 1**

consider the same formulation of $W$ as previous. We learn the optimal $W$ along with optimal $P$.
$$\arg\min_{W, P} \frac{1}{m} \sum_{i=1}^m \frac{1}{n_i}
\sum_{j=1}^{n_i} \ell(P^TX^i_j, Y^i_j, w_i) + \lambda \,
\text{rank}(W) $$
with constraint $PP^T = I$. This ensures low redundancy inside the projection matrix.
- notice using rank to regularise $W$, to improve relatedness of parameters, as previous.

---
**model 2**

construct a neural network, with unique output layer for each task. that is, output layer will receive from varying hidden nodes, with varying weights, but otherwise input and hidden layers are the same.

the neural network can be considered a feature extraction procedure; output layers learn from the same features.

## feature-based and parameter-based multi-task learning

after projecting samples into a shared feature space, we can again consider that hypotheses might be related by some shared parameter.
$$w^i = w_0 + \Delta w^i, \forall i \in\{1, \dots, m\}$$
this gives us the following objective to find $P^*$ and construct $W^* = w_0 + \Delta W^*$.

$$
\arg\min_{w_0, \Delta W, P} \frac{1}{m} \sum_{i=1}^m \frac{1}{n_i}
\sum_{j=1}^{n_i} \ell(P^TX^i_j, Y^i_j, w_0 + \Delta w^i) +
\lambda ||W||_F^2$$
with $PP^T = I$.

