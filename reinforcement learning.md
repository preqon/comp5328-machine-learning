# reinforcement learning

given an agent that interacts with an environment, by making actions based on policies, where each action, once made, is given some reward, we would like to maximise the reward the agent receives.

## notation

- action: $A \in \mathcal{A} = \{a_0 \dots a_{n_A} \}$
- state: $S \in \mathcal{S} = \{s_0 \dots s_{n_S} \}$
- time step: $t$
- reward: $R \in \mathcal{R} = \{r_0, \dots, r_{n_R}\}$ 
- discount factor: $\gamma$
- policy to choose an action: $\pi \in \Pi = \{ \pi_0, \dots, \pi_{n_\Pi} \}$ 

## problem

At each time step, agent is in some state $S_t = s_t$. It chooses an action $A_t = a_t$, based on a policy $\pi$. Dependent on $s_t$ and $a_t$, the agent receives reward $R_t = r_t$, and goes into state $S_{t+1} = s_{t+1}$.

an agent's policy is constructed probabilistically - what is the probability each action will be chosen, given the current state? this is written as
$$\pi(A_t | S_t = s_t)$$
we will also consider each possible next state as a probability distribution
$$ P(S_{t+1}| S_t = s_t, A_t = a_t)$$
we consider each possible reward, given the current state and action, as a probability distribution
$$P(R_t | S_t = s_t, A_t = a_t) $$
before calculating accumulated award, we choose a discount factor $0 < \gamma < 1$. 
this arises from the notion that with increasing time steps, future reward becomes less valuable to the current agent.

we want to find the optimal policy $\pi^*$ such that the accumulated reward is maximised
$$\pi^* = \arg \max_{\pi \in \Pi} \mathbb{E} \left[ 
\sum_{t \ge 0} \gamma^t r_t \right] : S_0 = s_0 $$
with
$$r_t \sim P(R_t | S_t = s_t, A_t = a_t), \quad
a_t \sim \pi(A_t | S_t = s_t), \quad 
\text{and} \quad s_{t+1} \sim P(S_{t+1}| S_t = s_t, A_t = a_t)$$

## q-value function

the function $Q : \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$ finds the expected value of accumulated reward, given an initial state, an initial action, and a policy.

$$Q(s_0, a_0) = \mathbb{E} \left[ 
\sum_{t \ge 0} \gamma^t r_t \right] : S_0 = s_0, A_0 = a_0, \Pi_Q = \pi $$
with
$$r_t \sim P(R_t | S_t = s_t, A_t = a_t), \quad
a_t \sim \pi(A_t | S_t = s_t), \quad 
\text{and} \quad s_{t+1} \sim P(S_{t+1}| S_t = s_t, A_t = a_t)$$

the **optimal q-value function** $Q^* : \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$ finds the maximum expected value of accumulated reward, given an initial state, an initial action, over all policies.
$$Q^*(s_0, a_0) = \max_{\pi \in \Pi} \mathbb{E} \left[ 
\sum_{t \ge 0} \gamma^t r_t \right] : S_0 = s_0, A_0 = a_0 $$
with
$$r_t \sim P(R_t | S_t = s_t, A_t = a_t), \quad
a_t \sim \pi(A_t | S_t = s_t), \quad 
\text{and} \quad s_{t+1} \sim P(S_{t+1}| S_t = s_t, A_t = a_t)$$

for each state $s$, given a current action $a$, and a current reward $r$, the accumulated reward *after the next optimal action* can be written as follows. 
$$Q^*(s, a) = r + \gamma \mathbb{E}_{s'} \left[ 
\max_{a'} Q^*(s', a')
\right] $$
with
$$s' \sim P(S' | S = s, A = a) $$
where $a'$ is a next possible action.
the form of the above is called a **bellman equation**.

## q-learning

consider, in an environment that returns immediate rewards (we do not have oracle knowledge of future rewards), we can update an estimation of the optimal q-value function for a given state-action pair, $\hat{Q}$, iteratively after each state change, inspired by the bellman equation form.

for current state $s$, current action $a$, current reward $r$ (observed due to $(s,a)$), we update the estimation following
$$\hat{Q}_{i+1}(s, a) = \hat Q_i(s,a) + \eta \biggl( 
r + \lambda \max_{a'}\hat Q_i(s', a') - \hat Q_i(s,a)
\biggr)$$
with
$$s' \sim P(S' | S = s, A = a) $$
where $a'$ is a next possible action, and $0 < \eta < 1$ is a learning rate hyperparameter.

i.e., in each iteration $\hat Q (s,a)$ is updated by adding the (scaled) summation of
- difference between true current reward and estimated current reward
- discounted reward due to next estimated optimal action

it can be proven that, with infinite iterations, the above formulation approaches $Q^*$

**q-table**:
used to store q-values for state-action pairs. at each iteration, an entry in this table is updated.

**q-learning** is thus as follows

> Initialise Q(s,a) for all state-action pairs arbitrarily
> for each agent episode do
> 	initialise s
> 	for s is not a terminal state do
> 		$a \leftarrow \arg \max_a Q(s,a)$
> 		perform $a$ and observe $(r, s')$
> 		$Q(s,a) = Q(s,a) + \eta(r + \gamma \max_{a'}Q(s', a') - Q(s,a))$
> 		$s \leftarrow s'$

an alternative algorithm called **sarsa** is given below

> Initialise Q(s,a) for all state-action pairs arbitrarily
> for each agent episode do
> 		initialise s
> 		$a \leftarrow \arg \max_a Q(s,a)$
> 		for s is not a terminal state do
> 				perform $a$ and observe $(r, s')$
> 				$a' \leftarrow \arg \max_{a'} Q(s',a')$
> 				$Q(s,a) = Q(s,a) + \eta(r + \gamma Q(s', a') - Q(s,a))$
> 				$s \leftarrow s'$
> 				$a \leftarrow a'$ 


the difference
- in q-learning, the action in the current iteration may not be the estimated optimal action from the previous iteration.
	- we choose an action to update the estimated q-value, but do not necessarily perform that action
- in sarsa, the action in the current iteration *is* the estimated optimal action from the previous iteration.
	- we choose an action to update the estimated q-value, and perform that action.

**target policy**: the policy used, when choosing from possible actions, to update q-values.
**behaviour policy**: the policy used, when choosing from possible actions, to update the agent's action.

**on-policy** algorithm: an RL algorithm that has target policy consistent with behaviour policy.
**off-policy algorithm**: an RL algorithm that does not have target policy consistent with behaviour policy.

q-learning is an off-policy algorithm.
sarsa is an on-policy algorithm.

## deep q-learning network

this is where we do a little gaming. we get a little freaky :p

consider that states and actions can be continuously distributed.

it becomes infeasible to store $Q(s, a)$ in a table.

we can use a neural network to approximate the storage of $Q(s,a)$, i.e. parameterise $Q(\cdot ,\cdot)$ with weights $W$. 

objective function, for the current state $s$, and current action $a$, is given by
$$( r + \gamma \max_{a'} Q_W(s', a') - Q_W(s,a))^2$$
with
$$s' \sim P(S' | S = s, A = a) $$

**experience replay**:
samples used to optimise the objective function are drawn randomly from a database storing generated experiences 
- where an experience is $(s, a, r, s')$

this is better than just iterating through consecutive states, since doing so can lead to overfitting to some trajectories.

**prioritised experience replay**:
as above, but samples are weighted, such that higher weights are drawn more often.
larger weights are chosen for samples that would lead to larger updates during optimisation of the objective (i.e. lead to greater loss).

the algorithm for deep q-learning with prioritised experience replay is given below.

> Initialise $Q(s,a)$ for all state-action pairs arbitrarily.
> Initialise experience bank $B \leftarrow \emptyset$ 
> Observe $s_0$, choose arbitrary $a_0$
> for $t=1$ to $T$ do
> 		 $a \leftarrow \arg \max_a Q_W(s_{i-1},a)$ 
> 		 Observe $(s_t, r_t, \gamma_t)$
> 		 store $(s_{t-1}, a_{t-1}, r_t, \gamma_t, s_t, a)$ in $B$ with maximal priority
> 		 every few iterations do
> 				 draw a sample ($s, a, r, \gamma, s', a')$ from $B$ according to priority
> 				  $\delta \leftarrow ( r + \gamma \max_{a'} Q_W(s', a') - Q_W(s,a))^2$
> 				 decrease this sample's priority by $|\delta|$
> 				 update weights $W \leftarrow W = \eta \nabla ( r + \gamma \max_{a'} Q_W(s', a') - Q_W(s,a))^2$ 
> 		end
> end


## policy gradient ascent

ok, this is pretty epic.

we use a neural network with weights $\theta$ to parametrise a policy $\pi$, and thus learn the best policy - which, in certain contexts, will require less complexity than learning the best $(s,a)$ pairs.

we write the expected reward using policy parametrised by $\theta$ as

$$J(\theta) = \mathbb{E} \left[ \sum_{t \ge 0} \gamma^t r_t
\right] : \Pi_J = \pi_\theta $$
so we want to optimise the following objective
$$\theta^* = \arg \max_\theta J(\theta) $$
let $r(\tau)$ be the reward of a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$
the probability distribution of trajectories when following $\theta$ is given by $P(\tau;\theta)$.[^1]

we can write $J(\theta)$ in terms of $r(\tau)$ as
$$J(\theta) = \mathbb{E}_{\tau \sim P(\tau;\theta)}[r(\tau)] $$

$$ = \int_{\tau \sim P(\tau;\theta)} r(\tau)P(\tau;\theta) \thinspace d\tau $$
this lets us calculate the gradient of $J(\theta)$ as follows
$$ \nabla J(\theta) =
\int_{\tau \sim P(\tau;\theta)} r(\tau) \nabla_{\theta}P(\tau;\theta) 
\thinspace d\tau $$
i.e. by finding the gradient of probability distribution $P(\tau ; \theta)$, and integrating rewards over all trajectories.

now, we **cannot** approximate the above integral with empirical mean, using 'enough' sample trajectories, e.g. by
$$\nabla \hat J(\theta) = \frac{1}{n}\sum^n_{i=1}
r(\tau_i) \nabla_\theta P(\tau_i; \theta)$$
this is because the integral **is not an expected value**, since neither term (reward, gradient) is the probability distribution being drawn from when integrating.

we do a little gaming with $\nabla J(\theta)$. add in another $P(\tau;\theta)$ term while maintaining equality.
$$ \nabla J(\theta) =
\int_{\tau \sim P(\tau;\theta)} r(\tau) P(\tau;\theta)
\frac{\nabla_{\theta}P(\tau;\theta)}{P(\tau;\theta)}
\thinspace d\tau  $$
now the gradient term can be written as gradient of a logarithm
$$ =
\int_{\tau \sim P(\tau;\theta)} r(\tau) P(\tau;\theta)
\nabla_{\theta}\log{P(\tau;\theta)}
\thinspace d\tau  $$
and our integral is an expected value by defnition
$$ = \mathbb{E}\left[ r(\tau) \nabla_\theta \log P(\tau; \theta) \right] $$
so we use empirical mean to approximate as follows
$$\nabla \hat J(\theta) = \frac{1}{n}\sum^n_{i=1}
r(\tau_i) \nabla_\theta \log P(\tau_i; \theta) $$
now how do we find the probability distribution of trajectories following $\theta$?
first, assume Markov condition holds (that previous trajectory always leads to current state) i.e.

$$P(s_{t+1}, a_t | s_0, s_1, \dots, s_t) = P(s_{t+1}, a_t | s_t) $$
then we have
$$ P(\tau; \theta) = \prod_{t \ge 0}P(s_{t+1} | s_t, a_t)\pi_\theta
(a_t| s_t)$$
and so we write our gradient term as
$$\nabla_\theta \log P(\tau_i; \theta) =
\nabla_\theta \sum_{t \ge 0} \log P(s_{t+1} | s_t, a_t) + \log\pi_\theta
(a_t| s_t)$$
we can find the gradients of left and right terms inside the sum, and sum those.
however,  $\nabla_\theta \log P(s_{t+1} | s_t, a_t) = 0$, since it is a gradient with respect to $\theta$ of a function without $\theta$. 
so we are left with
$$\nabla_\theta \log P(\tau_i; \theta) = \sum_{t \ge 0} \nabla_\theta \log\pi_\theta
(a_t| s_t) $$
which we substitute into our estimator $\hat J (\theta)$
$$ \nabla \hat J(\theta) = \frac{1}{n}\sum^n_{i=1} \sum_{t \ge 0}
r(\tau_i) \nabla_\theta \log\pi_\theta (a_t| s_t) $$
giving us a way to estimate the gradient with respect to $\theta$ of the expected reward using policy parametrised by $\theta$.

we perform gradient ascent to learn $\theta$ and find the best policy to maximise reward.

the algorithm for policy gradient ascent is given below.

> initialise $\theta$ arbitrarily
> for each episode $(s_0, a_0, r_0 \dots s_T, a_T, r_T) \sim \pi_\theta$ do
>		$\Delta \theta \leftarrow 0$
>		$r \leftarrow 0$
>		for $t = 0$ to $T$ do
>				$\Delta \theta \leftarrow \Delta \theta + \alpha \nabla_\theta \log\pi_\theta (a_t| s_t)$ 
>				 $r \leftarrow r + \gamma r_t$
>		end
>		$\theta \leftarrow \theta + \eta \Delta \theta$
> end

[^1]: P(x;y) means probability of $x$, where $x$ is a function with parameter $y$.