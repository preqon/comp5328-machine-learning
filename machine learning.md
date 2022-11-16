# machine learning

Machine learning is the construction of a statistical model which aims to match the underlying distribution some data is drawn from.

#### formally

A machine learning algorithm is a mapping to find a hypothesis to fit the data

$$\mathcal{A} : S \in (\mathcal{X} \times \mathcal{Y})^n \mapsto h \in H $$
An algorithm $\mathcal{A}$ such that a dataset $S$ from all possible labeled data is mapped to a hypothesis $h$ from a hypothesis class $H$.

The mapping is an optimisation procedure that picks a hypothesis from a predefined hypothesis class to minimise or maximise the objective.

#### elements of supervised ML algorithms

1. input (training data) (set of examples i.e. features/label pairs)
2. predefined hypothesis class
3. [[objective function]]
4. [[optimisation method]]
5. output hypothesis
		i.e. the final 'model' or function that lets you choose a guess.

3 and 4 map an input (subset of all possible examples) to a hypothesis (element of hypothesis class).

---

some terminology

- 'regressor' = 'feature' : one of the dimensions in a sample $X$, or one of the dimensions in some description of $X$.
- **empirical risk**: empirical mean of class error.
- **expected risk**: expected value of class error.