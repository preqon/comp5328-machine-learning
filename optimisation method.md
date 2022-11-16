# optimisation method

An optimisation method is how we optimise an [[objective function]]. In the context of classification, it is a method to search through $h \in \mathcal{H}$ to ultimately find the optimal hypothesis.
$$\arg \min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{ Y_{i} \ne sign(h(X_i)) \}} $$
Note that 0-1 loss (shown above) is an exact measure of classification error. If we optimise expected risk, when using 0-1 loss, then we have found the best classifier in $\mathcal{H}$ by definition.

However using 0-1 loss is not always feasible. See [[loss functions and convex optimisation]].

### empirical risk minimisation

- Draw a 'large enough' set of samples $(X, Y)^n$  
- Output hypothesis $h \in \mathcal{H}$ which minimizes disagreements with $(X,Y)^n$

the best we can do to learn a classifier from training data is empirical risk minimisation (as opposed to minimising 'expected risk', which is ideal but not possible via statistical methods, as discussed in [[objective function]]).

