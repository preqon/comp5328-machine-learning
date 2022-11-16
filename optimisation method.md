# optimisation method

This is how we obtain the hypothesis that optimises the [[objective function]], so it is a method to search through $h \in H$ to ultimately find
$$\arg \min_{h \in H} \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{ Y_{i} \ne sign(h(X_i)) \}} $$
Above is 0-1 loss, which gives the best classifier, by definition.
Written as a sum over $n$ samples, this is empirical loss, which as $n$ approaches infinite, approaches expected loss.

See [[loss functions and convex optimisation]].

### empirical risk minimisation

- Draw a 'large enough' sample $(X, Y)^n$  
- Output hypothesis $h \in \mathcal{H}$ which minimizes disagreements with $(X,Y)^n$

the best we can do to learn a classifier from training data is empirical risk minimisation (as opposed to minimising 'expected risk', which is ideal but not possible via statistical methods)

