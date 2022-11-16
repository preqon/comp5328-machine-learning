# objective function

A function that minimises or maximises some numerical value. 

In context of classification, objective function is the measure of error on all possible data.

e.g. least squares objective function
- optimising this objective finds $h$ (a hypothesis) that produces the minimum average squared error
$$ \arg \min_{h} \frac{1}{n} \sum_{i=1}^{n} (y - h(x))^2 $$
---

## how to use an objective function in classification

We rely on the objective function to build the best classifier. 
- since intuitively, we want the classifier that has the minimum class error on training data

e.g. find $h$ that produces minimum average classification error (as measured by 0-1 loss function) over all inputs
$$\arg \min_h \frac{1}{|D|} \sum_{i \in D} 1_{\{ Y_{i} \ne sign(h(X_i)) \}} $$
**law of large numbers**:
As size of $D$ grows to infinite, the above objective turns into the expected value of classification error.
i.e. finding the empirical mean of class error, across random independent samples, is an unbiased estimator of the expected value of class error.
this law is also assumed in the Monte Carlo method.

Note: at the expected value of classification error, optimising the objective function *exactly returns* the best classifier $h$.

Since we do not have infinite data, we use a large number of samples to estimate the expected value of class error.

If the objective function is not convex or smooth, then it is hard to optimise. See [[optimisation method]].

---
**empirical risk**: empirical mean of class error.
**expected risk**: expected value of class error.
