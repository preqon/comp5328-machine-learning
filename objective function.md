# objective function

A function that minimises or maximises some numerical value. 

In context of classification, objective function is the measure of error on all possible data.

e.g. least squares objective function
- finds $h$ (a model) that produce the minimum average squared error
$$ \arg \min_{h} \frac{1}{n} \sum_{i=1}^{n} (y - h(x))^2 $$
note regressor = feature.

We use the objective function to build the best classifier. 
- since intuitively, we want the classifier that has the minimum class error on training data

e.g. find $h$ that produce minimum average classification error (as measured by 0-1 loss function) over all inputs
$$\arg \min \frac{1}{|D|} \sum_{i \in D} 1_{\{ Y_{i} \ne sign(h(X_i)) \}} $$
As size of D grows to infinite, the above function turns into arg min of expected value of classification error i.e. this is an unbiased estimator of the expected value.

Note: at the expected value of classification error, the objective function *exactly returns* the best classifier $h$.

We do not have infinite data, so we use samples to estimate the expected value.

If the objective function is not convex or smooth, then it is hard to optimise. See [[optimisation method]].
