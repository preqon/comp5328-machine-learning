# dictionary learning and non-negative matrix factorisation

A dictionary provides a set of basis vectors, used to describe the possible samples drawn from a distribution, in lower dimension. Dictionary learning (or 'matrix factorisation') can be thought of dimension reduction or feature extraction, whereby the most salient features of the data are chosen to represent each sample.

An approximation of the original data matrix $X$ is given in the form 
$$ X \approx D R $$
- $D$ columns can interpreted as the basis vectors
- $R$ columns can interpreted as projections of samples from $X$ into a low dimensional subspace, via basis vectors. $$R = D^TX$$
- So the matrix product $DR$ approximates each sample with some linear combination of basis vectors.

Let's say $R$ was one dimension only, the vector $\alpha$. 
Consider some sample $x \in \mathbb{R}^d$ , and $D \in \mathbb{R}^{d \times k}$ 

$$\alpha ^{*} = \arg \min_{\alpha \in \mathbb{R}^k}
|| x - D\alpha ||^2$$
- where $|| x ||^2 = \sqrt{x^Tx}$ is L2 Norm.

So the optimal $\alpha$, once combined with $D$ should closely reproduce $x$.

So for training sample set $X$, we would like to optimise

$$\arg \min_{D, R} = || X - DR ||_F^2$$
- where R contains $\alpha_i$ for each $x_i$.
- $||X||_F$ is the Frobenius norm given by $$\sqrt{\sum_{i=1}^d \sum_{j=1}^n X^2_{i,j}}.$$
Note, the objective is convex with respect to either $D$ or $R$ but not to both.
We can fix one and solve for the other.

Once we have local minimum solutions for $D$ and $R$ we have
$$X \approx D^*R^* = (D^*A)(A^{-1}R^*) $$
We sometimes normalise each column of $D$.

### principle components analysis

- special requirement: columns of $D$ are orthonormal to eachother.

Given a set of n-dimensional samples $X \in \mathbb{R}^{n \times m}$, PCA considers each samples' projection onto an r-dimensional space $R \in \mathbb{R}^{m \times r}$ via basis vectors $D \in \mathbb{R}^{n \times r}$. 
$$D^TX = R $$

We want weights (i.e. components) in each basis vector such that the variance of our projection is maximised. 
- ensures $R$ represents the samples well, since it increases the likelihood that different samples map to unique points in $R$'s subspace.

Individual basis vectors $w \in D$ are optimised using the following objective function.
$$\max_{w} w^TCw $$
$$\text{s.t.} \quad w^Tw = 1 $$
where $C \in \mathbb{R}^{n \times n}$ is the covariance matrix of $X$ (describing the covariance between each pair of the samples' dimensions) given by
$$ \frac{1}{m} \sum_{i=1}^m (x_i - \bar{x}) (x_i - \bar{x})^T $$

So we optimise such that each $w$ maximises weighted covariance between pairs of dimensions. The basis vectors found can be thought of as pushing samples' dimensions apart optimally. It can then be shown that each $w$ is exactly the leading eigenvectors of $C$ with the largest eigenvalues.

### singular value decomposition

- data samples are projected, using the left singular vectors of $X$, that have the largest singular values. 
- these left singular vectors turn out to be the eigenvectors of $XX^T$.

### k-means clustering

Finds $k$ number of clusters in data.
Each cluster can be thought of as a column in $D$; each column in R is then 1-hot.
A sample which is the mean of a cluster can be produced as $D\alpha$.

## NMF

In a wide range of data mining tasks data matrices are often non-negative. Yet approximations via PCA and SVD yield negative values inside the samples' projections. 
- NMF constrains $D$ and $R$ to be non-negative, given a non-negative matrix $X$. 
- The approximation $DR$ can then be considered parts-based, in the sense that there are no subtractive components, only additive.
- parts based dimension reduction produces great interpretability in results.
- however factorisation is not unique, and there can be feature redundancy among basis vectors.

We minimise the objective as above

$$\arg \min_{D, R} = || X - DR ||_F^2$$

but using a multiplicative update rule, with the NMF constraints.
- this update rule makes the objective non-increasing
- the objective will not change (stays invariant) if and only if updated values for D and R are 'at a stationary point of the distance' (?)

Note that the objective above is Euclidean distance loss on approximation error $(E_{ij} = X_{ij} - \sum_{k=1}^K D_{ik}R_{jk})$ 
$$\sum_{i=1}^{n} \sum_{j=1}^{m} || \space X_{ij} - (DR)_{ij} \space || \thinspace ^2 $$
A local minimum can be found, by first randomising $D$ and $R$ then iterating over the following multiplicative update rules:
$$R_{ij} \leftarrow R_{ij} \frac{(D^TX)_{ij}} {(D^TDR)_{ij}} $$
$$D_{ij} \leftarrow D_{ij} \frac{(XR^T)_{ij}} {(DRR^T)_{ij}} $$

### NMF variants

- loss functions other than squared loss
- compensation for feature redundancy
- improving generalisation ability
