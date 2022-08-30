

# Oral Exam Questions

## Computational Statistics and Data Analysis

### 1) Random numbers 

- Give some basic formulation in the mathematical framework of probability theory.
  - What is sample space? What is an outcome and what is an event?
  - What is the power set?
  - What is a sigma algebra?
  - What is a measure space?
  - What are the Kolmogorov axioms?
  - What is a probability space?
  - What are some conclusions?
  - What is a probability function?
- Difference between measure and distribution?
- Power set in real space/continuous regime?
- What is the formal definition of a random variable?
  - What is a realization?
- What is the formal definition of a probability distribution?
- Some definition and rules for conditional probability?
  - Baye's law? Derive.
  - What if A, B independent?
- What is the formal definition of the expectation value and variance?
  - Properties and reformulation of both?
  - What if X, Y independent?
  - What is correlation and what covariance?
- Give the Cauchy-Schwartz inequality.
- Give the Chebyshev inequality.
  - What is the condition?
- What is the weak formulation of the law of large numbers?
- Difference between summary statistics for sample and population?
- How to get probabilities in a discrete random process? What about the continuous case?
  - What is probability density, what does p(x)dx mean?
- Give the definition of a histogram and its probabilities.
  - What type of approximation(s) are we using?

### 2) Operations on random numbers

- How can we express properties of variance and covariance as a scalar product?
- What does the law of large numbers mean for the error of an estimator, e.g. the mean?
  - Dice example?
- How to transform a random variable?
  - Reformulate this without the integral. What is the probability density of the new variable?
  - What properties does the Jacobian have?
  - What about multiple transformations?
- What is a CDF?
- What are percentiles?
  - What about considering both sides of the distribution?
- What are the (two) problems of histograms and what are alternatives?
  - And what are their benefits?
- Explain rejection sampling.
- Explain inversion sampling.
  - What do we need? Why is it theoretically always possible?
- What are the pdf and cdf for a Gaussian?
- What is the error function, why is it useful?
  - How to read off percentiles?
- What is a Markov process of nth order for p(x, y)?
- What is a sum/product/ratio distribution
  - Gaussian example.
- Multivariate distributions?
  - What are marginalisation and conditionalisation?
  - And for statistical independence?

### 3) Random distributions

* What is a (nth) moment of a distribution?
* What is the characteristic function? How is it related to moments?
* Are distributions uniquely defined by their moments? Can we recreate the distribution from estimated moments?
* What is the moment generating function and why is it useful? Relation to Laplace transform?
* What are cumulants, how are they related to the characteristic function?
* What is the cumulant generating function?
* How to compute cumulants? (Faa di Bruno’s formula)
* What are the cumulants, moments, characteristic function of the Gaussian? How can it help to measure Gaussianity?
* Relation of CDF and percentiles?
* What is the Binomial distribution? What the Bernoulli distribution?
	* When to use it?
	* Mean and standard deviation?
* Mean and std of Bernoulli random walk?
* What is the multi nominal distribution?
* What is the meaning of the binomial coefficient and the binomial theorem (here: normalization)?
* What is the Poisson distribution?
	* When to use it?
	* Mean and std? Meaning of mean?
* What distribution for histogram bin and their counting error? Under what conditions? 
	* What happens to the absolute and relative error when changing number of histogram bins?
	* What trade-off for histograms?
* Meaning of relative error?
* What do the shapes of the Poisson and Binomial distribution look like for different parameters and number of events?
* What is Stirling’s approximation?
* What are the limits of and connection between Poisson, Binomial, Gaussian?
	* Compute these, using Stirling’s approximation, Taylor around peak.
* Write down the multivariate Gaussian distribution.
* What are the properties of the Covariance matrix? Why is this important?
* What is PCA, what are the principal components? 
	* When to use it?
	* Percentage of variance explained? 
	* What are the eigenvalues and eigenvectors?
	* How to perform a PCA from data?
* What is Pearson’s correlation coefficient?
* How to sample from a multivariate Gaussian? What is the Cholesky decomposition?
* Conditionalize/marginalize Gaussian.
	* How does variance change?
	* Calculate, complete the square.
	* What is the Schur complement?
	
### 4) Magic of Gaussian distribution
* What are the special properties?
* How to compute the moments?
* Where are the inflection points?
* Meaning of curvature of the natural log? (Fisher Information)
* How to compute moment generating and characteristic functions?
* What is skewdness? What is kurtosis?
	* For different values?
* What is the Edgeworth expansion? Is it exact?
* How does characteristic function change when scaling or adding distributions? And the cumulants?
* What is the central limit theorem?
	* Conditions?
	* Derivation with cumulants?
	* Convolution and addition of r.v.? For Cauchy?
* What is entropy, what are Shannon’s axioms?
* What distributions maximize Shannon entropy under what conditions?
* Continuous Shannon entropy and Renyi entropy?
	* Limit of Renyi entropy?
* Are there other measures of entropy?
* What is the variational principle?
	* Entropy as functional
	* Computation for two different conditions.
* Shannon entropy for Gaussian?
	* Edgeworth expansion?
* Entropy related to phase space volume?
* What is relative entropy/Kullback-Leibler divergence?
	* Properties? 
	* When doing inference from data?

### 5) Statistical Tests
* What are they? What types are there?
* 

### 6) Linear regression and likelihood

#### Linear regression 

- What is the goal of data fitting? What are the things we want to find out? (regarding shape of the likelihood)
- What distributions do we get for repeated measurements? 
  - What type of distribution is this in the Bayesian sense?
- How is the t-test related to data fitting?
- For fitting a straight line: what quantity do we minimize, what is it called?
  - What type of map is it, mathematically speaking?
  - If we set slope m=0, what does our optimal solution become?
- How do we derive the matrix equation for solving the linear fitting problem exactly?
  - Why matrix? System of ...
  - Under what condition is it soluble? 
  - What do we do to solve exactly for the parameters m, b?
- What is our (approx.) y functin for multiple linear regression? What even does this mean?
  - Where does the bias b appear in this equation?
- How do we fit a polynomial to the data? What does the function for the (approx.) y look like? 
  - Relation to multiple linear regression with linear features?
  - Why is it still linear when using polynomial features?
- How does the model for y in multiple linear regression look like in matrix form?
- What does the matrix equation (after taking derivative w.r.t. the features) look like for multiple linear regression (or polynomial)? (Both in matrix notation and element-wise with all the moments).
  - What is relation between X^T * X and the covariance of the regressors?
- From the matrix formulation for multiple linear regression, how do we get the exact solution? What is it called?
  - Derive this equation completely in matrix notation!
- What condition needs to be fulfilled for the normal equation to be soluble and why? How many parameters do we have, how many datapoints to we need to solve?
  - For what number of features and datapoints to we get an exact overfit?
  - What is the property of the system of eqns. when n < m+1? What happens to X^T * X? And columns need to be what?
- What is error weighting in the least-squares approach? When to use it?
  - What does chi-squared look like when std is not =1?
- Why do we have to be careful when interpreting polynomial regression?
- What is Pearson's correlation coefficient? 
  - What does it mean for the relationship between two variables? In particular, what does it mean when calculated between dependent and independent variable in terms of explanatory power?
  - How is it related to slope in linear regression?
  - What do we use to approximate standard deviation in real data?
  - Does slope directly tells us the strength of a linear relationship?

- Why/when use this approach over gradient descent and vice-versa?

#### Likelihood

- What is likelihood, what question does it answer?
  - Write it down for n iid Gaussian variables.
  - Is it a probability density function? Why?
  - What would you call it if not a distribution?
  - Does one likelihood value alone tell us something?

- How do we get to the OLS from the (iid) Gaussian likelihood? What do we optimize?
  - Why is it the same as maximizing log L?
  - Why is it the same as minimizing chi squared?
- For what reasons is it more convenient to use the log-likelihood? Both mathematically and computationally?
- What is the relationship between the likelihood and the posterior? 
  - Why do we want the posterior? And of what exactly?
- In Bayes law, what are the components and what are their meaning (in data fitting context)?
  - Interpretation of normalizing factor as marginal likelihood? 
- What do we call the maximum likelihood principle?
  - What happens if we repeat the measurement? In terms of biased and unbiased estimators?
  - What are the problems we can encounter? (Name three)
- How do we derive the OLS approach from the likelihood principle?
- Under what set conditions can we apply the OLS approach? What is the theorem called?
  - What do we NOT have to assume?
  - If all the conditions are fullfilled, what type of estimator does OLS give us?
- If we repeat the measurement, what distribution will we find for the chi-squared? Assume Gaussian errors.
  - How would we derive the function of this distribution? Trick with Fourier space?
  - What does it look like?
  - What are the mean and variance?

- What happens to the posterior when we repeat the measurements? 
  - What happens to the prior?
  - How does the shape change and why?
- What prior to choose for first measurement? And why?
- What quantity could we use to quantify the consistency of our fit to data?
- What p-value could we propose regarding chi-squared and the goodness of our fit?
  - Correct interpretation?
  - When would this approach be useful?
- What does the Gaussian likelihood look like for correlated datapoints?
- What is different when we repeat our experiment but the true underlying model remains the same?
- What if the features are correlated? What does this affect and what does it not affect?









## Time Series Analysis and Recurrent Neural Networks

--> also have a look into the tutorials etc.

--> check graphs on slides if not covered in my summary notes

--> ...

### 1) Introduction and basic terms

- How is a time series defined? What is usually the goal(s) in TSA? (Name four)
- What are some examples of time series, where to they occur?
- What is model-based TSA?
  - Statistical principle?
- How to examine local features?
- Definition of autocorrelation function?
  - What is the meaning/interpretation? --> positive and negative?
  - What does it look like for 1. no correlation, 2. "isolated" correlation, 3. regular correlation?
  - Properties of the graph? Draw the graph and its axes.
  - Dow Jones example: how does the trend and yearly cycle appear in the autocorrelation plot?
- Definition of first-order return plot?
  - Meaning/interpretation?
  - What are some downsides, or when might it not work perfectly? What to do then?
  - What does it look like? (Point pattern)
  - Dow Jones example: How is it influenced by the trend? (Could discover more non-linear relation)
  - Does it actually reflect a proper functional relationship between x_t and x_t+1?
- Connection of return plot and dynamics of a system? What part of a recursive map does it show?

- What does Fourier transform give us in TSA?
- What does the power spectrum express? What different insights can we get from it?

- What are the first things to do in TSA/appraoches to get a first look?
- What would non-linear vs. linear oscillation look like? And on return map?
- Are we always interested in the whole time series?
  - Do we need to plot return plot for entire time series?
- What distribution can we use to model time series/error in time series? (E.g. correct or not vs. counting number of errors, or in continuous case)
- How to deal with binary time series or count processes? 
  - E.g. how to visualize improvement in task perfomed correctly or incorrectly?
  - Example? (E.g. fine enough binning for events/spikes in neural activity)
  - Write down the autocorrelation function for a binary time series.
- What is a sigmoid function? What parameters do affect offset and slope?
- What to do in case of irregular sampling rates (e.g. spike time series shown for different units, or customers entering shops)? What could be the problem with the suggested approach? What do we lose, what do we gain? (--> e.g. example with spike trains)
- Three different images of autocorrelation and cross-correlation (/-covariance)? And their meaning? How to apply this to (binary point process) spike data? 
- Why do autocorrelation and cross-correlation have difference properties/appearances? (E.g. peak, symmetry)
- What other types of pattern could we look for? (E.g. spatio-temporal patterns like in what order customers enter a shop; or units of neurons => hierarchical patterns)
- Do peaks in time series always have meaning?
- In internet traffic example, ways of predicting peaks of high traffic? I.e. ways to predict certain patterns?

### 2) Auto- and cross-correlations and ARMA models

#### Auto- and cross-correlations, stationarity and power spectrum

- Basic premise for the following section? How does this define our approach? (Model-based --> infer model)

- Explain ergodicity. How does it help us?

- What is stationarity in the weak sense?

- Write down the formulation of stationarity in its strong formulation.

- What type of expectation value are we normally considering for time series, e.g. for stationarity? (over diff. realisations/ensemble)

- Can we always tell if a time-series is stationary? What are two ways of checking for (non-)stationarity?

- What are the moments of the conditional distribution? Are they also assumed to be constant? (for x_t|x_t-1 and x_t|x'_t-1)

  - What would be a condition under which this would be the case?

- Name three ways of dealing with non-stationary TS as well as their up- and downsides.

  - How can you remove frequencies? 
  - What information could be lost in "differencing TS"?

- What are the different types of differencing in TSA? What can they remove?

  - Why choose this over subtracting a fitted model or vice-versa?
  - Can you name a simple random TS example for a time series and its 1st-differenced TS? 

- Do we always want to remove the non-stationarity? Why?

- Name an example of a change point in a time series.

- What is white noise? Write down the definition (for 1D).

- Write down different formulations of auto- and cross-covariance and -correlation. (E.g. the notation as gamma and rho) 

  (check in notes for right formulation, e.g. still use time lag in the cross correlation)

  - What type of mean is the mu here?
  - Difference between covariance and correlation?
  - What values can they take?

- What are some practical problems when computing the auto-/cross-correlation for a time series?

- What are the two assumptions we almost always make?

  - What is the acov a function off when assuming stationarity? How can you then write it with rho and gamma?
  - Write down the approximation from data of: mean, variance, autocovariance (for latter: what might be practical problem and when does it not really matter?)
  - What symmetry follows from this?

- What is the power spectrum of a time series? What does it mean?

  - Is it the same as a Fourier transform? Why?
  - What can we learn from the highest value in the spectrum?
  - What information does it (not) contain? Is it invertible?
  - Pros and cons vs. FT?

- What is the Wiener-Khinchin theorem?

  - When is it valid? 
  - Do autocorrelation and power spectrum have independent meanings?

- What is the difference between Fourier series and transform? 

  - Interpretation of the terms of the Fourier series? 
  - What assumption do we need to make for the Fourier series?

- Relation (avg.) power and coefficients of Fourier series? How do we plot the power spectrum here?

- Power spectral density vs. energy spectral density? When does PSD exist?

- Relation between power, energy and amplitude of signal?

#### ARMA models (univariate)

- When are ARMA models applicable (under what condition)? When not applicable, what could we do?
  - What is an ARIMA model?
  - What is an ARMAX model?
  - What distribution for the error term? 
- What is the nature of the error? Can it be zero?
- Why even use ARMA when we can simply use RNN?
  - How is the equation related to an RNN?
- What is the merit in using linear models?
  - What are they completely defined by?
- What type of models are ARMA models (in the ML sense)? What is the advantage of this?
- Write down the general (univariate) ARMA model. What are the different components? 
  - What is the idea of the decomposition theorem?
- What are the limits of ARMA models?
- What are the parameters to estimate? 
- What are the things to determine/do in inference? (name 5)
- What is the autocorrelation for AR(1) and MA(1) models? What is their behavior?
- How are AR and MA models related? Derive this!
  - Why are these expressions valid as AR and MA model, respectively?
- For an AR(1) process, how is its stationarity related to the a1 coefficient? Or: what does the a1 coefficient of an AR(1) model tell us about the properties of the process?
  - If stationary, what convergence do we observe?
- How are the coefficients of the AR(1) model related to the autocorrelation function? Derive!
  - Why can we assume that a0=0? And that E[e_t * x_t-1] = 0?
  - Why is E[x_t * x_t-1] = acov(1) here? (mean-centering)
  - How to generalize this to any order? Name of the equations? Conditions?
- What is a partial autocorrelation function? How is it different to the autocorrelation function?
  - How do we compute the partial autocorrelation function on a time series? (using AR model fit)
- How do we find the optimal orders q, and p of AR and MA models?
  - What function goes to/towards zero (and which one does not) and under what exact condition for both cases?
  - What little assumtion do we make? (for a0, b0)
  - What cutoff could we choose for partial autocorrelation function in this case?
  - Explain the idea of "explaining variance" to a significant degree.
  - What is a residual time series?
- Match the following terms in an AR model: target, lagged target, error.

### 3) Statistical Inference in ARMA models

- For univariate ARMA models: what assumption do we make to reduce to AR model?
- How do we solve a univariate AR(p) model using linear regression?
  - How do we write down the MSE term? 
- Explain the interpretation of AR model inference as linear regression.
  - What do the regressors in multiple linear regression correspond to in the AR model inference?
  - While having a similar likelihood, why can we not simply take the product over individual univariate Gaussians, like usually in linear regression?
  
- What is a VAR model? What is the meaning of the entries of the A_i matrices? (i=1,...,p)
  - What is the covariance matrix of the error term?
- What is the relationship between AR(p) and VAR(1)? How do we show this?
  - Demonstrate for AR(2).
  - What about k-variate VAR(p) and k*p-variate VAR(1)?
- What is the condition regarding stationarity for a VAR(1) model?
  - Sketch the proof, what assumptions are we making?
  - (Non-)stationarity in all directions?
- What are we assuming for E[e_t * e_t']? And for e_t in general? 
- The mean of what distribution is x_t written as AR(p) model?
- Write down the joint likelihood function. How can we write it as a product?
  - What type of process do we have here by model assumption (AR(p))?
  - From where to where does the product run?

- Write down the LSE resulting from the MLE. What is the form of the resulting multivariate Gaussian?
  - Especially regarding the covariance matrix?

- How do we do the MLE for VAR models?
  - E.g. what would be the estimator for the covariance matrix?

- How to test two VAR models against each other?
  - What is the distribution of the test statistic?
  - What is an important difference to most common statistical test situations?
  - What do we need to make sure about the error term? How can we do this regarding our model design? How could we make sure/test that this is fulfilled?

- How to get optimal order for VAR models?
- What else could we test about VAR models?

### 4) Granger Causality and Point and Count processes

#### Granger causality

- What is the use case of Granger causality? How do you formulate the condition for X to "Granger cause" Y?
  - Is this a specific or a general expression?
  - Why is this expression in this form often hard to work with?
- How to formulate the Granger causality setting with AR models?
  - What formulation corresponds to "given Z without X" and "given" Z?
  - What is the testing procedure? What d.o.f. to use?
- ...
- ...



### 5+6) Into to State Space Models, EM Algorithm

- What model class do latent variable/state space models belong to?

- What is the principle assumption of state space models? 

  - What are we now looking for? Instead of what?

- Write down the typical formulation of a state space model (dicrete vs. continuous).

  - What are the two models we need to consider and what are their names?
  - What dimensions do X and Z have?

- What property should the latent DS have? What about the observed time series?

- Besides Markov property, what is another essential assumption for state space models?

  - What about correlations between x_t, x_t'? (I.e. consider p(xt, xt') = p(xt)p(xt')?)

- Why do DS have the Markov property? What does this mean in terms of determinism?

  - What exactly is the state space?

- Write down a short expression for the distributions of xt and zt in a SSM.

- In a state space model, what are the quantities that we ultimately want to estimate?

  - What sets of parameters? 
  - What moments?
  - What probability distribution?

- For inference in a state space model, what would be the two basic approaches?

  - What is the difference between them?

- Explain Jensen's inequality. For what type of functions does it hold?

  - Can you name a continuous example from probability?

- In a SSM, what do we optimise w.r.t. what in the MLE approach?

  - How would you write it in this case and why?

- What is Variational Inference? When to use it?

  - What is the basic principle?
  - What does the variational density approximate?
  - Interpret the individual terms that show up? Which ones are constant w.r.t. the parameters to be estimated? (Here, assume that we have not parameterized the approx. posterior/variational density)
  - What is the name of the final expression? What do we want to do with it for our MLE?
  - When is the bound tight?
  - What are the two components of the ELBO?
  - What are the arguments of the probability functions? (Upper or lower case x, z?)

- What is the Expectation Maximization algorithm, generally speaking? When to use it?

  - What exactly do we estimate? What type of estimate is it?
  - Describe the different steps in detail.
  - Is it a Bayesian approach?
  - Is it approximate or exact? Will it converge? To what?
  - Why does it work, why is the E-step quantity the one to maximize?

- What is a MAP estimate?

- What is the EM algorithm for our VI approach with the ELBO?

  - Why is argmax_q of ELBO equal to the general E step?
  - Why is argmax_thate of ELBO equal to the general M step?
  - What is a possible condition to stop the algorithm?
  - What if we could compute/knew q exactly?
  - What are the arguments of the probability functions? (Upper or lower case x, z?)

- Formulate a linear SSM and what parameters we want to find optimal choices for. 

  - What is the optimization problem, where does the ELBO appear here?

- Consider the M-step for a linear SSM.

  - How can we rewrite the joint likelihood and hence the expectation of the joint log likelihood? What are the different terms in the expression? 

    (How do we rewrite P(Z), P(X|Z)?)

    Also: how do we go from capital X, Z to the product/sum expression?

  - What distribution do we now insert? What do they look like, what are the means?

  - Explain the steps of the calculation and what we end up with.

  - What are the resulting moments that we need to obtain? Why only first and second order? Is this exact in our case? Where do we get these moments from?

  - Why are the results the same as for linear VAR models? So what is the advantage of this? E.g. give result for A.

- How is the E-step done for linear SSM? What is the name?

  - What do we estimate here? What for?
  - What would we normally have to do but what can we do here?
  - What do we split the posterior into? 

- Write down scheme for the filter-smoother recursion. What is the mathematical expression for z_t?

  - What does the filter do, what does the smoother do?
  - At time step t, what datapoints can the respecive loop take into account?

- What is the filter recursion?

  - Write down the filter recursion. How can we compute it and why?

  - In the last expression, what is likelihood, prior and posterior? What is a conjugate prior? How do we use this here?

    Write down the distribution at time step t and t-1. Now, what do we need to find?

  - How do we solve the integral, what do we get from it? What is the last step and what do we then obtain?

    (What is the name of the equations for the 1-step forward density, i.e. the first step in this question?)

  - What is the one-step forward density?

  - What are the expressions for the quantities we obtain and how to interpret those?

  - In the end, we get a recursive expression for what?

- What is the smoother recursion?

  - What does the smoother step "add" to our estimation of the posterior?
  - What is the principle difference to the filter recursion?
  - What is the advantage after already having done the computation of the filter step?
  - What datapoints can every step take into account?

- After filter-smoother recursion(s), what are the moments we get? 

  - Of what exact distribution?
  - What other calculation do we need to make?

- How does the Kalmann filter-smoother fit into the EM framework?

- What do we use as a first guess in the Kalmann filter smoother?

- What is the general notion of the filtering problem for stochastic processes?

### 7) Poisson SSM and Dynamical Systems Theory

#### Poisson SSM

- When would we use a Poisson SSM rather than a linear SSM?
  - How does it compare to a linear one? What are the observation and the latent model?
  - How do we compute the M-step, what is the difference to the linear model? What are the different terms in the resulting expression?
  - What are the different "components" of the observation model? (see GLM)
  - How do we expand the E[log ...] expression, what do we find?
  - What moments do we need for optimisation? What optimization can be done analytically?
  - What makes the numerical optimization here easier?
- How does one do the E-step for a Poisson SSM? What do we use?
  - Write down the expression for the posterior and point out the difference to the case of th linear SSM.
  - What approximation can we make here under what assumption?
  - What do we need to compute to get the final distribution?
  - What expression do we optimize?
  - How do you find the mean and covariance matrix of a Gaussian through optimization?
  - What distribution did we find the mean and covariance for in this problem?
  - How often do we need to solve this optimization problem?
  - What about the smoother step?

#### Dynamical systems

- What are dynamical systems? (in discrete or continuous time?)
  - Why do we mostly deal with discrete-time DS?
- What properties does a dynamical system have?
  - In the deterministic case?
- Where can a dynamical system be represented?
- What is the formulation of a univariate linear system?
  - How does it evolve in time, and under which condition?
  - What is a first association we can make with memory?
- What is a fixed point? 
  - What is the condition for a univariate linear DS to end up in a fixed point?
  - Derive the FP for such a DS.
  - What types of FP are there?
- What is the condition for a stable FP in a univariate linear DS?
  - Stationarity of ...?
- What is the basin of attraction?
  - Explicitly for a fixed point?
- Why is the DS formulation useful for our approaches to TSA?
- What is a first order return map and how is it useful for dynamical systems?
- What is a cobweb plot?
- Name all possible dynamical behaviors for a (first univariate) linear DS and the respective conditions!
  - How can you visualize this?
  - Interpretation/meaning of line attractor? Why no actual line "attractor"? 
  - What cycles can we observe? How many different cycles can there be? Bewteen what point do we cycle?
  - What about periodicity in higher dimensions?
  - What about the multivariate linear case? Stability of a FP?
  - What about line attractor in higher dimensions?
  - Related to stationarity of ...?

### 8) Non-linear Dynamical Systems and RNNs

#### Non-linear DS

- Prime example for non-linear map? 
  - Defined for what values usually?
  - What happens to the dynamical objects when changing the parameter?
  - What is a bifurcation and what is a bifurcation/control parameter
  - What is a bifurcation graph?
  - What behavior for logistic map at certain values?
- How to determine stability for a non-linear map (first in 1d)?
  - How to derive this rule?
  - In multivariate case? 
- How to assess the stability of cycles?
- Consider continuous DS.
  - What are limit cycles? What are their properties regarding stability? (name 3)
  - What is a method of assessing limit cycles and their stability? Explain the method.

- What is chaos, how do we characterize it? What is behavior and its "signature"?
  - What is a chaotic attractor? What about other dynamical objects?
  - What is a measure of chaotic behavior? (univariate vs multivariate?)
  - Does it diverge/converge in all directions? What does the phase space trajectory do? And what does it not do?
  - What about predictability?
  - What consequences does this have for machine learning?
  - Is it probabilistic? 

#### RNNs

- What are RNNs? 
  - In the dynamical sense/recursive map sense?
  - Compared to FNNs? What are the allowed inputs and outputs?
  - Can you draw a scheme of a general RNN?
- What are activation functions and what role(s) do they play?
  - Name four.
- Write down equation of RNN and explain the different terms.
- What is another way of writing the RNN equation and what is the "unrolled" form of an RNN?
  - Also, how do you write the not-unrolled form?
- What are the different modes of training for an RNN? (or like in ML in general...)
- Name advantages and disadvantages compared to linear models. 
  - Name possible applications.
  - Dynamical behavior?
- What does the dynamical behavior of a univariate RNN with sigmoid activation function look like?
  - Draw graph and function of the RNN.
  - What parameter is relevant for a bifurcation?
  - What bifurcation would we observe?
  - What can we do with an external input? How is this related to memory?
  - We say the system after its bifurcation is ...?
- What units of an RNN receive the input?

### 9) Universality and Training of RNNs

#### Universality

- What are two properties of the RNN? (regarding universality)
  - Or: what is universality?
  - Explain the to cases.
  - Why is this important for our applications?
- What is the universal approximation theorem for FNNs?
  - What would be the mathematical formulation?
  - What would a network for this proof look like?
  - Sketch the idea behind the theorem and its proof.
  - What parameters define height and position of the sigmoid building blocks?
  - What are the "hyper parameters" on which the precision in this model depends?
  - What to do for a multivariate function?
- What is a PLRNN? How can you rewrite a normal RNN to fit this equation?
- What steps are necessary to prove the dynamical universality of RNNs?
  - What seems to be a problem at first when looking at many common proofs? What type of RNN do they use?
  - How to reformulate a discrete-time RNN into a continuous one? (example of PLRNN)
  - What is a flow field of an ODE, how can we approximate this?
  - What are the last two steps?
  - What is one condition the weight matrix would have to fulfill for an RNN?

#### Training of RNNs

- What are the two big training paradigms for RNNs relevant for us?
  - For both cases, give a mathematical formulation.
  - Something to be careful about the time indices?
- What losses could we use for the training of RNNs and in what situations would we use them?
  - Explain the expressions of these loss functions.
  - What activation function would we use in case of a cross-entropy loss?
- What is the difference between classification and regression?
  - Give some examples relevant for RNNs/TSA.
  - How to design your RNN for a classification task? What would be the pdf in the end?
- How does gradient descent work?
  - Sketch in 1D.
  - Give equation for higher dimensions. 
  - What are some problems with this? (name at least 3)
  - Under what circumstances would it converge to a local minimum?
  - How to maximize log-likelihood with gradient descent?
- What are some remedies for the problems of gradient descent?
  - What would be the two problems we could tackle?
  - Name numerous remedies for both problems. 
- How does stochastic gradient descent work?
  - What are its advantages?
  - What do we need to balance?
- How does the Newton-Raphson algorithm work?
  - What are its upsides?
  - What are some of its problems and possible remedies?
  - For what case does it work the best?
- What is RTRL?
  - Show the computation of the gradient of the weight matrix.
  - How can we interpret this equation, what property does it have?
  - What does that mean for the behavior of the training/loss?
  - How do we use this recursive equation in practice?
  - What is the recursive rule, exactly?
- What is BPTT?
  - How does it differ from RTLR?
  - Write down the scheme for an RNN and its unrolled-in-time form to explain this.
- How does backpropagation work in general?
  - What do we have to do first? And why?
- What types of regularization can you name? How do they differ/what do they do?
- What is the bias-variance trade-off?
  - Which one is underfitting, which one overfitting? What do those mean?
  - Why is it a trade-off?
- What principle differs from modern ML to classical statistics?
- What is an inductive bias?
- What is cross-validation and how to use it?
  - What is the basic principle of a train-val-test split?
  - What is the advantage of cv over a simple train-val-test split?
  - How many times do we train our model?
  - What are the use-cases of this?
  - What ratios are good for this type of split?
- What is the EVG problem?
  - How can you illustrate it with DS theory?
  - RTRL vs. BPTT?
  - Is it possible for other NNs?
- What are different solutions to the EVG problem?
- What is an LSTM? Draw a scheme!
  - What activation functions do we use?
  - What do the gates do? What are their values?
  - What is the meaning of the cell state?
  - Why does this architecture help with the EVG problem?
- What is a GRU?
- What is MAR and how does it work? What if not exactly enforced?
- Do gated RNNs help against exploding gradients again?
  - What else to do?

- Why "recurrent" NN?









MAKE SURE TO KNOW THE EVP AND BPTT EXPRESSIONS BY HEART

HWAT ABOUT MUTUAL INFORMATION???

why do we take extremem at peak in chapter 7?? 

DOUBLE CHECK THE ASSESSING STABILITY ON POINCARE MAP

KNOW SOLUTIONS OF PARAMETERS FOR LINEAR AR/VAR MODELS BY HEART SO CAN COMPARE THEM TO LINEAR M STEP

EXPLICIT CALCULATIONS? E.G. FOR LINEAR SSM 

STABILITY OF A p CYCLE?

​	DONT FORGET THE END OF CHAPTER 3

logistic regression as GLM

assumption for DS: not x_t = F(x_t-1) but with underlying DS with latent variable z_t! So observations do not just depend on themselves.

- unbiased approximator for the variance??

have a look at exercises or the comp stats exam...