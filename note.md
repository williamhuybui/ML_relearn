
**Virtual Environment** 
```
python -m venv myenv
source myenv/bin/activate
deactivate
```

**Git**
```
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/williamhuybui/ML_relearn.git
git push -u origin main
```
Requirement
`pip freeze > requirements.txt`


**Statistics 101**
**Variance ($\sigma^2$)**: how spread out is the dataset.
$$\sigma^2=\frac{\sum (x_i-\mu) ^2}{N}$$
**Standard deviation ($\sigma$)**: used to identify outlier. How many standard deviation from the mean?
**Probability density function**: provide the probability of a given range of value.
**Probability mass function:** provide probability for discrete data. Rolling a dice. 
**Uniform distribution**: all event has the same chance
**Gaussian distribution**: Given by probability density function. 

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

**Exponential pdf**: The exponential distribution models the time until the next event occurs in a Poisson process 
$$f(x) = \lambda e^{-\lambda x}$$
Example: If calls to a customer service center arrive at an average rate of 5 per hour $\lambda$, the time between calls $x$ follows an exponential distribution. 
**Poisson Probabillity mass function**: The Poisson distribution models the number of occurrences of an event in a fixed interval of time/space given a known average rate.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

My website gets on average 500 visits per day ($\lambda$). What's the odds of getting 550 ($k$)?

Other concept: Percentile, Box and whisker diagram.

Moments: quantitative measures of the shape of a probability density function:
$$\int^{\infty}_{-\infty}(x-c)^n f(x)dx$$
for moment n around value c.
If n = 1, (first moment), this is mean
If n = 2, (second moment) this is variance.
If n = 3, (third moment) this is skew. For normal distribution, negative value means skew to the left, and vice versa.
If n = 4, (fourth moment), this is kurtosis which describe the sharp of the tail. Negative value means, flatter tail (which means less extreme values). Positive kurtosis means thicker tail.
