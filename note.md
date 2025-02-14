
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

# Statistics 101

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

If $n = 1$, (first moment), this is mean

If $n = 2$, (second moment) this is variance.

If $n = 3$, (third moment) this is skew. For normal distribution, negative value means skew to the left, and vice versa.

If $n = 4$, (fourth moment), this is kurtosis which describe the sharp of the tail. Negative value means, flatter tail (which means less extreme values). Positive kurtosis means thicker tail.

**Plot:** 
`sns.distplot()`
`sns.pairplot()`
`sns.jointplot()`: (hist + scatter)
`sns.lmplot()`: (scatter with linear regression)
`sns.boxplot(x,y)`: (Show all boxes for cat in x)
`sns.swarmplot()`: Like boxplot, but it show the datapoints. 
`sns.countplot(x, data)`: count vs category in column data[x] 

**Covariance:** Measure how two variables change together. Positive show linear relationship, negative is opposite, and 0 is none. It is measure by the dot product of diviation from the means of the 2 variables. 

**Correlation** Covariance with standardization, make the value stay between -1 and 1

**Conditional Probability**
$$P(B|A) = \frac{P(A|B)\cdot P(B)}{P(A)}$$

# Predictive model

**Linear Regression**

Linear regression is a statistical method used to model the relationship between a dependent variable $y$ and one or more independent variables $x$. It estimates the best-fitting line using Least Squares or Gradient Descent.

**Least Squares Method**
* Goal: Minimize the sum of squared differences between the actual data points and the predicted values.

* Formula for minimizing squared error (cost function):

$$J(m, b) = \sum (y_i - (m x_i + b))^2$$

In statistics, regression is also considered a **Maximum Likelihood Estimation (MLE)** technique when the residuals (errors) follow a normal distribution.

**Gradient Descent (Alternate Method)**
Instead of Least Squares, Gradient Descent is used for optimization, especially for large datasets.

Steps in Gradient Descent:
1.	Compute the gradient of the cost function.

2.	Update parameters iteratively to minimize the error.

3.	Repeat until convergence.
	•	Update Rules:

$$m = m - \alpha \frac{\partial J}{\partial m}, \quad b = b - \alpha \frac{\partial J}{\partial b}$$

where  $\alpha$  is the **learning rate**.


**R-Squared ( R^2 )**: Coefficient of Determination. It's measures the quality of the fit.

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

where:
	•	 $\hat{y}_i$  = predicted values,
	•	 $\bar{y}$  = mean of actual values.
	•	Interpretation of  $R^2$ :
	•	 0  → The model explains none of the variability.
	•	 1  → The model explains all of the variability.
Higher  $R^2$  values indicate a better fit.

5. Example: Linear Regression in Python

Using SciPy to compute linear regression parameters:

from scipy import stats
```
# Compute regression
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

# Print results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")

```
**Linear Regression Assumptions**
•	Linearity: Relationship between X and Y should be linear.
•	Independence: Observations should be independent (no autocorrelation).
•	Homoscedasticity: Residuals should have constant variance.
•	Normality of Errors: Residuals should be normally distributed.
•	No Multicollinearity: Predictor variables shouldn’t be highly correlated (check VIF).

Prevent with **regularization** (Lasso/Ridge)

**Polynomial Regression**

`np.poly1d(x,y,n)` : Return a lambda function that takes array of x and y and create a polynomial of order n

**Multiple Regression**: Regression with more than 1 feature
**Multivariate Regression**: Like multiple regression but can have more than one variable to predict

Still have independence assumption. 

 Ordinary Least Squares (OLS)
	•	A method to estimate regression coefficients by minimizing the sum of squared errors.
	•	Used in linear, multiple, and multivariate regression.

Formula for OLS Regression Coefficients:


\beta = (X^TX)^{-1}X^Ty

where:
	•	 X  = matrix of predictor variables (features),
	•	 y  = vector of observed outputs (target variable),
	•	 \beta  = vector of regression coefficients,
	•	 (X^TX)^{-1}X^T  is the Moore-Penrose pseudoinverse used to find the best fit.

How It Works:
	•	Step 1: Construct the feature matrix  X  and response vector  y .
	•	Step 2: Compute  (X^TX)^{-1}X^T .
	•	Step 3: Multiply by  y  to obtain  \beta , which contains the estimated coefficients.

```
import numpy as np

# Sample data (X: Features, y: Target variable)
X = np.array([[1, 2], [2, 3], [3, 4]])  # Feature matrix
y = np.array([3, 5, 7])  # Target values

# Add a column of ones for the intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Compute OLS estimate
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("OLS Coefficients:", beta)
```

```# Bucket data
bins = np.arange(0, 50000, 10000)
pd.cut(df['Milage'], bins)
```