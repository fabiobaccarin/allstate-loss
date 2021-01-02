# Allstate Claims Severity
Kaggle competition ([source](https://www.kaggle.com/c/allstate-claims-severity/overview)). Objective: predict how much will cost an insurance claim

## Target preprocessing
Because this is a regression problem, it is fundamental to understand the distribution of the target. Gaussian targets are much easier to predict and also give more reliable performance metrics. Our target is typically distributed for a monetary value: it follows a [power law](https://en.wikipedia.org/wiki/Power_law), which is highly skewed (about 3.84 skewness). This requires that we do some transformation to stabilize the variance. We have three options:
1. Natural logarithm
2. Yeo-Johnson transformation
3. Quantile transformation

I did all three of them. For each of them, I recorded the time required to use it and the respective p-value (on a negative logarithm scale) on the skewness test. To decide among them, I used a cost-effectiveness analysis, dividing the time required to fit (in milliseconds) by the inverse of the scaled p-value. The results are in Table 1 below.

|Transformation|Insignificance|Time|Cost Effectiveness Ratio|
|:--- |---: |---: |---: |
|Yeo-Johnson|14.91|281.56|18.89|
|Quantile Transformer|1.28|69.77|54.35|
|Log|0.02|5.51|330.30|

The rationale here is very simple. The cost of running different transformations is the time required to use them, for time is the factor driving all sorts of costs (financial or computational). The benefit we obtain from running them here is the reduction in skewness, which I measured by the inverse of the scale p-value on the skewness test. So, the surer we can be about the skewness not being different from zero, the greater the benefit. I called this metric "Insignificance". When I divide the cost by the benefit, I get how much it costs to have an arbitrary amount of confidence in the symmetry of our distribution.

So, we see that Yeo-Johnson has the smallest ratio, meaning that it is the one for which the benefits most outweigh the costs. Notice that is is by far the most expensive to run, but it is worthy of its cost because of a disproportional decrease in skewness compared to the other two transformations. Here, alternatives are so dissimilar that we don't need to worry about uncertainty in time measurements. Although it is good to have such measures, here it is of little practical value.

## Dropping categorical features with no variance after preprocessing
A quick look at Pandas Profiling's results hints that many categorical features will be constant after preprocessing. So I made a script to find out which of them will be. I found that 57 out of 116 features will be in the group (49 percent).

## Continuos preprocessing
For continuous features, the Quantile Transformation is the most cost-effective transformation to resolve skewness. This is shown in Table 2 below.

|Transformation|Insignificance|Time|Cost Effectiveness Ratio|
|:--- |---: |---: |---: |
|Quantile Transform|20.76|9.42|0.45|
|Yeo-Johnson|1.70|16.97|9.98|
|Logarithm|0.03|1.35|40.85|

Many continuous features are highly correlated (Pearson correlation above 75% percent), as shown in Figure 1 below.

![Figure 1: Correlations above 75% in continuous features](reports/figures/01ContFeaturesCorr75Matrix.png)

So solve this, I chose to do a principal component analysis. In order to decide how many components to retain, I computed how much explained variance I get for each component. The results are in Table 3 below:

|PCA Component|Explained Variance|Accumulated Explained Variance|
|:--- |---: |---: |
|Component 1|43.99%|43.99%|
|Component 2|13.19%|57.18%|
|Component 3|10.29%|67.47%|
|Component 4|7.18%|74.65%|
|Component 5|6.78%|81.43%|
|Component 6|5.60%|87.03%|
|Component 7|3.85%|90.89%|
|Component 8|2.79%|93.67%|
|Component 9|2.26%|95.94%|
|Component 10|1.94%|97.88%|
|Component 11|1.18%|99.06%|
|Component 12|0.54%|99.60%|
|Component 13|0.36%|99.96%|
|Component 14|0.04%|100.00%|

Because there is only 14 continuous features, I decided to retain 11 components, accounting for a little over 99% of the total variance in the continuous feature matrix.