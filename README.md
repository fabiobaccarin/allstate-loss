# Allstate Claims Severity
Kaggle competition ([source](https://www.kaggle.com/c/allstate-claims-severity/overview)). Objective: predict how much will cost an insurance claim

## Target preprocessing
Because this is a regression problem, it is fundamental to understand the distribution of the target. Gaussian targets are much easier to predict and also give more reliable performance metrics. Our target is typically distributed for a monetary value: it follows a [power law](https://en.wikipedia.org/wiki/Power_law), which is highly skewed (about 3.84 skewness). This requires that we do some transformation to stabilize the variance. We have three options:
1. Natural logarithm
2. Yeo-Johnson transformation
3. Quantile transformation

I did all three of them. For each of them, I recorded the time required to use it and the respective p-value (on a negative logarithm scale) on the skewness test. To decide among them, I used a cost-effectiveness analysis, dividing the time required to fit (in milliseconds) by the inverse of the scaled p-value. The results are in Table 1 below.

|Transformation|Insignificance|Time|CostEffectivenessRatio|
|:--- |---: |---: |---: |
|Yeo-Johnson|14.91|278.46|18.68|
|Quantile Transformer|1.28|65.00|50.63|
|Log|0.02|5.36|321.26|


The rationale here is very simple. The cost of running different transformations is the time required to use them, for time is the factor driving all sorts of costs (financial or computational). The benefit we obtain from running them here is the reduction in skewness, which I measured by the inverse of the scale p-value on the skewness test. So, the surer we can be about the skewness not being different from zero, the greater the benefit. I called this metric "Insignificance". When I divide the cost by the benefit, I get how much it costs to have an arbitrary amount of confidence in the symmetry of our distribution.

So, we see that Yeo-Johnson has the smallest ratio, meaning that it is the one for which the benefits most outweigh the costs. Notice that is is by far the most expensive to run, but it is worthy of its cost because of a disproportional decrease in skewness compared to the other two transformations. Here, alternatives are so dissimilar that we don't need to worry about uncertainty in time measurements. Although it is good to have such measures, here it is of little practical value.

## Dropping categorical features with no variance after preprocessing
A quick look at Pandas Profiling's results hints that many categorical features will be constant after preprocessing. So I made a script to find out which of them will be. I found that 57 out of 116 features will be in the group (49 percent).