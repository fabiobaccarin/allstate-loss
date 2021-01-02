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

## Correlation analysis
Many features are highly correlated (Pearson correlation above 75%), as shown in Figure 1 below. There are 18 features in such a situation, both categorical (after preprocessing) and numerical.

![Figure 1: Correlations above 75%](reports/figures/01ContFeaturesCorr75Matrix.png)

So solve this, I chose to do a PCA. In order to decide how many components to retain, I computed how much explained variance I get for each component. I processed continuous and categorical features seperately because continuous features have much larger variance. This induce a bias in PCA towards them.

Curiously, for both features the number of selected components was 6, accounting for a little over 99% of total variation. The explained variance for each component are shown in Tables 3 and 4 below.

|PCA Component|Explained Variance|Accumulated Explained Variance|
|:--- |---: |---: |
|Component 1|62.78%|62.78%|
|Component 2|14.81%|77.59%|
|Component 3|11.49%|89.09%|
|Component 4|4.77%|93.85%|
|Component 5|4.45%|98.30%|
|Component 6|0.97%|99.26%|
|Component 7|0.24%|99.51%|
|Component 8|0.24%|99.74%|
|Component 9|0.18%|99.92%|
|Component 10|0.08%|100.00%|
|Component 11|0.00%|100.00%|

|PCA Component|ExplainedVariance|AccumulatedExplainedVariance|
|:--- |---: |---: |
|Component 1|77.34%|77.34%|
|Component 2|12.17%|89.51%|
|Component 3|4.42%|93.93%|
|Component 4|3.38%|97.31%|
|Component 5|1.57%|98.87%|
|Component 6|1.04%|99.91%|
|Component 7|0.09%|100.00%|


Because the number of components is relatively low, I decided to retain components to account for a little over 99% of the total. To understand how each feature is used, I plotted the weights of each feature for every component. The results are shown in Figures 2 and 3 below.

![Figure 2: PCA weights - categorical features](reports/figures/02CatPCAWeights.png)
![Figure 3: PCA weights - continuous features](reports/figures/03ContPCAWeights.png)

## Feature ranking
To understand which features are the best, I did a simple ranking using Spearman's correlation after some preprocessing of the target and of categorical features. We can visualize some of them in Figure 3 below.

![Figure 4: volcano plot for features](reports/figures/04FeatureRanking.png)

We see many features highly significant and strong. As shown in Table 4 below, the best features are categorical and reach up to 47% correlation with the target. Naturally, their statistical significance is so great that Numpy setted them to inifinity. Here, I assigned 1,000 to those features in to make them appear in Figure 3.

|Feature|Association Strength|Statistical Significance|Rank|Group|
|:--- |---: |---: |---: |---: |
|cat80|-47.14|1000.00|1.00|1.00|
|cat79|-39.65|1000.00|2.00|1.00|
|cat12|-34.22|1000.00|3.00|1.00|
|cat10|29.98|1000.00|4.00|1.00|
|cat81|-29.06|1000.00|5.00|1.00|
|cat1|-27.78|1000.00|6.00|2.00|
|cat2|-26.58|301.34|7.00|2.00|
|cat9|-25.37|273.68|8.00|2.00|
|cat11|-24.79|260.95|9.00|2.00|
|cat72|-24.47|254.05|10.00|2.00|
|cat101|-24.05|245.22|11.00|3.00|
|cat13|-23.00|223.77|12.00|3.00|
|cat3|-19.87|166.28|13.00|3.00|
|cat90|-19.10|153.50|14.00|3.00|
|cat6|-18.88|149.87|15.00|3.00|
|cat23|-18.72|147.38|16.00|4.00|
|cat50|18.20|139.19|17.00|4.00|
|cat100|-18.11|137.89|18.00|4.00|
|cat36|-17.71|131.80|19.00|4.00|
|cat73|-17.58|129.88|20.00|4.00|
|cat4|-13.25|73.72|21.00|5.00|
|cat38|13.01|71.11|22.00|5.00|
|cat5|12.75|68.37|23.00|5.00|
|cat25|11.59|56.62|24.00|5.00|
|cat103|-10.89|50.02|25.00|5.00|
|cat44|-9.70|39.90|26.00|6.00|
|cat82|-9.65|39.44|27.00|6.00|
|cat102|-9.54|38.60|28.50|6.00|
|cat8|-9.54|38.60|28.50|6.00|
|cont2|9.51|38.36|30.00|6.00|
|cat111|8.60|31.49|31.00|7.00|
|cat26|7.36|23.30|32.00|7.00|
|cont11|6.61|18.96|33.00|7.00|
|cont12|6.53|18.54|34.00|7.00|
|cont7|6.42|17.96|35.00|7.00|
|cat87|-6.30|17.29|36.00|8.00|
|cont3|6.28|17.21|37.00|8.00|
|cat53|5.77|14.63|38.00|8.00|
|cat75|5.75|14.56|39.00|8.00|
|cat94|-4.01|7.43|40.00|8.00|
|cat113|3.60|6.10|41.00|9.00|
|cat83|-3.50|5.80|42.00|9.00|
|cont8|3.24|5.07|43.00|9.00|
|cat84|3.20|4.94|44.00|9.00|
|cat92|3.09|4.66|45.00|9.00|
|cont6|3.09|4.65|46.00|10.00|
|cat37|2.40|3.01|47.00|10.00|
|cont5|-2.38|2.97|48.00|10.00|
|cat110|-1.87|1.99|49.00|10.00|
|cat27|1.80|1.87|50.00|10.00|
|cat104|1.80|1.87|51.00|11.00|
|cat107|-1.78|1.84|52.00|11.00|
|cont10|1.73|1.76|53.00|11.00|
|cat98|-1.72|1.73|54.00|11.00|
|cont9|1.68|1.67|55.00|11.00|
|cat88|1.58|1.53|56.00|12.00|
|cat91|-1.45|1.33|57.00|12.00|
|cat112|1.43|1.30|58.00|12.00|
|cat109|1.41|1.28|59.00|12.00|
|cat86|-1.34|1.18|60.00|12.00|
|cont13|1.28|1.10|61.00|13.00|
|cat116|1.22|1.02|62.00|13.00|
|cat114|-1.20|1.01|63.00|13.00|
|cont1|-1.11|0.89|64.00|13.00|
|cat108|-1.06|0.84|65.00|13.00|
|cat96|1.00|0.77|66.00|14.00|
|cont14|0.93|0.70|67.00|14.00|
|cat93|0.87|0.63|68.00|14.00|
|cat115|0.84|0.61|69.00|14.00|
|cat106|0.73|0.50|70.00|14.00|
|cat105|-0.70|0.48|71.00|15.00|
|cont4|-0.67|0.45|72.00|15.00|
|cat97|-0.59|0.38|73.00|15.00|
|cat95|0.28|0.15|74.00|15.00|
|cat99|-0.22|0.12|75.00|15.00|

Categorical features dominate by far. The strongest continuous feature is `cont2` with about 9.5% association. It is in the 30th place.

## Feature selection
Because doing nothing is always an option, I ran a dummy regressor that always predicts the mean loss. I needed this to measure the benefit in predictive power between models in relation to this benchmark in percentage. This way, for each model I can compute how much the RMSE drops compared to the cost of running the model (represented by the time to fit) for each model.