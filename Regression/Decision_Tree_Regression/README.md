# Decision Tree Regression: -

Tree based learning algorithms are known to be one of the best aalgorithms in Machine-Learning. Tree based methods empower 
predictive models with high accuracy, stability and ease of interpretation. Unlike linear models, they map non-linear relationships 
quite well and are adaptable at solving any kind of problem at hand (classification or regression).

In Regression trees, the value obtained by terminal nodes in the training data is the "mean" response of observation falling in that region. 
Thus, if an unseen data observation falls in that region, it’ll make the prediction with the "mean value".

The Tree follows a top-down greedy approach known as recursive binary splitting. It is known as ‘greedy’ because, 
the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to 
a better tree.

### How does a tree decide where to split?

The algorithm uses the standard formula of variance to choose the best split. The split with lower variance is selected as the 
criteria to split the population

For more information: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

### Execution: -

To run the code, type `DTR_Pred.py`

```
run DTR_Pred.py
```

OR

Open the iPython notebook `DTR_Pred.ipynb`

```
Open DTR_Pred.py
```
