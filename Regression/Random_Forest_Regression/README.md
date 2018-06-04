# Random Forest Regression: -

Random Forest is a flexible and easy to use machine learning algorithm that produces a great result even without hyper-parameter tuning
most of the time. It is also one of the most used algorithms, because of it’s simplicity and the fact that it can be used for 
both classification and regression tasks.

### How do Random-Forests work?

The forest it builds is an ensemble(group) of Decision Trees, most of the time trained with the “bagging” method. 
The general idea of the bagging method is that a combination of learning models increases the overall result.

To say it in simple words: Random forest builds multiple decision trees and merges them together to get a more 
accurate and stable prediction

Important: The random-forest algorithm brings extra randomness into the model, when it is growing the trees. Instead of searching for the 
"best feature" while splitting a node, it searches for the "best feature among a random subset of features. 
This process creates a wide diversity, which generally results in a better model.

For more information: https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd

### Execution: -

To run the code, type `RFR_Pred.py`

```
run RFR_Pred.py
```

OR

Open the iPython notebook `RFR_Pred.ipynb`

```
Open RFR_Pred.ipynb
```
