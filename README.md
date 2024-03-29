# Decision_Trees

A decision tree in machine learning is a method used for making decisions based on data. It's like a flowchart where each branch represents a choice between different options, leading to different outcomes or predictions. The goal is to maximize purity(i.e., all data points in a node belong to the same class) or minimize impurity. 

Decision Trees work well on tabular structured data. Decision Trees are a straightforward and versatile tool in machine learning, but they have some limitations, especially when dealing with complex, real-world scenarios.



---
### How it works:
In a decision tree, data starts at the root node, undergoes a series of splits (based on the features of the data), and eventually ends up in one of the leaf nodes, which provides the decision or prediction of the model. The goal is to have pure nodes (i.e., all data points in a node belong to the same class).


1. Nodes:

   - Root Node: This is the starting point of the tree. It represents the entire dataset, which then gets divided into two or more homogeneous sets.
  
   - Decision Node: When a sub-node splits into further sub-nodes, it's called a decision node. These are the points where the data is split based on a certain condition or attribute.

2. Splits:

   - Splits are the criteria or conditions that divide nodes into two or more sub-nodes. For example, in a decision tree classifying animals, a split might be based on whether the animal can fly. Each split aims to make the resulting sub-nodes as pure as possible, meaning the data in each sub-node is more similar to each other and different from the data in other sub-nodes.

3. Leaves or Leaf Nodes:

   - Leaf nodes are the final output of the decision tree. These nodes do not split any further and contain the outcome or prediction. In classification trees, each leaf node represents a class label. In regression trees, they represent a continuous value.

4. Node Impurity:
   
   - The goal is to have pure nodes (i.e., all data points in a node belong to the same class). Impurity is used as a criterion for splitting nodes. This is a measure of the homogeneity of the labels at the node. 

---
### Advantages:

- Easy to Understand and Interpret: Decision trees can be visualized, making them easy to understand even for people without technical knowledge. They mimic human decision-making.

- Handles both Numeric and Categorical Data: They can work with different types of data – whether it's numbers or categories.

- Requires Little Data Preprocessing: Unlike some other algorithms, decision trees often don't require extensive preprocessing of data (like normalization or scaling).

- Useful for Feature Selection: Decision trees can identify the most effective variables for classification or prediction.

### Disadvantages:

- Overfitting: Decision trees can create overly complex trees that do not generalize well from the training data, known as overfitting.

- Can be Unstable: Small changes in the data might result in a completely different tree being generated.

- Biased with Imbalanced Data: If some classes dominate, decision trees can create biased trees. Balancing the dataset is usually required.

- Not the Best for Continuous Variables: They are not well-suited for tasks where prediction of continuous values (like predicting house prices) is needed.




---
## Important terms related to decision trees in machine learning:

- Branch / Edge: These are the lines connecting nodes, representing the flow from one node to another. A branch represents the outcome of a test and connects to the next node or leaf.

- Pruning: This is a technique used to reduce the size of a decision tree by removing parts of the tree that do not provide power to classify instances. Pruning helps in reducing the complexity of the final classifier, and hence improves predictive accuracy by reducing overfitting.

- Depth of a Tree: The depth of a tree is the length of the longest path from the root node down to the farthest leaf node. In many algorithms, you can set the maximum depth to prevent the tree from becoming too complex.

- Splitting Criterion: This refers to the metric used to decide how the data at a node will be split. Common criteria include Gini impurity and entropy for classification trees, and variance reduction for regression trees.

- Information Gain: This is used to decide which feature to split on at each step in building the tree. Information gain measures how much "information" a feature gives us about the class.

- Binary Trees: These are a specific type of decision tree where each node has at most two children. This is common in many decision tree algorithms.

- Ensemble Methods: Techniques like Random Forests and Gradient Boosted Trees use multiple decision trees to improve predictive performance.



---

## When Does the Splitting Stop?

Determining when to stop splitting the nodes is crucial to avoid overfitting. 

- Pure Node:
   - When a leafe node is 100% of one class.  

- Increase in Impurity:
   - Stop splitting if the split does not decrease the impurity significantly.
  
- Maximum Depth:
   - Pre-set a maximum depth of the tree. Once this depth is reached, the tree stops growing.

- Minimum Samples for a Split:
   - Specify the minimum number of samples that a node must have to consider a split. If the number of samples is below this threshold, the node is not split.

- Minimum Samples per Leaf Node:
   - Set the minimum number of samples that a leaf node must have. This ensures that each leaf has a sufficient number of instances.

- Pruning:
   - After building a large tree, prune it back by removing branches that provide little predictive power. This can be done using techniques like cost complexity pruning (also known as weakest link pruning).


---
## How Does a Decision Tree Choose What Feature to Split On at Each Node?

Decision trees use a metric to choose the best feature to split on at each node. The goal is to find the feature and the split that will result in the most homogeneous sub-nodes (i.e., nodes with instances that are as similar as possible). 


#### The most common metrics used are:

- Gini Impurity:
  
   - Used in the CART (Classification and Regression Trees) algorithm, Gini impurity measures the frequency at which any element of the dataset will be mislabeled when it is randomly labeled according to the distribution of labels in the dataset.
   - A Gini impurity of 0 means all elements in the node belong to a single class.

- Entropy and Information Gain:
  
   - Entropy is a measure of disorder or uncertainty, and the goal of using entropy is to maximize the reduction in entropy after a split & this reduction is known as information gain.

- Variance Reduction:
  
   - Used in regression trees. The feature and split point that lead to the largest reduction in variance of the target variable are chosen.

#### Summary:

The algorithm will evaluate each feature and calculate the metric (like Gini impurity or information gain) for every possible split. The feature and split that result in the highest information gain (or lowest impurity) are chosen for the node.

# choosing a split on Discrete -> Classification:

![Screenshot 2024-01-20 at 1 06 36 PM](https://github.com/kasteway/Decision_Trees/assets/62068733/f73f9297-d6f3-4132-a05d-9373c12299a6) 

![Screenshot 2024-01-20 at 1 17 39 PM](https://github.com/kasteway/Decision_Trees/assets/62068733/23f359ba-0989-48af-86e3-e309350df107)


# choosing a split on Continuous -> Regression:

![Screenshot 2024-01-20 at 1 50 32 PM](https://github.com/kasteway/Decision_Trees/assets/62068733/33def2d5-ede2-411b-8f00-d9360bf158d8)

![Screenshot 2024-01-20 at 2 00 09 PM](https://github.com/kasteway/Decision_Trees/assets/62068733/72db9c21-8260-4ab3-a3b5-ada6630332d2)

--- 
## Other options of Trees:

1. Random Forest:
   - Random Forest is a collection of many decision trees.
   - Each tree makes its own prediction, and the final output of the Random Forest is decided based on the majority vote of all these trees for Classification & the average for Regression.
   - This makes the Random Forest a strong and reliable method in machine learning because it combines the decisions of many different models, reducing the chance of making a poor decision based on just one model's view.
     
3. Boosted Trees:
   - Boosting is a powerful ensemble technique in machine learning that combines multiple weak learners (typically decision trees) to form a strong learner. 
   - Boosting involves creating a series of decision trees, where each new tree tries to correct the mistakes of the previous ones.
   - Since each new tree is built to correct the mistakes of the previous ones, the final model (all the trees together) can be very good at making predictions, even for complex problems.
   - ### Popular Boosting Methods:
      - AdaBoost (Adaptive Boosting)
      - Gradient Boosting
      - XGBoost (eXtreme Gradient Boosting) *** Most popular ***
      - LightGBM (Light Gradient Boosting Machine)
      - CatBoost (Categorical Boosting)


---

## Random_Forests
** from sklearn.ensemble import RandomForestClassifier 

### Summary:


In machine learning, a "Random Forest" is like a large group of these friends, where each friend (or small decision tree) looks at the data (like restaurant options) and makes a decision (or prediction). However, instead of each friend deciding on their own, they all vote, and the restaurant with the most votes is chosen.

Therefore in this example:
| Who & What                  | Random Forest     |
|-----------------------------|-------------------|
| Groups of friends           | Decision Tree     |
| Restaurants                 | Predictions       |

So, a Random Forest is a collection of many decision trees. Each tree makes its own prediction, and the final output of the Random Forest is decided based on the majority vote of all these trees for Classification & the average for Regression. This makes the Random Forest a strong and reliable method in machine learning because it combines the decisions of many different models, reducing the chance of making a poor decision based on just one model's view.

##### Snapshot of Algorithm:
- is one of the most popular and most powerful machine learning algorithms
- can be used for both Regression & Classification
- Random Forest better than normal single decision trees. However, Gradient Boosting outperforms Random Forest
- Feature Importance - Which features appear most frequently in the decision tree root nodes
- Does both Bootstrapping(randomly selecting a subset of the data(rows from the data) with replacement for training each tree) & Feature Bagging(Train each tree on a random, different sample of features)
- is a special type of bagging ensemble incorperating a technique called RANDOM SUBSPACE method or FEATURE BAGGING to construct very different trees 
- Therefore, not all features will appear in all trees, which de-correlates the decision trees, helps prevent over focusing on certain features with high predicitve power**
- This allows for other features to have a greater opportunity to influence the decision trees
- This also means these other features can influence the model predictions





---

### Advantages & Disadvantages:

#### Advantages:
- Handles any data type, as well as missing data
- No scaling required
- Few hyperparameters to tune
- Handles curse of dimensionality
- Fast to train
- Well Suited for parallelization


#### Disadvantages:
- Less interpretable than decsion trees - Black Box
- Not as effective at reducing bias
- Other models may have better benchmarks


---

### How to Tune:


If we compare Random Forest to a single Decision Tree, there are several key hyperparameters that are different or unique to Random Forests. These hyperparameters help in controlling the behavior of the Random Forest model.

 **Number of Trees (n_estimators -> Default = 100):**
    This is perhaps the most important hyperparameter for a Random Forest. It specifies the number of trees in the forest. More trees usually mean better performance but also longer training time and chance of overfitting grows.

**Maximum Features (max_features  --> Default = 'auto'):**
   This parameter determines the maximum number of features that are considered for splitting a node. It can be set as a number, a percentage, or different heuristics like 'sqrt' or 'log2'. In a single Decision Tree,  usually, all features are considered for splitting a node, but in a Random Forest, limiting the number of features can lead to more diverse trees and reduce overfitting.
Recommend: Start with sqrt(number of features) then use a grid search for other possible values

**Bootstrap Samples[data rows] (bootstrap -> Default =TRUE):**
     This parameter decides whether or not to use bootstrap sampling when building trees. Bootstrap sampling means randomly selecting a subset of the data(rows from the data) with replacement for training each tree. This means, we are taking a subset of the features & a subset of the rows of data AKA Bootstrapped. This helps reduce correlation betwen trees because each tree is trained on a different subset rows of data & features which will likely better generalize.

**Out-of-Bag Error (oob_score -> Default =FALSE):**
     This is a method for estimating the generalization accuracy of the Random Forest. It uses the bootstrap samples not included in the training of each tree (the 'out-of-bag' samples) to estimate the model's performance. This is unique to     Random Forests and isn't a concept in a single Decision Tree. This will not impact the trees and only provides a way to measure the performance of the trees on the untrained data set similar to train/test split.

**Minimum Samples for Splitting (min_samples_split -> Default =2):**
    While this is also a hyperparameter for Decision Trees, it often plays a more crucial role in Random Forests because it affects each tree in the forest and thus has a compounded effect. This means that a node will be split if it contains 2 or more samples.

**Minimum Samples for a Leaf Node (min_samples_leaf -> Default =1):**
    This setting allows each leaf node to have as few as 1 sample.
 

---




---
## Boosting_on_Trees


### Summary:


Boosting is a powerful ensemble technique in machine learning that combines multiple weak learners (typically decision trees) to form a strong learner. Sometimes a single decision tree might not be very accurate because it can only consider a limited number of questions or scenarios. For instance, if you're trying to decide what to wear, your decision tree might have questions like "Is it raining?" or "Is it cold outside?" Each answer leads you down a different path to the final decision. It's like trying to decide on your entire day's schedule based only on the weather—it's helpful, but there's a lot more to consider.


Boosting involves creating a series of decision trees, where each new tree tries to correct the mistakes of the previous ones. It's like getting advice from a group of friends on what to wear. The first friend gives advice based on the weather, the next friend considers your choice and also adds advice based on what you're doing that day, and so on. Each friend's advice is focused on what the previous friends might have missed. The final decision (or prediction) in boosting isn't based on just one tree, but rather it's a combination of all the trees. Each tree has a say in the final decision, but trees that do a better job of correcting mistakes have a bigger say. It's like listening to all your friends' advice and then making a decision based on the most helpful suggestions.

Boosting can be very powerful. Since each new tree is built to correct the mistakes of the previous ones, the final model (all the trees together) can be very good at making predictions, even for complex problems. It's like having a team of experts where each expert learns from the mistakes of the others, making the team's overall decision-making process very effective.

##### Popular Boosting Methods:
- AdaBoost (Adaptive Boosting)
- Gradient Boosting
- XGBoost (eXtreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- CatBoost (Categorical Boosting)


Each of these methods has its strengths and is suited for different kinds of data and problems. Choosing the right one depends on the specific requirements of the task, such as the nature of the data, the size of the dataset, computational resources, and the specific problem being solved.






---

### Advantages & Disadvantages:

#### Advantages:
- High Accuracy: Boosting algorithms are known for their high accuracy, especially in complex data scenarios. They can capture complex patterns in data by combining multiple weak learners.
- Reduction of Bias and Variance: Boosting can reduce both bias and variance compared to single models. By sequentially focusing on incorrect predictions, boosting algorithms can adaptively improve on areas where a model is weak.
- Handling of Different Types of Data: Boosting algorithms can handle various types of data: numerical, categorical, binary, etc., and don’t necessarily require data pre-processing like normalization or dummy variable creation.
- Feature Importance Ranking: Most boosting algorithms inherently perform feature selection, providing a ranking of feature importance, which can be very insightful for model interpretation.
- Versatility: Boosting can be used for both classification and regression tasks and is generally effective across a wide range of applications and domains.



#### Disadvantages:
- Prone to Overfitting: If not carefully tuned, boosting models can overfit, especially on noisy datasets. The sequential nature of boosting means that it can keep adjusting to the noise in the data.
- Less Intuitive to Understand and Tune: The complexity of boosting algorithms makes them less intuitive than simpler models. They also have several hyperparameters (like the number of trees, learning rate, depth of trees, etc.) that require careful tuning to avoid overfitting and underfitting.
- Computationally Expensive: Building sequential models can be computationally expensive and time-consuming, particularly with large datasets.
- Poor Performance with High-Dimensional Sparse Data: In cases of high-dimensional sparse data (like text data), boosting algorithms might not perform as well as other algorithms like neural networks.
- Not Ideal for Real-time Predictions: Due to their complexity, boosting models can be slower to make predictions, which might not be ideal for applications requiring real-time responses.


---

### Which Boosting to use:


There are several boosting methods, each with its unique approach


######    AdaBoost (Adaptive Boosting): from sklearn.ensemble import AdaBoostClassifier or AdaBoostRegressor

- How it Works: AdaBoost starts with a simple decision tree and gives equal weight to all training examples. After each tree is built, it increases the weight of the examples that were misclassified, so the next tree focuses more on those difficult cases.
- Key Feature: The algorithm adapts by giving more focus to hard-to-classify instances, making it very effective for varied datasets.
- Usage: Commonly used in classification problems, especially where understanding the contribution of different features is important.

######    Gradient Boosting: from sklearn.ensemble import GradientBoostingClassifier or GradientBoostingRegressor

- How it Works: Gradient Boosting builds trees in a sequential manner, where each new tree is made to correct the errors of the previous ones. However, instead of adjusting weights like AdaBoost, it uses a gradient descent algorithm to minimize the loss (error).
- Key Feature: It’s more flexible than AdaBoost as it allows optimization of arbitrary differentiable loss functions.
- Usage: Widely used for both regression and classification tasks; highly effective but can be prone to overfitting.

######    XGBoost (eXtreme Gradient Boosting):

- How it Works: An advanced implementation of gradient boosting with better performance and speed. It includes several optimizations to increase efficiency and computational speed.
- Key Feature: Offers several advanced features like handling missing values, tree pruning, and regularization to avoid overfitting.
- Usage: Extremely popular in machine learning competitions and practical applications due to its performance and speed.

######    LightGBM (Light Gradient Boosting Machine):

- How it Works: Similar to other gradient boosting methods but more efficient. It uses a technique of Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) which makes it faster and requires less memory.
- Key Feature: Efficient with large datasets and can handle a high dimension of features.
- Usage: Ideal for scenarios where computational resources are a constraint.

######    CatBoost (Categorical Boosting):

- How it Works: A gradient boosting method optimized for categorical features. It transforms categorical values into numerical ones using various statistics on combinations of categorical features and their interaction with the target variable.
- Key Feature: Specifically designed to handle categorical data efficiently without extensive pre-processing.
- Usage: Highly effective for datasets with a large number of categorical features.


 

---

### Data:

The data set used for this analysis comes from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom). This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525).  Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.  This latter class was combined with the poisonous one.  The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

- Dataset Characteristics -> Multivariate

- Subject Area -> Biology

- Associated Tasks -> Classification

- Feature Type -> Categorical

- Instances -> 8124

- Features -> 22



---

### Tips:

- Boosting can be used for both Classification & Regression problems
- Feature Importance -> lists the most useful features from the dataset after running the alogrithm
- model_name.feature_importances_ ->
- model_name.feature_importances_.argmax() -> Which column is the most important
- X.columns[model_name.feature_importances_.argmax()] -> Get name of important column
- feats = pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['Importance'])



---

### Tips:

- AdaBoostClassifier [Scikit Learn AdaBoostClassifier]([https://archive.ics.uci.edu/dataset/73/mushroom](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
- base_estimator -> default is none meaning it will use "DecisionTreeClassifier" initialized with max_depth = 1 as the stump (if desired, another estimator can be manually selected)

---



