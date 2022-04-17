# Constraint-Fairness-to-Logistic-Regression
Classification Mechanism towards Fairness Constraints

### Abstract
Algorithmic methods are widely applied in various fields to support decision-making in recent years. However, while algorithms aim to make rational decisions that use all necessary information possible, some sensitive information (e.g. gender, race etc.) included may lead to unfairness by hurting certain groups of people who obtain one or more sensitive features. This project aims to study the measurement of fairness on convex margin classifiers with a novel measure technique of decision boundary fairness that is proposed in the paper Fairness Constraints: Mechanisms for Fair Classification [(Zafar et al, 2017)](https://arxiv.org/abs/1507.05259) to avoid unfair decision making in two notions: disparate treatment and disparate impact. To show the generality of the technique, logistic regression model is used as the convex margin based classifier and the experiment is conducted on two real-world datasets: _Adult Income Dataset_ and _US Census Data_ to discuss the extent of fairness in income level and recruitment process, respectively. Four experimental groups with different level of control for fairness and accuracy are identified in the paper. Experimental result shows that the logistic regression classifier yields satisfying accuracy without reverse discrimination when set its objective to maximize accuracy subject to fairness constraints.



### Description 
#### General code
_utils.py_: Includes all the helper functions used to train model, generate related plots and summarize results 

_loss_function.py_: Contains the loss function initiation and calculation behinds the scene.


#### 1) Reproducing results in paper [(Zafar et al, 2017)](https://arxiv.org/abs/1507.05259) 

**To run the mechanism, use the command _python3 demo_constraints.py_ in terminal**

_prep_adult_data.py_: Read data from the online open source dataset, load data into features set (X), label set (y) and sensitive attributes (x_control)

_demo_constraints.py_: The main file used to trigger model traning and perform experiments with different constraints set.


#### 2) Validating the mechanism with new dataset [US_Census_data](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990))

**To run the mechanism, use the command _python3 new_data_demo.py_ in terminal**

_prep_demo_data.py_: Perform data pre-processing (such as one-hot feature data, generate binary labels), load data into features set (X), label set (y) and sensitive attributes (x_control)

_new_data_demo.py_: The main file used to trigger model traning for new dataset and perform experiments with different constraints.

_MIE424Project_USCensus_Pandas.ipynb_: Data cleaning code

_USCensus1990_clean_iClass.csv_: Cleaned data (by _MIE424Project_USCensus_Pandas.ipynb_)

_USCensus1990raw.attributes.txt_: Data description downloaded with the data files. 


#### img folder
This folder contains all the plots generated through the experiments, _demo_org_x.png_ are plots generated with the _Adult_Income_data_, and _demo_new_x.png_ are plots for new dataset. 


### Reference 
[Fairness Constraints: Mechanisms for Fair Classification](https://github.com/wnstlr/fair-classification) 

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi.

20th International Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, April 2017.
