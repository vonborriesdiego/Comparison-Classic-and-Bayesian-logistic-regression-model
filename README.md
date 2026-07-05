# Accuracy Comparison of Classical and Bayesian Logistic Regression Models
## Abstract
This project focus on the accuracy comparison between the classci and Bayesian logistic regression methods in Python to identify clinical factors associated with ICU admission in pediatric patients with candidemia. The analysis combines predictive modeling and posterior inference to estimate which covariates are most strongly associated with ICU transfer.

## Objective

The main goal of this project is to model the probability of ICU admission using clinical and treatment-related variables, and to identify which factors are most strongly associated with increased ICU odds.

## Dataset

The dataset contains 98 pediatric candidemia cases with demographic, clinical, treatment, and outcome-related variables. The target variable is ICU admission.

## Methodology

The workflow was implemented entirely in Python using two approaches, one followed a classical approach while the other followed a Bayesian approach:

* Data preprocessing and feature preparation in Pandas and NumPy
* Bayesian logistic regression modeling in PyMC
* Posterior analysis and diagnostics with ArviZ
* Model evaluation and classification metrics with scikit-learn
Visualization with Matplotlib

The analysis included:

* posterior summaries and credible intervals
* posterior odds ratios for interpretation
* trace plots and R-hat diagnostics to assess convergence
* posterior predictive checks for calibration
* separation plots to evaluate discrimination
* Pareto k / LOO diagnostics to detect influential observations
* confusion matrix, accuracy, recall, precision, F1-score, and AUC to assess predictive performance
Key Results

Across both Bayesian models, the same two variables showed the strongest and most consistent association with ICU admission:

Mechanical Ventilation
Septic Shock (use of amines)

These variables had posterior credible intervals above zero, indicating strong evidence of a positive association with ICU admission risk.

# Main findings
Across both analysis, the model that showed a better performace where the ones with the bayesian approach. The main findings are described as follows:

* Mechanical Ventilation was associated with a substantial increase in the odds of ICU admission.
* Septic Shock showed the strongest association with ICU transfer, with a markedly higher posterior odds ratio.
* Predictive performance

<img width="585" height="517" alt="Screen Shot 2026-07-05 at 6 20 10 PM" src="https://github.com/user-attachments/assets/3db7d422-f59e-482d-adcd-4d53c93523bb" />

The model showed good predictive performance on this dataset:

Accuracy: 82.7%
AUC: 0.80
Recall for ICU admissions: 88.7%

These results suggest that the model has good discriminative ability and is effective at identifying high-risk patients, which is particularly valuable in a clinical setting.

# Conclusion

This project shows how Bayesian logistic regression can be used not only as a predictive tool, but also as an interpretable framework for clinical risk analysis. The results suggest that mechanical ventilation and septic shock are the strongest predictors of ICU admission in this pediatric candidemia cohort, while the Bayesian framework provides a principled way to quantify uncertainty in the estimated effects.

# Future Work

* External validation on a larger cohort to evaluate generalizability and robustness of the model.
* Regularized or hierarchical Bayesian models to improve stability in small-sample clinical datasets with many predictors.
* Comparison with classical machine learning models such as penalized logistic regression, random forests, or gradient boosting for predictive benchmarking.
