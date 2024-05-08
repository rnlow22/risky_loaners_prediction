# Risky Loaners: Project Overview
- The objective of this project is to identify potential risky loaners, whom having high loan risk from the loan application. 
- The goal of predicting loans application with high risk is to avoid or minimize financial losses to lenders, to allow lenders to assess the likelihood of borrowers defaulting on their loans.

![Screenshot 2024-05-08 at 4 56 46 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/44ef8b6f-5b97-44b5-9120-0df31252c540)

## References:
**Python Version:** 3.10 <br />
**Packages:** numpy, pandas, seaborn, matplotlib, Scikit-Learn, StatsModels, Scientific Python (SciPy) <br />
**Slide:** [Risky Loaners Model Slides.pdf](https://github.com/rnlow22/risky_loaners_prediction/blob/main/Risky%20Loaners%20Model%20Slides.pdf) <br />
**ipynb File:** [Risky Loaners Model.ipynb](https://github.com/rnlow22/risky_loaners_prediction/blob/main/Risky%20Loaners%20Model.ipynb) <br />

# Feature Engineering
### Target Variable
**loanStatus** is used to defined the target variable, whereby:
1. `target == 1` : **Risky Loaners**, Loans that carry a risk of default or are unlikely to be repaid according to the agreed term and predefined schedule. This leads to the possibility of cost incurred on third parties (i.e., Administrator, Legal fees) for late payments. <br />(`loanStatus in ['Charged Off Paid Off','External Collection','Internal Collection','Settled Bankruptcy','Charged Off']`)
2. `target == 0` : **Non-Risky Loaners**, Loans that are paid according to the agreed term and predefined schedule as set by the lender. <br /> (`loanStatus in ['Paid Off Loan','Settlement Paid Off']`)

### Numerical Variable
To predict the loan application with high risk, it is important to look into several areas, including credit score of the borrowers, by analyzing the borrower's creditworthiness based on their credit history.

From the data provided, _payment.csv_ and _loan.csv_ are the best data option we can analyze the historical payment behaviour. Hence, new features are created based on business understanding from _payment.csv_ and _loan.csv_:

1. **Previous loan Count:** The number of previous funded loan
2. **Previous Bad Loan Count:** The number of previous funded loan that has turned into bad loan
3. **Previous Paid Off Loan Count:** The number of previous funded loan that has been paid off
4. **Difference between full payment and Originally Scheduled Payment Amount:** The difference in term of the full payment made to lenders and the Originally Scheduled Payment Amount for the previous loan
5. **Existing Debt Amount (RM):** The total amount of debt based on the previous funded loan that has not been paid off
6. **Previous Payment is Collection:** The number of payment that is collection for the previous loan
7. **Number of Success Payment Made:** The number of Success payment made for the previous loan
8. **Number of Failed Payment Made:** The number of Failed payment made for the previous loan
9. **Median Payment Amount (RM) of Success Payment Made:** The median Payment Amount (RM) of Success Payment Made for the previous loan
10. **Median Payment Amount (RM) of Failed Payment Made:** The median Payment Amount (RM) of Failed Payment Made for the previous loan
11. **Minutes from Application to Originiated (mins):** The minutes differences from Originated Datetime and Application Datetime
12. **Ratio of Failed over Success Payment:** Number of Failed Payment divided by Number of Success Payment and Small Value

### Categorical Variable
Furthermore, feature engineering has also been applied to the columns **inquiryonfilecurrentaddressconflict**, **morethan3inquiriesinthelast30days**, **inquirycurrentaddressnotonfile** using the following conditions:
- Replacing True to 1
- Replacing False to 0
- Imputing missing values with 0

Besides that, feature engineering has also been applied to the columns **overallmatchresult**, **nameaddressmatch** using the following conditions:
- Replacing invalid to -1
- Replacing other to -1
- Replacing unavailable to -1
- Imputing missing values with -1
- Replacing mismatch to 0
- Replacing partial to 0.5
- Replacing match to 1

# Data Transformation
In this section, the following was carried out:

1. Normality Test:
    - $𝐻_0$ : Numerical feature is normally distriuted
    - $𝐻_1$  : Numerical column is not normally distributed. <br />
    where if  $𝑝<0.05$ , it is statistically significant to reject the null hypothesis. Hence, there is sufficient evidence to conclude that numerical feature is not normally distributed.

2. From the above test, numerical features that are not normal are identified and transformed with the following steps (these steps are only applicable to in-time training data):
    1. The features are being scaled (i.e., Standard Scaler).
    2. Then, features are being scaled via Min Max Scaler which would restrict range of the numerical feature between [0, 1]
    3. The features are then further transformed via Yeo-Johnson Transformation so that the resulting features will be more normally distributed.
    
3. Out-time validation data will be transformed with the scaler trained from step [2]. The transformed data will be capped between [0,1]

Both non-normal features and transformed features are kept for model training performance comparison in the next section.

Example of the data Transformation:
![Screenshot 2024-05-08 at 6 07 21 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/b6d8a29f-0c47-4831-ad6c-f0879839ddbe)

# Exploratory Data Analysis
## Understanding Linear Relationship between Independent vs Target Variable:
As a high level exploratory study of the dataset, it is helpful to understand whether there is any different behaviour patterns between different group of loaners (i.e., Non-First Loaners and First Loaners): 

### non-Normal Datasets
![image](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/3d0713ce-b24a-4554-819e-a3ad078d491e)

### Transformed Datasets
![image](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/03934fb1-cf39-427b-b238-667f52857c54)

Based on the above observations, for both non-Normal and transformed datasets, the heat map shows a different characteristics in first loan vs non-first loan analysis, for example:
1. **Annual Percentage Rate** has higher correlation with the target in first loan data sample as compared to non-first loan data sample
2. **Number of Failed Payment** is in the top 7 features (top 8 for transformed features) that is correlated with target in the non-first loan data sample, however this is not applicable to first loan data sample.

Hence, here is the proposed model building strategy:
1. To use non-linear models such as tree models instead of linear models where the trees is able to split based on the distinctive behaviour of data samples by different categories (Perhaps First loan vs non-first loan).
2. To build separate linear models for first loan and non-first loan. However, we have observed that only 3,057 (~ 12%) data is capturing the non-first loan. This means that there are insufficient data for the proposal model buiding strategy.

For experimental purposes, linear model such as Logistic Regression will be included in the baseline model performance comparison.

## Understanding  Correlation between Independent Variables:
There are 6 pairs of highly correlated independent variables identified from Correlation Analysis:
![Screenshot 2024-05-08 at 7 09 13 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/fa892209-6c1d-41e2-99b0-5ac0fdc5461e)

Highly correlated independent variables are removed to prevent multicollinearity for model building, using the following 2 approaches:
![Screenshot 2024-05-08 at 7 11 41 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/cc3a4aae-9123-497f-b918-6c0385b592c5)

- Approach A is employed during the first phase of the model experiment, wherein the performances of various algorithm types with default parameters are compared.
- Approach B is implemented after finalizing tree models as the chosen algorithm in the first phase of the model experiment.

# Model Building
- The dataset is divided into the following samples:
    - In-time Train data samples: A random split of 80% of the data occurring before Year 2017
    - In-time Test data samples: A random split of 20% of the data occurring before Year 2017
    - Out-time Validation data samples: All data from Year 2017
- The purpose of monitoring Out-time Validation is to ensure the robustness of the model performance and avoid finalizing model that is overfitting.
- Recall and AUC have been designated as the primary and secondary performance indicators throughout the entire model development process in this project, supported by the following rationale:
    - Recall: Recall is selected as the primary performance indicator because the project's objective is to accurately identify all instances of true risky loan applicants. The model serves as a reference for business stakeholders to screen loan applications before approving and funding loans.
    - AUC: AUC is chosen as the secondary indicator due to its importance in developing a model capable of effectively distinguishing between positive and negative instances across all potential thresholds.

### End-to-End Model Development Flow Chart
![Screenshot 2024-05-08 at 8 45 29 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/6b37889d-c3ad-46ab-b61a-2bfa8ffcbaeb)

# Post Model Analysis
![image](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/f3b01f4d-d0b6-44fa-b5c4-5ff7a005dc84)

From the above feature importances chart, the top 5 important predictors are:
1. Clear Fraud Score
2. Originally Scheduled Payment Amount (Transformed)
3. Annual Percentage Rate (%) (Transformed)
4. Minutes difference between application datetime to originated datetime
5. Number of unique inquiries for the consumer seen by Clarity in the last 365 days

### Analysis of Data Distribution Plot
![Screenshot 2024-05-08 at 8 50 55 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/eee1f4f8-fff3-4137-80a6-57d4591501d7)

### Analysis of Partial Dependence Plot
![Screenshot 2024-05-08 at 8 52 29 PM](https://github.com/rnlow22/risky_loaners_prediction/assets/30455582/b0535af5-bded-463e-b5ca-a7179fcaa314)


# Business Impact evaluated using Out-time Validation - Year 2017 Q1


# Future Enhancement
