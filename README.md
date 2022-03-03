# ML Version control Beta

Automatic machine learning tool

# To-Do ...
1. ~feature importance method~
2. ~classification report with matplotlib~
3. ~roc_curve (FP analysis)~
4. ~tuning must be changed to (split_data -> fit_predict -> best_estimator) with grid search~
5. Unit-test all methods
6. Review Docstring PEP8
7. Test in workspace to find execution erros
 7.1 Maybe the model_data -> model would not work in execution,
    Check how to store this fitted model in a variable

## System design

![ML Version control brainstorming](static/project_design.png "ML Version control")

---
## Design

## Instances

## Methods

## Example

### CreateMlModel

```python
from mlvc.ml_model import MlModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ml_model = MlModel('udacity_ml_devops_first_project')
```

```python
STATES = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
    'Income_Category_Churn', 'Card_Category_Churn'
    ]
```

```python
df = ml_model.data_loading(df_path=r"./data/bank_data.csv")
df.head()
```

```python
stats_data = ml_model.data_statistics()
stats_data['numeric_stats']
```

```python
ml_model.data_categoric_to_binary(
    target_name= 'Churn', 
    col_name= 'Attrition_Flag', 
    base_value= "Existing Customer"
    )
```

```python
ml_model.data_hist_plot(col_name='Churn')
```

```python
ml_model.data_hist_plot(col_name='Customer_Age')
```

```python
ml_model.normalized_data_plot(col_name='Marital_Status',plot_type='bar')
```

```python
ml_model.data_dist_plot(col_name='Total_Trans_Ct')
```

```python
ml_model.data_heatmap_plot()
```

```python
ml_model.data_feature_encoder(col_name='Gender',target_col='Churn')
```

```python
ml_model.data_feature_encoder(col_name='Education_Level',target_col='Churn')
```

```python
ml_model.data_feature_encoder(col_name='Marital_Status',target_col='Churn')
```

```python
ml_model.data_feature_encoder(col_name='Income_Category',target_col='Churn')
```

```python
ml_model.data_feature_encoder(col_name='Card_Category',target_col='Churn')
```

```python
ml_model.data_build_ml_matrix(target_col='Churn', states_key=STATES)
```

```python
data_proccessed = ml_model.split_test_train_data(test_size=0.3, random_state=42)
```

```python
lrc = ml_model.tuning(
    model_name = 'logistic_regression',
    model_algorithm = LogisticRegression(), 
    param_grid = None, 
    folds = None, 
    grid_search = False,
    best_estimator = False
)
```

```python
rfc = ml_model.tuning(
    model_name = 'random_forest_classifier',
    model_algorithm = RandomForestClassifier(random_state=42), 
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }, 
    folds = 5, 
    grid_search = True,
    best_estimator = True
)
```

```python
ml_model.tp_rate_analysis(models=(lrc,rfc))
```

```python
ml_model.saving(model=lrc)
ml_model.saving(model=rfc)
```

```python
lrc = ml_model.loading(path_to_model='./models/logistic_regression.pkl')
rfc = ml_model.loading(path_to_model='./models/random_forest_classifier.pkl')
```

...

## Contact

http://www.linkedin.com/in/rodrigo-lucchesi/





  


