# mlvc

This Python library seeks easier machine learning steps to develop a production model for commercialization for example. 

It uses [scikit-learn](https://scikit-learn.org/stable/) as a machine learning library, [pandas](https://pandas.pydata.org/) for data analysis and manipulation tool, [seaborn](https://seaborn.pydata.org/) for data visualization as well and [gitpython](https://gitpython.readthedocs.io/en/stable/) for version control.

The idea to start working on this project was taken from the [udacity](https://www.udacity.com/)  nanodegree: Machine learning DevOps; a tool to ml production development [deployment environment](https://en.wikipedia.org/wiki/Deployment_environment). 

Therefore the framework architecture is based on three pillars in a Machine Learning pipeline:

- Modeling -> class MlModeling ->  It has the necessary methods for machine learning pipelines. It contains tasks necessary methods for EDA and pre-process stage, training & predicting. Also, produce graphical information from those as reports or plots.
- Control -> MlController -> Not implemented -> Idea: Act as controller on machine learning models, targeting better accuracy giving relaibility to what is being developed. 
- Versioning -> MlVersioning ->  Not implemented -> Idea: Act as a version control tool as Git is for any software. However, acting more precisely to machine learning. 
- Communication -> MlApis -> Not implemented -> Idea: Act as a plataform to expose easily the developed models in Rest APIs.
---

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





  


