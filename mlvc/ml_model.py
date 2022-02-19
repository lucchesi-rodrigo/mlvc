"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string


# import libraries
# from operator import methodcaller
# import os
# import logging as log
# from winreg import REG_RESOURCE_REQUIREMENTS_LIST
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# import churn_library_solution as cls
# import matplotlib.pyplot as plt
# import seaborn as sns

"""
# TODO: 

- Review Project Code and map the instance methods with the project requirements 

Instance methods to be created to this lib:

1. read_csv
2. statistics
3. encode to binary categorical
4. histogram target, column
5. bar_plot df col normalized
6. dist_plot col
7. heatmap sns correlation matrix
8. define hypothesy matrix and target matrix
9. encode columns with groupby ... understand groupby
10. train_test_split
11. model tunning with GridSearch -> get params and model
12  plot_roc_curve ?
13. save best model with joblib
14. open and test model
15. shap.TreeExplainer(cv_rfc.best_estimator_)
16. feature importance with bar plot  
17. Report model tunning


- Write the code with the test and document function by function
- Each method
  - pep8 and pylint -> Friday first working version
"""

# log.basicConfig(
#     filename='./logs/churn_library.log',
#     level = logging.INFO,
#     filemode='w',
#     format='%(name)s - %(levelname)s - %(message)s')
# #Add log external call config
class MlModel:
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn']

    def __init__(self, name):
        self.__name__ = name
        

    # OK
    def load_data(self, df_path: str) -> pd.DataFrame:
        """
        Returns from a valid path a dataframe from a csv file

        Parameters
        ----------
        df_path: str
            Path to the csv file

        Returns:
        --------
        df: pd.DataFrame
            A pandas dataframe

        Examples:
        ---------
            >>> df = load_data('path/file.csv')
        """
        try:
            self.df = pd.read_csv(df_path)
            logger.info(
                f'SUCCESS: import_data({df_path}) -> msg: dataframe read successfully -> df:' f'{self.df.head().to_dict()}'
                )
            return self.df
        except FileNotFoundError as exc:
            logger.error(f'ERROR: import_data({df_path}) -> Exception: {exc}')
            raise
    
    # OK
    def df_statistics(self):
        """
        Perform data analysis with pandas methods like describe,
        isnull and shape

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used in ml project

        Returns:
        --------
        self.stats_data: Dict
            Dictionary containing the statistics info

        Examples:
        ---------
            >>> model = mlvc.MLVersionControl('test')
            >>> stats = model.df_statistics()
        """
        try:
            self.stats_data = {
                'shape': self.df.shape,
                'null_values': self.df.isnull().sum(),
                'numeric_stats': self.df.describe()
            }
            logger.info(
                f'SUCCESS: df_statistics() ->'
                f'msg : dataframe statistics callculated successfully'
                f'output -> stats_data : {self.stats_data} '
                )
            return self.stats_data
        except BaseException as exc:
            logger.error(
                f'ERROR: df_statistics() -> Exception: {exc}'
                )
            raise
    
    # OK
    def df_hist_plot(self,col_name: str) -> None:

        try:
            fig = self.df[col_name].hist()
            logger.info(
                f'SUCCESS: df_hist_plot(col_name={col_name}) ->'
                f'msg : dataframe histogram created'
                f'output -> {fig}'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR: df_hist_plot(col_name={col_name}) -> Exception: {exc}'
                )
            raise
    
    # OK
    def df_bar_plot(self,col_name: str, plot_type: str) -> None:
        try:
            fig = self.df[col_name].value_counts('normalize').plot(kind=plot_type);
            logger.info(
                f'SUCCESS: df_hist_plot(col_name={col_name}) ->'
                f'msg : dataframe histogram created'
                f'output -> {fig}'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR: df_hist_plot(col_name={col_name}) -> Exception: {exc}'
                )
            raise

    def df_heatmap_plot(self) -> None:
        #sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        try:
            fig = sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
            logger.info(
                f'SUCCESS: df_hist_plot() ->'
                f'msg : dataframe histogram created'
                f'output -> {fig}'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR: df_hist_plot() -> Exception: {exc}'
                )
            raise
    # def encode_col(self, target_name: str, col_name: str,condition: str) -> None:
    #     """
    #     Perform eda pipeline on df and save figures to images folder

    #     Parameters
    #     ----------
    #     df: pd.DataFrame
    #         Dataframe to be used in ml project

    #     Returns:
    #     --------
    #     None

    #     Examples:
    #     ---------
    #         >>> perform_eda(df)
    #     """
    #     #self.df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    #     try:
    #         self.df[target_name] = self.df[col_name].apply(lambda val: 0 if val == condition else 1)
    #         log.info('')
    #     except:
    #         log.error('')
    #         raise






    # def encoder_helper(self, category, target_col, response):
    #     """
    #     """
    #     category_lst = []
    #     try:
    #         category_groups = self.df.groupby(category).mean()[target_col]

    #         for val in self.df[category]:
    #             category_lst.append(category_groups.loc[val])

    #         self.df[f'{category}_{target_col}'] = category_lst
    #         log.info('')
    #     except:
    #         log.error('')
    #         raise
    
    # def target_instance_matrix(self, target_col:str, states: list()):
    #     try:
    #         self.y = self.df[target_col]
    #         self.X = pd.DataFrame()
    #         self.X[states] = self.df[states]
    #         self.ml_data = {'X':self.X,'y':self.y}
    #         log.info('')
    #         return self.ml_data
    #     except:
    #         log.error('')
    #         raise

    # def train_test_split(self,test_size: float, random_state:int)):
    #     """
    #     """
    #     try:
    #         self.data_processed = train_test_split(
    #             self.X,
    #             self.y,
    #             test_size= 0.3,
    #             random_state=42)

    #         self.X_train, self.X_test, self.y_train, self.y_test = self.data_processed
    #         log.info('')
    #         return self.data_processed
    #     except:
    #         log.error('')
    #         raise

    # def best_models(self,model_algorithm: dict, param_grid: dict, folds:int):
        
    #     try:
    #         model = model_algorithm.value()

    #         cv_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds)
    #         cv_model.fit(self.X_train, self.y_train)

    #         y_train_preds = cv_model.best_estimator_.predict(self.X_train)
    #         y_test_preds = cv_model.best_estimator_.predict(self.X_test)
        
    #         self.test_report = classification_report(self.y_test, y_test_preds)
    #         self.train_report = classification_report(self.y_train, y_train_preds)
    #         log.info('')
    #         return self.test_report, self.train_report
    #     except:
    #         log.error('')
    #         raise

    # def model_performance(self):
    #     # Check book pipeline names
    #     # scores


    #     print('logistic regression results')
    #     print('test results')
    #     print(classification_report(y_test, y_test_preds_lr))
    #     print('train results')
    #     print(classification_report(y_train, y_train_preds_lr))

    # def classification_report_image(
    #     self,y_train,
    #     y_test,
    #     y_train_preds_lr,
    #     y_train_preds_rf,
    #     y_test_preds_lr,
    #     y_test_preds_rf):
    #     """
    #     produces classification report for training and testing results and stores report as image
    #     in images folder
    #     input:
    #             y_train: training response values
    #             y_test:  test response values
    #             y_train_preds_lr: training predictions from logistic regression
    #             y_train_preds_rf: training predictions from random forest
    #             y_test_preds_lr: test predictions from logistic regression
    #             y_test_preds_rf: test predictions from random forest

    #     output:
    #             None
    #     """
    #     pass
    
    # def feature_importance_plot(self,model, X_data, output_pth):
    #     '''
    #     creates and stores the feature importances in pth
    #     input:
    #             model: model object containing feature_importances_
    #             X_data: pandas dataframe of X values
    #             output_pth: path to store the figure

    #     output:
    #             None
    #     '''
    #     pass

    # def train_models(self,X_train, X_test, y_train, y_test):
    #     '''
    #     train, store model results: images + scores, and store models
    #     input:
    #             X_train: X training data
    #             X_test: X testing data
    #             y_train: y training data
    #             y_test: y testing data
    #     output:
    #             None
    #     '''
    #     pass

