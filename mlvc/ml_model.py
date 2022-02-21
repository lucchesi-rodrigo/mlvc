"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from loguru import logger
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

"""
# TODO: 

- Review Project Code and map the instance methods with the project requirements 

Instance methods to be created to this lib:

1. REDO


- Write the code with the test and document function by function
- Each method
  - pep8 and pylint -> Friday first working version
"""
# SET constants in different value
class MlModel:

    def __init__(self, name):
        self.__name__ = name

    #  OK TESTED 
    def data_loading(self, df_path: str) -> pd.DataFrame:
        """
        Converts a csv file into a dataframe

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
            >>> df = load_data(df_path= 'path/file.csv')
        """
        try:
            self.df = pd.read_csv(df_path)
            logger.info(
                f'SUCCESS -> import_data({df_path}) -> '
                f'MSG -> Dataframe read successfully -> '
                f'OUTPUT -> df.head(): {self.df.head().to_dict()} .'
                )
            return self.df
        except FileNotFoundError as exc:
            logger.error(
                f'ERROR -> import_data({df_path}) -> '
                f'MSG -> Could not import data object ! -> '
                f'EXCEPTION -> {exc} .')
            raise
    
    #  OK TESTED 
    def data_statistics(self):
        """
        Perform data analysis into a data file

        Parameters
        ----------
        None

        Returns:
        --------
        stats_data: Dict
            Dictionary containing the data analysis information

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> stats = model.df_statistics()
        """
        try:
            self.stats_data = {
                'shape': self.df.shape,
                'null_values': self.df.isnull().sum(),
                'numeric_stats': self.df.describe()
            }
            logger.info(
                f'SUCCESS -> df_statistics() -> '
                f'MSG -> dataframe statistics callculated successfully -> '
                f'OUTPUT -> stats_data: {self.stats_data} .'
                )
            return self.stats_data
        except BaseException as exc:
            logger.error(
                f'ERROR: df_statistics() -> '
                f'MSG -> Could not perform data analysis process ! -> '
                f'EXCEPTION -> {exc} .'
                )
            raise
    
    #   OK TESTED
    def data_hist_plot(self,col_name: str) -> None:

        try:
            fig = self.df[col_name].hist()
            fig = fig.get_figure()
            fig.savefig(f'plots/hist_plot_{col_name}.pdf')
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
    
    #  OK TESTED
    def data_bar_plot(self,col_name: str, plot_type: str) -> None:
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

    def data_dist_plot(self,col_name:str) -> None:
        #sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        try:
            fig = sns.distplot(self.df[col_name])
            logger.info(
                f'SUCCESS: df_dist_plot(col_name= {col_name}) ->'
                f'msg : dataframe histogram created'
                f'output -> {fig}'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR: df_dist_plot(col_name= {col_name}) -> Exception: {exc}'
                )
            raise

    #  OK TESTED
    def data_heatmap_plot(self) -> None:
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
    
    def data_col_categoric_to_binary(self, target_name: str, col_name: str,condition: str) -> None:
        """
        Perform eda pipeline on df and save figures to images folder

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used in ml project

        Returns:
        --------
        None

        Examples:
        ---------
            >>> perform_eda(df)
        """
        #self.df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        try:
            self.df[target_name] = self.df[col_name].apply(lambda val: 0 if val == condition else 1)
            logger.info(
                f'SUCCESS: df_hist_plot() ->'
                f'msg : dataframe histogram created'
                f'output -> {self.df.columns.to_list()}'
                )
            return self.df
        except BaseException as exc:
            logger.error(
                f'ERROR: df_hist_plot() -> Exception: {exc}'
                )
            raise
    
    #  OK  but no unit-test
    def feature_encoder(self, category, target_col, response):
        """
        """
        category_lst = []
        try:
            category_groups = self.df.groupby(category).mean()[target_col]

            for val in self.df[category]:
                category_lst.append(category_groups.loc[val])

            self.df[f'{category}_{target_col}'] = category_lst
            logger.info(
                f''
                )
        except:
            logger.error(
                f''
                )
            raise
    
    #  OK  but no unit-test
    def build_ml_matrix(self, target_col:str, states: List):
        try:
            self.y = self.df[target_col]
            self.X = pd.DataFrame()
            self.X[states] = self.df[states]
            self.ml_data = {'X':self.X,'y':self.y}
            logger.info(
                f' SUCCESS ->'
                )
            return self.ml_data
        except:
            logger.error(
                f'ERROR ->'
                )
            raise
    
    # TODO : Test in workspace! -> Working?
    def test_train_data_dev(self, test_size: float, random_state: int):
        """
        """
        try:
            self.data_processed = train_test_split(
                self.X,
                self.y,
                test_size= test_size,
                random_state= random_state)

            self.X_train, self.X_test, self.y_train, self.y_test = self.data_processed
            logger.info(
                f' SUCCESS ->'
                )
            return self.data_processed
        except BaseException as exc:
            logger.error(f' ERROR -> {exc}')
            raise
    
    # TODO : Test in workspace! NOT working logistic regression
    def best_params(
        self,
        model_algorithm: Tuple, 
        param_grid: Dict, 
        folds: int, 
        grid_search: bool,
        best_estimator: bool
        ):
        """
        DOCS
        """
        try:

            model_algorithm = model_algorithm
            
            if grid_search:
                cv_model = GridSearchCV(
                    estimator=model_algorithm, 
                    param_grid=param_grid, 
                    cv= folds
                )

            cv_model.fit(self.X_train, self.y_train)

            if best_estimator:
                y_train_preds_cv_model = cv_model.best_estimator_.predict(self.X_train)
                y_test_preds_cv_model = cv_model.best_estimator_.predict(self.X_test)

            y_train_preds_cv_model = cv_model.predict(self.X_train)
            y_test_preds_cv_model = cv_model.predict(self.X_test)

            logger.info(classification_report(self.y_test, y_test_preds_cv_model))
            logger.info(classification_report(self.y_train, y_train_preds_cv_model))
        except BaseException as exc:
            logger.error('ERROR -> {exc}')
            raise

    # TODO : Test in workspace!      
    def tp_rate(self,model_1,model_2):
        try:
            # plots
            lrc_plot = plot_roc_curve(model_1, self.X_test, self.y_test)
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            rfc_disp = plot_roc_curve(model_2.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
            lrc_plot.plot(ax=ax, alpha=0.8)
            plt.show()
            logger.info('SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'ERROR -> {exc}')

    # TODO : Test in workspace!
    def saving(model_name,model):
        try:
            joblib.dump(model.best_estimator_, f'./models/{model_name}.pkl')
            logger.info(f'SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'Error -> {exc}')

    # TODO : Test in workspace!
    def loading(model_name):
        try:
            joblib.load(f'./models/{model_name}.pkl')
            logger.info(f'SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'Error -> {exc}')

    # TODO : Test in workspace!
    def output_explanation(self):
        try:
            explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            logger.info('SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'ERROR -> {exc}')

    # TODO : Test in workspace!
    def feature_importance(self):
        try:
            # Calculate feature importances
            importances = cv_rfc.best_estimator_.feature_importances_
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]

            # Rearrange feature names so they match the sorted feature importances
            names = [X.columns[i] for i in indices]

            # Create plot
            plt.figure(figsize=(20,5))

            # Create plot title
            plt.title("Feature Importance")
            plt.ylabel('Importance')

            # Add bars
            plt.bar(range(X.shape[1]), importances[indices])

            # Add feature names as x-axis labels
            plt.xticks(range(X.shape[1]), names, rotation=90);
            logger.info('SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'ERROR -> {exc}')

    # TODO : Test in workspace!
    def report(self):
        try:
            fig = plt.rc('figure', figsize=(5, 5))
            #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
            fig.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
            fig.text(0.01, 0.05, str(classification_report(self.y_test, self.y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
            fig.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
            fig.text(0.01, 0.7, str(classification_report(self.y_train, self.y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
            fig.axis('off');
            fig.savefig(f'plots/model_report.pdf')
            logger.info('SUCCESS -> ')
        except BaseException as exc:
            logger(f'ERROR -> {exc}')
            raise


