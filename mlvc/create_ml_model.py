"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string
import json
from lib2to3.pytree import Base
import os
import json
from datetime import datetime,date
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from typing import Dict,List,Tuple

from loguru import logger
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



class MlModel:
    """Class to process Machine Learning Models"""

    def __init__(self, name):
        """Init method which has as input the model instance name"""
        self.__name__ = name
 
    def data_loading(self, df_path: str) -> Dict:
        """
        Loads a csv file into a dataframe

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        df_path: str
            Path to the csv file

        Returns:
        --------
        df: pd.DataFrame
            A pandas dataframe

        Examples:
        ---------
            >>> model = mlvc.CreateCreateMlModel('test')
            >>> model.data_loading(df_path= 'path/file.csv')
        """
        try:
            self.df = pd.read_csv(df_path)
            logger.info(
                f'SUCCESS -> data_loading({df_path}) -> '
                f'MSG -> CSV file loaded successfully -> '
                f'OUTPUT -> df.head(): {self.df.head().to_dict()} .'
                )
            return self.df
        except FileNotFoundError as exc:
            logger.error(
                f'ERROR -> data_loading({df_path}) -> '
                f'MSG -> Could not import data object ! -> '
                f'OUTPUT -> None'
                f'EXCEPTION -> {exc} .')
            raise
       
    def data_analysis(self) -> Dict:
        """
        Perform data analysis in a dataframe

        Parameters:
        ----------
        self: CreateMlModel
            Create model object data

        Returns:
        --------
        stats_data: Dict
            Dictionary containing the data analysis information

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading(df_path= 'path/file.csv')
            >>> stats_data = model.data_analysis()
        """
        try:
            self.stats_data = {
                'shape': self.df.shape,
                'null_values': self.df.isnull().sum(),
                'numeric_stats': self.df.describe()
            }
            logger.info(
                f'SUCCESS -> data_analysis() -> '
                f'MSG -> Data analysis calculated successfully ! -> '
                f'OUTPUT -> stats_data: {str(self.stats_data)} .'
                )
            return self.stats_data
        except BaseException as exc:
            logger.error(
                f'ERROR -> data_analysis() -> '
                f'MSG -> Could not create statistics calculation ! -> '
                f'OUTPUT -> None'
                f'EXCEPTION -> {exc} .')
            raise

    def isolate_categ_and_num_cols(self) -> Tuple[List[str],List[str]]:
        """
        Isolate the categoric and numeric cols from pandas DataFrame

        Parameters:
        ----------
        self: CreateMlModel
            Create model object data
        
        Returns:
        --------
        numeric_cols: List[str]
            Numeric columns
        categoric_cols: List[str]
            Categoric columns

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> df = load_data(df_path= 'path/file.csv')
            >>> numeric_cols, categoric_cols = model.isolate_categ_and_num_cols()
            >>> numeric_cols
                ['x1','x2',...]
        """
        try:
            self.numeric_cols = [
                column for column in self.df.columns if self.df[column].dtype != 'object'
                ]
            self.categoric_cols = [
                column for column in self.df.columns if self.df[column].dtype == 'object'
                ]
            logger.info(
                f'SUCCESS -> isolate_categ_and_num_cols() -> '
                f'MSG -> Isolated df numeric and categoric columns ! -> '
                f'OUTPUT -> numeric_cols: {self.numeric_cols} , categoric_cols: {self.categoric_cols} .'
            )
            return self.numeric_cols, self.categoric_cols
        except BaseException as exc:
            logger.error(
                f'SUCCESS -> isolate_categ_and_num_cols() -> '
                f'MSG -> Could not isolate df numeric and categoric columns ! -> '
                f'OUTPUT -> None .'
                f'EXCEPTION -> {exc}'
            )
            raise 
  
    def data_hist_plot(self,col_name: str) -> None:
        """
        Create a histogram plot from a dataframe column

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        col_name: str
            Column name to generate the histogram plot from

        Returns:
        --------
        fig: matplotlib
            The fig's module to map the method execution to
            unit-testing

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_hist_plot(col_name= 'X')
        """
        try:
            fig = self.df[col_name].hist();
            fig = fig.get_figure()
            fig.savefig(f'plots/histograms/hist_plot_{col_name}.pdf')
            logger.info(
                f'SUCCESS -> data_hist_plot(col_name={col_name}) -> '
                f'MSG -> Dataframe histogram plot created ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig.__module__
        except BaseException as exc:
            logger.error(
                f'ERROR -> data_hist_plot(col_name={col_name}) ->'
                f'MSG -> Could not create dataframe histogram ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def normalized_data_plot(self,col_name: str, plot_type: str) -> None:
        """
        Create a chosen plot from a normalized pandas series

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        col_name: str
            Column name to generate the histogram plot from
        plot_type: str
            Plot type

        Returns:
        --------
        fig: matplotlib
            The fig's module to map the method execution to
            unit-testing

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.normalized_data_plot(col_name= 'X', plot_type= 'bar')
        """
        try:
            fig = self.df[col_name].value_counts('normalize').plot(kind=plot_type);
            fig = fig.get_figure()
            fig.savefig(f'plots/{plot_type}_plot_{col_name}.pdf')
            logger.info(
                f'SUCCESS -> normalized_data_plot(col_name={col_name} ,plot_type= {plot_type}) -> '
                f'MSG -> Created Pandas series plot {plot_type} ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig.__module__
        except BaseException as exc:
            logger.error(
                f'ERROR -> normalized_data_plot(col_name={col_name} ,plot_type= {plot_type}) -> '
                f'MSG -> Could not create Pandas series plot {plot_type} ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def data_dist_plot(self,col_name:str) -> None:
        """
        Create a dist plot from a pandas series using seaborn backend

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        col_name: str
            Column name to generate the dist plot from

        Returns:
        --------
        fig: matplotlib
            The fig's module to map the method execution to
            unit-testing

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_dist_plot(col_name= 'X')
        """
        try:
            fig = sns.distplot(self.df[col_name]);
            fig = fig.get_figure()
            fig.savefig(f'plots/distplots/distplot_plot_{col_name}.pdf')
            logger.info(
                f'SUCCESS -> data_dist_plot(col_name={col_name}) -> '
                f'MSG -> Created Pandas series dist plot ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig.__module__
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_dist_plot(col_name={col_name}) -> '
                f'MSG -> Could not create Pandas series dist plot ! ->'
                f'Exception -> {exc} .'
                )
            raise

    def data_heatmap_plot(self, color_pallette:str ='Dark2_r') -> None:
        """
        Create a heatmap plot from a pandas correlation matrix 
        using seaborn backend

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        color_pallette: str
            Color pallette to be used on the heatmap plot,
            default value: Dark2_r

        Returns:
        --------
        fig: matplotlib
            The fig's module to map the method execution to
            unit-testing

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_heatmap_plot()
        """
        try:
            fig = sns.heatmap(self.df.corr(), annot=True, cmap=color_pallette, linewidths = 2)
            fig = fig.get_figure()
            fig.savefig(f'plots/heatmaps/heatmap_{self.__name__}.pdf')
            logger.info(
                f'SUCCESS -> data_heatmap_plot(color_pallette= {color_pallette}) -> '
                f'MSG -> Created heatmap plot ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig.__module__
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_heatmap_plot(color_pallette = {color_pallette}) -> '
                f'MSG -> Could not create heatmap plot ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def data_categoric_to_binary(self, target_name: str ,col_name: str ,base_value: str) -> pd.DataFrame:
        """
        Convert a categorical (eg.: 2 value options: [high,low]) to binary

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        target_name: str
            New column name created from this process
        col_name: str
            Column name to be transformed to binary values
        base_value: str
            Value from categorical data to be converted to False -> 0
            
        Returns:
        --------
        df: pd.DataFrame
            New dataframe pre-processed

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_categoric_to_binary(target_name= 'z' ,col_name= 'X1' ,base_value= 'low')
        """
        try:
            self.df[target_name] = self.df[col_name].apply(lambda val: 0 if val == base_value else 1)
            logger.info(
                f'SUCCESS -> data_categoric_to_binary(target_name= {target_name} ,col_name= {col_name} ,base_value= {base_value}) -> '
                f'MSG -> Dataframe pre-processed successfully ! -> '
                f'OUTPUT -> df cols: {self.df} .'
                )
            return self.df
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_categoric_to_binary(target_name= {target_name} ,col_name= {col_name} ,base_value= {base_value}) -> '
                f'MSG -> Dataframe could not be pre-processed ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def data_feature_encoder(self, col_name: str, target_col: str) -> pd.DataFrame:
        """
        Groupby a features to create a new dataframe column

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        col_name: str
            Column name to be transformed to binary values
        target_col: str
            Grouped column
            
        Returns:
        --------
        df: pd.DataFrame
            New dataframe pre-processed

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_feature_encoder(col_name= 'X1', target_col= 'X3')
        """
        category_lst = []
        try:
            category_groups = self.df.groupby(col_name).mean()[target_col]

            for val in self.df[col_name]:
                category_lst.append(category_groups.loc[val])

            self.df[f'{col_name}_{target_col}'] = category_lst
            logger.info(
                f'SUCCESS -> data_feature_encoder(col_name= {col_name}, target_col= {target_col} ) -> '
                f'MSG -> Dataframe pre-processed successfully ! -> '
                f'OUTPUT -> df cols: {self.df.columns.to_list()} .'
                )
            return self.df
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_feature_encoder(col_name= {col_name}, target_col= {target_col} ) -> '
                f'MSG -> Dataframe could not be pre-processed ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def data_build_ml_matrix(self, target_col:str, states_key: List):
        """
        Builds a Machine learning matrix X(y)

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        target_col: str
            Target column
        states_key: List
            List of columns to keep in the ML matrix
            
        Returns:
        --------
        ml_data: Dict
            Dictionary grouping X(y): {'X':self.X,'y':self.y}

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_build_ml_matrix(target_col= 'y', states_key= ['x1','x2',...]) 
        """
        try:
            self.y = self.df[target_col]
            self.X = pd.DataFrame()
            self.X = self.df.filter(items=states_key)
            self.ml_data = {'X':self.X,'y':self.y}
            logger.info(
                f'SUCCESS -> data_build_ml_matrix(target_col= {target_col}, states_key= {states_key}) -> '
                f'MSG -> Machine learning matrix is created ! -> '
                f'OUTPUT -> y: {self.y.to_list()} , \nX: {self.X} .'
                )
            return self.ml_data
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_build_ml_matrix(target_col= {target_col}, states_key= {states_key}) -> '
                f'MSG -> Machine learning matrix could not be created ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def split_test_train_data(self, test_size: float, random_state: int) -> Tuple:
        """
        Split test and train data for machine learning task

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        test_size: float
            Ratio of whole database to be used on test
        random_state: int
            Seed to avoid random test and train split
            
        Returns:
        --------
        data_processed: Dict
            Tuple containing: X_train, X_test, y_train, y_test

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> data_processed = model.split_test_train_data(test_size= 0.3,random_state= 11) 
        """
        try:
            self.data_processed = train_test_split(
                self.X,
                self.y,
                test_size= test_size,
                random_state= random_state)

            self.X_train, self.X_test, self.y_train, self.y_test = self.data_processed
            logger.info(
                f'SUCCESS -> split_test_train_data(test_size= {test_size} ,random_state= {random_state} ) -> '
                f'MSG -> Train and test data created ! -> '
                f'OUTPUT \n-> X_train: {self.X_train.head(n=2)} \n-> X_test: {self.X_test.head(n=2)} '
                f'\n-> y_train: {self.y_train.head(n=2)} \n-> y_test: {self.y_test.head(n=2)}'
                )
            return self.data_processed
        except BaseException as exc:
            logger.error(
                f'ERROR  -> split_test_train_data(test_size= {test_size} ,random_state= {random_state} ) -> '
                f'MSG -> Train and test data not created ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def fit_predict(self,model_name: str = None, model_data: Dict={},model_algorithm=None):
        """
        Fit and predict with chosen model

        Parameters
        ----------
        model_algorithm: Tuple
            Model algorithm and some extra info
        model_data: Dict
            Object to store model information to experiments monitoring
        
        Returns:
        --------
        self: CreateMlModel
            Create model object data
        model_data: Dict
            Object to store model information to experiments monitoring

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.split_test_train_data(test_size= 0.3,random_state= 11) 
            >>> model.fit_predict(model_data={'name':'lrc'},model_algorithm=('LR,'LinearRegression())) 
        """
        try:
            fit_model = model_algorithm.fit(self.X_train, self.y_train)
            model_data['model_name'] = model_name
            model_data['model'] = fit_model
            model_data['y_train_predicted'] = fit_model.predict(self.X_train)
            model_data['y_test_predicted'] = fit_model.predict(self.X_test)
            logger.info(
                f'SUCCESS -> fit_predict(model_data={model_data},model_algorithm={model_algorithm}) -> '
                f'MSG -> Fitted & Predicted model ! -> '
                f'OUTPUT \n-> model_data: {model_data} .'
                )
            return model_data
        except BaseException as exc:
            logger.error(
                f'ERROR  -> fit_predict(model_data={model_data},model_algorithm={model_algorithm}) -> '
                f'MSG -> Fit-Predict not worked ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    def fit_predict_to_best_estimator(
        self,
        model_data: Dict={},
        model_name: str = None,
        model_algorithm: Tuple = None,
        param_grid: Dict = None, 
        folds: int = None, 
    ):
        """
        Find the best parameters for the machine learning algorithm chosen

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_name: str
            Machine learning model name
        model_algorithm: Tuple
            Model algorithm and some extra info
        param_grid: Dict
            List of pre-set parameters to use with GridSearch
        folds: int
            Number of folds to perform cross-validation

        Returns:
        --------
        model_data: Dict
            Dictionary containing the fitted model and its estimation using test
            and train data 

        Examples:
        ---------

            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.split_test_train_data(test_size= 0.3,random_state= 11) 
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
        """
        try:
            model = GridSearchCV(
                estimator=model_algorithm, 
                param_grid=param_grid, 
                cv= folds
            )
            fit_model = model.fit(self.X_train, self.y_train)
            model_data['model_name'] = model_name
            model_data['model']=fit_model
            model_data['y_train_predicted'] = fit_model.best_estimator_.predict(self.X_train)
            model_data['y_test_predicted'] = fit_model.best_estimator_.predict(self.X_test)
           
            logger.info(
                f'SUCCESS -> fit_predict_to_best_estimator( model_name={model_name}, model_algorithm={model_algorithm}, param_grid={param_grid}, cv={folds} ) -> '
                f'MSG -> Predicted best estimator and parameters were generated ! -> '
                f'OUTPUT \n-> model_data: {model_data} .'
                )
            return model_data
        except BaseException as exc:
            logger.error(
                f'ERROR  -> fit_predict_to_best_estimator( model_name={model_name}, model_algorithm={model_algorithm}, param_grid={param_grid}, cv={folds} ) -> '
                f'MSG -> Best estimator process not worked ! ->'
                f'Exception -> {exc} .'
                )
            raise
   
    def tp_rate_analysis(self,ml_models: List[Dict]=(None,False)):
        """
        Method to create insights from visual inspection of roc curve
        analysing true positives rate on classifier

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        models: List
            List with model_data and if it had GridSearch processment
            
        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.split_test_train_data(test_size= 0.3,random_state= 11) 
            >>> model_data_1 = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
            >>> model_data_2 = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
            >>> model.tp_rate_analysis(models=[model_data_1,model_data_2])
        """
        try:
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            for model_data,grid_search in ml_models:
                if grid_search:
                    model_plot = plot_roc_curve(model_data['model'], self.X_test, self.y_test, ax=ax, alpha=0.8)
                else:
                    model_plot = plot_roc_curve(model_data['model'], self.X_test, self.y_test)
                    model_plot.plot(ax=ax, alpha=0.8)
            plt.show()
            logger.info(
                f'SUCCESS -> tp_rate_analysis(ml_models= {ml_models})-> '
                f'MSG -> Model parameters generated ! -> '
                f'OUTPUT -> None .'
                )
            return plt
        except BaseException as exc:
            logger.error(
                f'ERROR  -> tp_rate_analysis(ml_models= {ml_models})-> '
                f'MSG -> Model parameters not generated ! ->'
                f'Exception -> {exc} .'
                )
            raise

    def feature_importance_plot_1(self,model_data: Dict=None):
        """
        Generates a matplotlib bar plot to describe feature importance on
        X matrix targeting dimensionality reduction to avoid overfitting
        and decrease model complexity -> Matplotlib backend

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_data: str
            Machine learning model data
            
        Returns:
        --------
        plt: matplotlib
            Matplotlib plot object

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
            >>> model.feature_importance_plot_1(model_data=model_data)
        """
        try:
            model = model_data['model']
            importances = model.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [self.X.columns[i] for i in indices]
            
            plt.figure(figsize=(20,5))

            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(self.X.shape[1]), importances[indices])
            plt.xticks(range(self.X.shape[1]), names, rotation=90)
            plt.savefig('line_plot.pdf')  
            logger.info(
                    f'SUCCESS -> feature_importance_plot_1(model_data= {model_data}) -> '
                    f'MSG -> Feature importance plot generated ! -> '
                    f'OUTPUT -> None .'
                    )
            return plt
        except BaseException as exc:
            logger.error(
                f'ERROR  -> feature_importance_plot_1(model_data= {model_data})  -> '
                f'MSG -> Feature importance calculations not executed ! ->'
                f'Exception -> {exc} .'
                )
            raise 

    def feature_importance_plot_2(self,model_data: Dict=None):
        """
        Generates a matplotlib bar plot to describe feature importance on
        X matrix targeting dimensionality reduction to avoid overfitting
        and decrease model complexity -> Shap backend

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_data: str
            Machine learning model data
            
        Returns:
        --------
        shap: Shap
            Figure with feature importance information

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
            >>> model.feature_importance_plot_2(model_data=model_data)        """
        try:
            model = model_data['model']
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(self.X_test)
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")
            logger.info(
                f'SUCCESS -> feature_importance_plot_2(model_data= {model_data}) -> '
                f'MSG -> Feature importance plot 2 (shap engine) generated ! -> '
                f'OUTPUT -> None .'
                )
            return shap
        except BaseException as exc:
            logger.error(
                f'ERROR  -> feature_importance_plot_2(model_data= {model_data}) -> '
                f'MSG -> Feature importance calculations not executed ! ->'
                f'Exception -> {exc} .'
                )
            raise 
   
    def clf_report(self, model_data: Dict=None):
        """
        Generates a classification report with estimator
        information 

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_data: str
            Machine learning model data
            
        Returns:
        --------
        plt: matplotlib
            Plot with classification report info

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5) 
            >>> model.clf_report(model_data=model_data)
        """
        try:
            y_train_predicted = model_data['y_train_predicted']
            y_test_predicted = model_data['y_test_predicted']
            model_name = model_data['model_name']
            plt.figure(figsize=(5,5))
            plt.text(0.01, 1.25, model_name, {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.05, str(classification_report(self.y_test, y_test_predicted)), {'fontsize': 10}, fontproperties = 'monospace') 
            plt.text(0.01, 0.6, model_name, {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.7, str(classification_report(self.y_train, y_train_predicted)), {'fontsize': 10}, fontproperties = 'monospace') 
            plt.axis('off')
            plt.savefig(f'reports/{model_name}_report.pdf')
            plt.show()
            logger.info(
                f'SUCCESS -> clf_report(model_data= {model_data}) -> '
                f'MSG -> Report created ! -> '
                f'OUTPUT -> None .'
                )
            return plt
        except BaseException as exc:
            logger.error(
                f'ERROR  -> clf_report(model_data= {model_data}) -> '
                f'MSG -> Report not loaded ! ->'
                f'Exception -> {exc} .'
                )
            raise

    @staticmethod
    def saving(model_data: Dict=None):
        """
        Save model at models folder in pickle format to use it later on experiments

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_name: str
            Model name
        model: 
            Model itself
            
        Returns:
        --------
        joblib: Joblib
            Object to map dumped object on unit-tests

        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5)        
            >>> saving(model_data= model_data)
        """
        try:
            assert model_data
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            model_data['timestamp'] = timestamp
            name = model_data['model_name']
            model = model_data['model']
            joblib.dump(model_data, f'./models/{name}_{timestamp}.pkl')
            logger.info(
                f'SUCCESS -> saving(model= {model}) -> '
                f'MSG -> Model saved as pickle file ! -> '
                f'OUTPUT -> None .'
                )
            return joblib
        except BaseException as exc:
            logger.error(
                f'ERROR  -> saving(model= {model})  -> '
                f'MSG -> Model not saved ! ->'
                f'Exception -> {exc} .'
                )
            raise

    @staticmethod
    def loading(path_to_model:str) -> Dict:
        """
        Load model at models folder to use it on experiments

        Parameters
        ----------
        path_to_model: str
            Model name
            
        Returns:
        --------
        model: Dict
            model data

        Examples:
        ---------
            >>> loading('test/rfc')
        """
        try:
            file_name = f'{path_to_model}.pkl'
            model = joblib.load(file_name)
            logger.info(
                f'SUCCESS -> loading(path_to_model= {path_to_model}) -> '
                f'MSG -> Model loaded ! -> '
                f'OUTPUT -> model: {model} .'
                )
            return model
        except BaseException as exc:
            logger.error(
                f'ERROR  -> loading(path_to_model= {path_to_model}) -> '
                f'MSG -> Model not loaded ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    @staticmethod     
    def remove_cols(cols_lst: List[str]=None,cols_to_rm: List[str]=None) -> List[str]:
        """
        Method to remove items from a certain list
        
        Parameters:
        ----------
        self: CreateMlModel
            Create model object data
        cols_lst: List[str]
            List to be changed
        cols_to_rm: List[str]
            List of columns to be removed from cols_lst

        Returns:
        --------
        cols_lst: List[str]
            List changed

        Examples:
        ---------
        >>> model = mlvc.CreateMlModel('test')
        >>> load_data(df_path= 'path/file.csv')
        >>> numeric_cols, categoric_cols = model.isolate_categ_and_num_cols()
        >>> numeric_cols = remove_cols(numeric_cols,['x1'])
        >>> numeric_cols
            ['x2',...]
        """
        for col in cols_to_rm:
            try:
                cols_lst.remove(col)
            except:
                logger.warning(
                    f'WARNING -> MSG -> Could not remove column: {col} !'
                )
                pass
        logger.info(
            f'SUCCESS -> remove_cols(cols_lst={cols_lst}, cols_to_rm={cols_to_rm}) -> '
            f'MSG -> Removed columns from list ! -> '
            f'OUTPUT -> cols_lst_old: {cols_lst} -> cols_lst_new: {cols_lst} .'
        )
        return cols_lst

