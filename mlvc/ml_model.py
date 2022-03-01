"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string
import json
import os
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
    

    def __init__(self, name):
        self.__name__ = name

    #   Unit-tested -> Integration-Tested
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
    
    #   Unit-tested -> Integration-Tested
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

    #   TODO: Unit-tested -> Integration-Tested -> Exception case not done
    def data_hist_plot(self,col_name: str) -> None:
        """
        Create a histogram plot from a dataframe

        Parameters
        ----------
        col_name: str
            Column name to generate the histogram plot from

        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_hist_plot(col_name= 'X')
        """
        try:
            fig = self.df[col_name].hist();
            fig = fig.get_figure()
            fig.savefig(f'plots/hist_plot_{col_name}.pdf')
            logger.info(
                f'SUCCESS -> df_hist_plot(col_name={col_name}) -> '
                f'MSG -> Dataframe histogram plot created ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig.__module__
        except BaseException as exc:
            logger.error(
                f'ERROR -> df_hist_plot(col_name={col_name}) ->'
                f'MSG -> Could not create dataframe histogram ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    #   TODO: Unit-tested -> Integration-Tested -> Exception case not done
    def normalized_data_plot(self,col_name: str, plot_type: str) -> None:
        """
        Create a specific plot from a pandas series normalized

        Parameters
        ----------
        col_name: str
            Column name to generate the histogram plot from
        plot_type: str
            Plot type

        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
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
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR -> normalized_data_plot(col_name={col_name} ,plot_type= {plot_type}) -> '
                f'MSG -> Could not create Pandas series plot {plot_type} ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    #   TODO: Unit-tested -> Integration-Tested -> Exception case not done
    def data_dist_plot(self,col_name:str) -> None:
        """
        Create a dist plot from a pandas series using seaborn backend

        Parameters
        ----------
        col_name: str
            Column name to generate the dist plot from

        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_dist_plot(col_name= 'X')
        """
        try:
            fig = sns.distplot(self.df[col_name]);
            fig = fig.get_figure()
            fig.savefig(f'plots/distplot_plot_{col_name}.pdf')
            logger.info(
                f'SUCCESS -> normalized_data_plot(col_name={col_name}) -> '
                f'MSG -> Created Pandas series dist plot ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR  -> normalized_data_plot(col_name={col_name}) -> '
                f'MSG -> Could not create Pandas series dist plot ! ->'
                f'Exception -> {exc} .'
                )
            raise

    #   TODO: Unit-tested -> Integration-Tested -> Exception case not done
    def data_heatmap_plot(self, color_pallete:str ='Dark2_r') -> None:
        """
        Create a heatmap plot from a pandas correlation matrix 
        using seaborn backend

        Parameters
        ----------
        None

        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_heatmap_plot()
        """
        try:
            fig = sns.heatmap(self.df.corr(), annot=True, cmap=color_pallete, linewidths = 2)
            fig = fig.get_figure()
            fig.savefig(f'plots/heatmap_{self.__name__}.pdf')
            logger.info(
                f'SUCCESS -> data_heatmap_plot() -> '
                f'MSG -> Created heatmap plot ! -> '
                f'OUTPUT -> {fig.__dict__} .'
                )
            return fig
        except BaseException as exc:
            logger.error(
                f'ERROR  -> data_heatmap_plot() -> '
                f'MSG -> Could not create heatmap plot ! ->'
                f'Exception -> {exc} .'
                )
            raise
    
    #   Unit-tested -> Integration-Tested 
    def data_categoric_to_binary(self, target_name: str ,col_name: str ,base_value: str) -> pd.DataFrame:
        """
        Convert a categorical (eg.: 2 value options: [high,low]) to binary

        Parameters
        ----------
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
            >>> model = mlvc.MlModel('test')
            >>> model.data_categoric_to_binary(target_name= 'z' ,col_name= 'X1' ,base_value= 'low')
        """
        try:
            self.df[target_name] = self.df[col_name].apply(lambda val: 0 if val == base_value else 1)
            logger.info(
                f'SUCCESS -> data_categoric_to_binary(target_name= {target_name} ,col_name= {col_name} ,base_value= {base_value}) -> '
                f'MSG -> Dataframe pre-processed succesfully ! -> '
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
    
    #   TODO: Unit-tested -> Integration-Tested -> Exception case not done
    def data_feature_encoder(self, col_name: str, target_col: str) -> pd.DataFrame:
        """
        Groupby a feature to create a new dataframe collumn?

        Parameters
        ----------
        col_name: str
            Column name to be transformed to binary values
        target_col: str
            Column to be compared? TODO: What is groupby??
            
        Returns:
        --------
        df: pd.DataFrame
            New dataframe pre-processed

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
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
                f'MSG -> Dataframe pre-processed succesfully ! -> '
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
    
    #  Unit-tested -> ...
    def data_build_ml_matrix(self, target_col:str, states_key: List):
        """
        Builds a Machine learning matrix X(y)

        Parameters
        ----------
        target_col: str
            Target column
        states_key: List
            List of columns to be keeped in the ML matrix
            
        Returns:
        --------
        ml_data: Dict
            Dictionary grouping X(y): {'X':self.X,'y':self.y}

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
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
    
    #  Unit-tested -> ...
    def split_test_train_data(self, test_size: float, random_state: int) -> Tuple:
        """
        Split test and train data for machine learning task

        Parameters
        ----------
        test_size: float
            Ratio of whole database to be used on test
        random_state: int
            Seed to avoid random test and train split
            
        Returns:
        --------
        data_processed: Dict
            Tuple conatining: X_train, X_test, y_train, y_test

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
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
            raise exc
    
    # TODO: -> Unit-test-> Test
    def fit_predict(self,model_data: Dict=None,model_algorithm=None):
        """
        Fit and predict with choosen model

        Parameters
        ----------
        model_algorithm: Tuple
            Model algorithm and some extra info
        model_data: Dict
            Object to store model information to experiments monitoring
        
        Returns:
        --------
        model_data: Dict
            Object to store model information to experiments monitoring

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.split_test_train_data(test_size= 0.3,random_state= 11) 
            >>> model.fit_predict(model_data={'name':'lrc'},model_algorithm=('LR,'LinearRegression())) 
        """
        try:
            model_algorithm.fit(self.X_train, self.y_train)
            model_data['y_train_predicted'] = model_algorithm.predict(self.X_train)
            model_data['y_test_predicted'] = model_algorithm.predict(self.X_test)
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
            raise exc
    
    # TODO: -> Unit-test-> Test
    @staticmethod
    def best_estimator(
        model_name: str = None,
        model_algorithm: Tuple = None,
        param_grid: Dict = None, 
        folds: int = None, 
    ):
        """
        Find the best parameters for the machine learning algorithm choosen

        Parameters
        ----------
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
            >>> model_data = best_estimator(model_algorithm=('LR,'LinearRegression()),param_grid= {data ...},folds= 5) 
        """
        try:
            model = GridSearchCV(
                estimator=model_algorithm, 
                param_grid=param_grid, 
                cv= folds
            )
            model_data ={
                'model_name':model_name,
                'model_results': model
            }
            logger.info(
                f'SUCCESS -> best_estimator( model_name={model_name}, model_algorithm={model_algorithm}, param_grid={param_grid}, cv={folds} ) -> '
                f'MSG -> Best estimator and parameters were generated ! -> '
                f'OUTPUT \n-> model_data: {model_data} .'
                )
            return model_data
        except BaseException as exc:
            logger.error(
                f'ERROR  -> best_estimator( model_name={model_name}, model_algorithm={model_algorithm}, param_grid={param_grid}, cv={folds} ) -> '
                f'MSG -> Best estimator worked ! ->'
                f'Exception -> {exc} .'
                )
            raise exc

    # TODO : Test in workspace!      
    def tp_rate_analysis(self,ml_models: List[Tuple]=(None,False)):
        """
        Method to create insights from visual inspection of roc curve
        analysing true positives rate on classifier

        Parameters
        ----------
        model_1:
            Model 1 to be analyzed
        model_2: int
            Model 2 to be analyzed
            
        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_1 = model.tuning(model_algorithm=('LR,'LinearRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model_2 = model.tuning(model_algorithm=('LR,'LogisticRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model.tp_rate_analysis(model_1,model_2)
        """
        try:
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            for model,grid_search in ml_models:
                if grid_search:
                    model_plot_1 = plot_roc_curve(model.best_estimator_, self.X_test, self.y_test, ax=ax, alpha=0.8)
                else:
                    model_plot_2 = plot_roc_curve(model, self.X_test, self.y_test)
                    model_plot_2.plot(ax=ax, alpha=0.8)
            plt.show()
            logger.info(
                f'SUCCESS -> tp_rate_analysis(ml_models= {ml_models})-> '
                f'MSG -> Model parameters generated ! -> '
                f'OUTPUT -> None .'
                )
            return  
        except BaseException as exc:
            logger.error(
                f'ERROR  -> tp_rate_analysis(ml_models= {ml_models})-> '
                f'MSG -> Model parameters not generated ! ->'
                f'Exception -> {exc} .'
                )
            raise

    # TODO : Test in workspace!
    @staticmethod
    def saving(model: Dict=None):
        """
        Save model at models folder in pickle format to use it later on experiments

        Parameters
        ----------
        model_name: str
            Model name
        model: 
            Model itself
            
        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_1 = model.tuning(model_algorithm=('LR,'LinearRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model_2 = model.tuning(model_algorithm=('LR,'LogisticRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model.tp_rate_analysis(model_1,model_2)
            >>> saving(model_name= 'LR', model= model_1)
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            model['timestamp'] = timestamp
            name = model['name']
            joblib.dump(model, f'./models/{name}_{timestamp}.pkl')
            logger.info(
                f'SUCCESS -> saving(model= {model}) -> '
                f'MSG -> Model saved as pickle file ! -> '
                f'OUTPUT -> None .'
                )
            return  
        except BaseException as exc:
            logger.error(
                f'ERROR  -> saving(model= {model})  -> '
                f'MSG -> Model not saved ! ->'
                f'Exception -> {exc} .'
                )
            raise

    # TODO : Test in workspace!
    @staticmethod
    def loading(path_to_model:str):
        """
        Load model at models folder to use it on experiments

        Parameters
        ----------
        path_to_model: str
            Model name
            
        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_1 = model.tuning(model_algorithm=('LR,'LinearRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model_2 = model.tuning(model_algorithm=('LR,'LogisticRegression()),param_grid= None,folds= None,grid_search= False,best_estimator= False) 
            >>> model.tp_rate_analysis(model_1,model_2)
            >>> saving(model_name= 'LR', model= model_1)
            >>> model_1 = loading(model_name= 'LR')
        """
        try:
            model = joblib.load(f'{path_to_model}.pkl')
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
    
    
    # TODO : Test in workspace!
    def output_explanation(self):
        try:
            explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            logger.info('SUCCESS -> ')
        except BaseException as exc:
            logger.error(f'ERROR -> {exc}')

    # TODO: -> Unit-test-> Test
    def feature_importance_plot(self,model):
        """
        Generates a matplotlib bar plot to describe feature importance on
        X matrix targeting dimensionality reduction to avoid overfitting
        and decrease model complexity. 

        Parameters
        ----------
        model: str
            Machine learning model fitted
            
        Returns:
        --------
        None

        Examples:
        ---------
            >>> model = mlvc.MlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_1 = model.tuning(model_algorithm=('LR,'LinearRegression()))
        """
        try:
            importances = model.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [self.X.columns[i] for i in indices]
            
            plt.figure(figsize=(20,5))

            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(X.shape[1]), importances[indices])
            plt.xticks(range(X.shape[1]), names, rotation=90)
            plt.savefig('line_plot.pdf')  
            logger.info(
                    f'SUCCESS -> feature_importance_plot(model= {model}) -> '
                    f'MSG -> Feature importance plot generated ! -> '
                    f'OUTPUT -> None .'
                    )
            return  
        except BaseException as exc:
            logger.error(
                f'ERROR  -> feature_importance_plot(model= {model})  -> '
                f'MSG -> Feature importance calculations not executed ! ->'
                f'Exception -> {exc} .'
                )
            raise exc('Feature importance calculations not executed !')
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


