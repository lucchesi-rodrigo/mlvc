"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from loguru import logger
from typing import Dict, List, Tuple
import json
from lib2to3.pytree import Base
import os
import json
from datetime import datetime, date
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
sns.set()


class MlModeling:
    """Class to process Machine Learning Models"""

    def __init__(self, model_name, model_algorithm,model_version,model_notes=[]) -> None:
        """Init method which has as input the model instance name"""
        try:
            self.model_name = model_name
            assert self.model_name is not None
            self.model_algorithm = model_algorithm
            assert self.model_algorithm is not None
            self.model_datetime = datetime.now()
            self.model_version = model_version
            self.model_notes = model_notes
            self.model_data = {
                'model_name':self.model_name,
                'model_algorithm':self.model_algorithm,
                'model_datetime':self.model_datetime ,
                'model_version':self.model_version,
                'model_notes':self.model_notes
            }
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise AssertionError(       
                f"[ERROR -> __init__({self})] -> "
                f"MSG -> Could not build MlModeling instance ! -> Exception: {exc}!"
            )

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
            logger.info(
                f'[SUCCESS -> data_loading({df_path})] -> '
                f'MSG -> data_loading starting process !'
            )
            self.df = pd.read_csv(df_path)
            data_sample = self.df.head(n=2).to_json()
            logger.info(
                f'[SUCCESS -> data_loading({df_path})] -> '
                f'MSG -> data_loading finished process ! -> '
                f'OUTPUT -> df sample: {data_sample}'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise FileNotFoundError(       
                f"[ERROR -> data_loading({df_path})] -> "
                f"MSG -> Could not find file with this path ! -> Exception: {exc}!"
            )

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
            logger.info(
                f'[SUCCESS -> data_analysis()] -> '
                f'MSG -> Data analysis starting process ! -> '
            )
            self.stats_data = {
                'shape': self.df.shape,
                'null_values': self.df.isnull().sum(),
                'numeric_stats': self.df.describe().to_json()
            }
            numeric_stats = self.stats_data['numeric_stats']

            logger.info(
                f'[SUCCESS -> data_analysis()] -> '
                f'MSG -> Data analysis finished process ! -> '
                f'OUTPUT -> stats_data numeric_stats: {numeric_stats} !'
            )
            return self.stats_data
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> data_analysis()] -> "
                f"MSG -> Could not calculate DataFrame statistics -> Exception: {exc}!"
            )

    def isolate_categ_and_num_cols(self) -> Tuple[List[str], List[str]]:
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
            >>> model.data_loading(df_path= 'path/file.csv')
            >>> numeric_cols, categoric_cols = model.isolate_categ_and_num_cols()
            >>> numeric_cols
                ['x1','x2',...]
        """
        try:
            logger.info(
                f'[SUCCESS -> isolate_categ_and_num_cols()] -> '
                f'MSG -> isolate_categ_and_num_cols starting process ! -> '
            )
            self.numeric_cols = [
                column for column in self.df.columns if self.df[column].dtype != 'object']
            self.categoric_cols = [
                column for column in self.df.columns if self.df[column].dtype == 'object']
            logger.info(
                f'[SUCCESS -> isolate_categ_and_num_cols()] -> '
                f'MSG -> Data analysis finished process ! -> '
                f'OUTPUT -> numeric_cols: {self.numeric_cols}, categoric_cols: {self.categoric_cols}!'
            )
            return self.numeric_cols, self.categoric_cols
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> isolate_categ_and_num_cols()] -> "
                f"MSG -> Could not isolate categorical and numerical columns -> Exception: {exc}!"
            )

    def data_hist_plot(self, col_name: str) -> None:
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
            logger.info(
                f'[SUCCESS -> data_hist_plot(col_name={col_name})] -> '
                f'MSG -> data_hist_plot starting process ! -> '
            )
            fig = self.df[col_name].hist()
            fig = fig.get_figure()
            fig.savefig(f'plots/histograms/hist_plot_{col_name}.pdf')
            logger.info(
                f'[SUCCESS -> data_hist_plot(col_name={col_name})] -> '
                f'MSG -> data_hist_plot finished process ! -> '
                f'OUTPUT -> file saved at: plots/histograms/hist_plot_{col_name}.pdf !'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> data_hist_plot(col_name={col_name})] -> "
                f"MSG -> Could not create dist plot -> Exception: {exc}!"
            )

    def normalized_barplots_data_plot(self, col_name: str) -> None:
        """
        Create a barplot from a normalized pandas series

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
            >>> model.normalized_data_plot(col_name= 'X', plot_type= 'bar')
        """
        try:
            logger.info(
                f'[SUCCESS -> normalized_data_plot(col_name={col_name})] -> '
                f'MSG -> normalized_barplots_data_plot starting process ! -> '
            )
            fig = self.df[col_name].value_counts(
                'normalize').plot(kind='bar')
            fig = fig.get_figure()
            fig.savefig(f'plots/barplots/barplot_plot_{col_name}.pdf')
            logger.info(
                f'[SUCCESS -> normalized_barplots_data_plot(col_name={col_name})] -> '
                f'MSG -> normalized_barplots_data_plot finished process ! -> '
                f'OUTPUT -> file saved at: plots/barplots/barplot_plot_{col_name}.pdf !'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> normalized_barplots_data_plot(col_name={col_name})] -> "
                f"MSG -> Could not create barplot ! -> Exception: {exc}!"
            )

    def data_dist_plot(self, col_name: str) -> None:
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
            logger.info(
                f'[SUCCESS -> data_dist_plot(col_name={col_name})] -> '
                f'MSG -> data_dist_plot starting process ! -> '
            )
            x_list = self.df[col_name].tolist()
            x = pd.Series(x_list, name=f"{col_name}")
            logger.info(f'x:{x}')
            fig = plt.figure(figsize=(20,10))
            sns.distplot(x)
            fig.get_figure()
            fig.savefig(f'plots/distplots/distplot_plot_{col_name}.pdf')
            logger.info(
                f'[SUCCESS -> data_dist_plot(col_name={col_name})] -> '
                f'MSG -> data_dist_plot finishing process ! -> '
                f'OUTPUT -> file saved at: plots/distplots/distplot_plot_{col_name}.pdf !'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> normalized_barplots_data_plot(col_name={col_name})] -> "
                f"MSG -> Could not create distplot ! -> Exception: {exc}!"
            )

    def data_heatmap_plot(self, color_pallette: str = 'Dark2_r') -> None:
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
            logger.info(
                f'[SUCCESS -> data_heatmap_plot()] -> '
                f'MSG -> data_heatmap_plot starting process ! -> '
            )
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(
                self.df.corr(),
                annot=True,
                cmap=color_pallette,
                linewidths=2)
            fig.get_figure()
            fig.savefig(f'plots/heatmaps/heatmap_{self.model_name}.pdf')
            logger.info(
                f'[SUCCESS -> data_heatmap_plot()] -> '
                f'MSG -> data_heatmap_plot finishing process ! -> '
                f'OUTPUT -> file saved at: plots/heatmaps/heatmap_{self.model_name}.pdf !'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f"[ERROR -> data_heatmap_plot()] -> "
                f"MSG -> Could not create heatmap ! -> Exception: {exc}!"
            )

    def data_categoric_to_binary(
            self,
            target_name: str,
            col_name: str,
            base_value: str) -> pd.DataFrame:
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
            logger.info(
                f'[SUCCESS -> data_categoric_to_binary(target_name={target_name},col_name={col_name},base_value={base_value})] -> '
                f'MSG -> data_categoric_to_binary starting process ! -> '
            )
            self.df[target_name] = self.df[col_name].apply(
                lambda val: 0 if val == base_value else 1)
            data_sample = self.df.head(n=2).to_json()
            logger.info(
                f'[SUCCESS -> data_categoric_to_binary(target_name={target_name},col_name={col_name},base_value={base_value}) ] -> '
                f'MSG -> data_categoric_to_binary finishing process ! -> '
                f'OUTPUT -> data sample: {data_sample} !'
            )
            return self.df
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> data_categoric_to_binary(target_name={target_name},col_name={col_name},base_value={base_value}) ] -> '
                f"MSG -> Could not convert the column ! -> Exception: {exc}!"
            )

    def data_feature_encoder(
            self,
            col_name: str,
            target_col: str) -> pd.DataFrame:
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
            logger.info(
                f'[SUCCESS -> data_feature_encoder(col_name={col_name},target_col={target_col})] -> '
                f'MSG -> data_feature_encoder starting process ! -> '
                f'OUTPUT -> data sample: {data_sample} !'
            )
            category_groups = self.df.groupby(col_name).mean()[target_col]

            for val in self.df[col_name]:
                category_lst.append(category_groups.loc[val])

            self.df[f'{col_name}_{target_col}'] = category_lst
            
            data_sample = self.df.head(n=2).to_json()
            logger.info(
                f'[SUCCESS -> data_feature_encoder(col_name={col_name},target_col={target_col})] -> '
                f'MSG -> data_feature_encoder finishing process ! -> '
                f'OUTPUT -> data sample: {data_sample} !'
            )
            return self.df
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[SUCCESS -> data_feature_encoder(col_name={col_name},target_col={target_col})] -> '
                f"MSG -> Could encode data ! -> Exception: {exc}!"
            )

    def data_build_ml_matrix(self, target_col: str, states_key: List) -> Tuple[pd.DataFrame,pd.DataFrame]:
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
            logger.info(
                f'[SUCCESS -> data_build_ml_matrix(target_col={target_col}, states_key={states_key})] -> '
                f'MSG -> data_build_ml_matrix starting process ! -> '
            )
            self.y = self.df[target_col]
            self.X = pd.DataFrame
            self.X = self.df.filter(items=states_key)
            X_sample = self.X.head(n=2).to_json()
            y_sample = self.y[:2].tolist()

            logger.info(
                f'[SUCCESS -> data_build_ml_matrix(target_col={target_col}, states_key={states_key})] -> '
                f'MSG -> data_build_ml_matrix finishing process ! -> '
                f'OUTPUT -> X sample: {X_sample}, y_sample: {y_sample} !'
            )
            return self.y, self.X
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[SUCCESS -> data_build_ml_matrix(target_col={target_col}, states_key={states_key})] -> '
                f"MSG -> Could create ml data ! -> Exception: {exc}!"
            )

    def split_test_train_data(
            self,
            test_size: float,
            random_state: int) -> Tuple:
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
            logger.info(
                f'[SUCCESS -> split_test_train_data(test_size={test_size},random_state={random_state})] -> '
                f'MSG -> split_test_train_data starting process !'
            )
            self.data_processed = train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state)

            self.X_train, self.X_test, self.y_train, self.y_test = self.data_processed
            x_train = self.X_train.head().to_json()
            X_test = self.X_test.head().to_json()
            y_train = self.y_train.to_list()[:2]
            y_test = self.y_test.to_list()[:2]

            logger.info(
                f'[SUCCESS -> split_test_train_data(test_size={test_size},random_state={random_state})] -> '
                f'MSG -> split_test_train_data finishing process !'
                f'OUTPUT -> x_train: {x_train}, X_test: {X_test}, y_train: {y_train}, y_test:{y_test} !'
            )
            return self.data_processed
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> split_test_train_data(test_size={test_size},random_state={random_state})] -> '
                f"MSG -> Could split ml data ! -> Exception: {exc}!"
            )

    def fit_predict(
            self
            ) -> Dict:
        """
        Fit and predict with chosen model

        Parameters
        ----------
        self: CreateMlModel
            Create model object data

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
            logger.info(
                f'[SUCCESS -> fit_predict()] -> '
                f'MSG -> fit_predict starting process !'
            )
            fit_model = self.model_algorithm.fit(self.X_train, self.y_train)
            self.model_data['model_name'] = self.model_name
            self.model_data['model'] = fit_model
            self.model_data['y_train_predicted'] = fit_model.predict(self.X_train)
            self.model_data['y_test_predicted'] = fit_model.predict(self.X_test)
            logger.info(
                f'[SUCCESS -> fit_predict()] -> '
                f'MSG -> fit_predict finishing process !'
            )
            logger.debug(
                f'[SUCCESS -> fit_predict()] -> '
                f'MSG -> fit_predict finishing process !'
                f'OUTPUT -> self.model_data: {self.model_data.__dict__}  !'
            )
            return self.model_data
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> fit_predict()] -> '
                f"MSG -> Could split ml data ! -> Exception: {exc}!"
            )

    def fit_predict_to_best_estimator(
        self,
        param_grid: Dict = None,
        folds: int = None,
    ) -> Dict:
        """
        Find the best parameters for the machine learning algorithm chosen

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
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
            logger.info(
                f'[SUCCESS -> fit_predict_to_best_estimator(param_grid= {param_grid},folds= {folds})] -> '
                f'MSG -> fit_predict_to_best_estimator starting process !'
            )
            model = GridSearchCV(
                estimator=self.model_algorithm,
                param_grid=param_grid,
                cv=folds
            )
            fit_model = model.fit(self.X_train, self.y_train)
            self.model_data['model_name'] = self.model_name
            self.model_data['model'] = fit_model
            self.model_data['y_train_predicted'] = fit_model.best_estimator_.predict(
                self.X_train)
            self.model_data['y_test_predicted'] = fit_model.best_estimator_.predict(
                self.X_test)

            logger.info(
                f'[SUCCESS -> fit_predict_to_best_estimator(param_grid= {param_grid},folds= {folds})] -> '
                f'MSG -> fit_predict_to_best_estimator starting process !'
            )
            logger.debug(
                f'[SUCCESS -> fit_predict_to_best_estimator(param_grid= {param_grid},folds= {folds})] -> '
                f'MSG -> fit_predict_to_best_estimator starting process !'
                f'OUTPUT -> self.model_data: {self.model_data.__dict__}  !'
            )
            return self.model_data
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> fit_predict_to_best_estimator(param_grid= {param_grid},folds= {folds})] -> '
                f"MSG -> Could execute fit_predict_to_best_estimator ! -> Exception: {exc}!"
            )

    def tp_rate_analysis(self, ml_models: List[Tuple[Dict,bool]] = None) -> None:
        """
        Method to create insights from visual inspection of roc curve
        analysing true positives rate on classifier

        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        models: List
            List with model_data and if it had GridSearch procedure

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
            logger.info(
                f'[SUCCESS -> tp_rate_analysis(ml_models= {ml_models})] -> '
                f'MSG -> tp_rate_analysis starting process !'
            )
            ax = plt.gca()
            for model_data, grid_search in ml_models:
                if grid_search:
                    model_plot = plot_roc_curve(
                        model_data['model'], self.X_test, self.y_test, ax=ax, alpha=0.8)
                else:
                    model_plot = plot_roc_curve(
                        model_data['model'], self.X_test, self.y_test)
                    model_plot.plot(ax=ax, alpha=0.8)
            model_plot.get_figure()
            model_plot.savefig(f'plots/tp_plots/tp_{self.__name__}.pdf')
            plt.show()

            logger.info(
                f'[SUCCESS -> tp_rate_analysis(ml_models= {ml_models})] -> '
                f'MSG -> tp_rate_analysis finishing process !'
                f'OUTPUT -> file saved at: plots/tp_plots/tp_{self.__name__}.pdf !'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> tp_rate_analysis(ml_models= {ml_models})]  -> '
                f"MSG -> Could execute tp_rate_analysis ! -> Exception: {exc}!"
            )

    def feature_importance_plot_1(self, model_data: Dict = None):
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
            logger.debug(
                f'[SUCCESS -> feature_importance_plot_1(model_data= {model_data})] -> '
                f'MSG -> feature_importance_plot_1 starting process !'
            )
            logger.info(
                f'[SUCCESS -> feature_importance_plot_1(model_data)] -> '
                f'MSG -> feature_importance_plot_1 starting process !'
            )
            model = model_data['model']
            importance = model.best_estimator_.feature_importances_
            indices = np.argsort(importance)[::-1]
            names = [self.X.columns[i] for i in indices]

            plt.figure(figsize=(20, 5))

            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(self.X.shape[1]), importance[indices])
            plt.xticks(range(self.X.shape[1]), names, rotation=90)
            plt.savefig('plots/feature_importance/feature_importance_plot1_{self.__name__}.pdf')

            logger.debug(
                f'[SUCCESS -> feature_importance_plot_1(model_data= {model_data})] -> '
                f'MSG -> feature_importance_plot_1 finishing process ! -> '
                f'OUTPUT -> plots/feature_importance/feature_importance_plot1_{self.__name__}.pdf'
            )
            logger.info(
                f'[SUCCESS -> feature_importance_plot_1(model_data)] -> '
                f'MSG -> feature_importance_plot_1 starting process ! ->'
                f'OUTPUT -> plots/feature_importance/feature_importance_plot1_{self.__name__}.pdf'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> feature_importance_plot_1(model_data= {model_data})] -> '
                f"MSG -> Could execute feature_importance_plot_1 ! -> Exception: {exc}!"
            )

    def feature_importance_plot_2(self, model_data: Dict = None):
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
            logger.debug(
                f'[SUCCESS -> feature_importance_plot_2(model_data= {model_data})] -> '
                f'MSG -> feature_importance_plot_2 starting process !'
            )
            logger.info(
                f'[SUCCESS -> feature_importance_plot_2(model_data)] -> '
                f'MSG -> feature_importance_plot_2 starting process !'
            )
            model = model_data['model']
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(self.X_test)
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")
            shap.savefig('plots/feature_importance/feature_importance_plot2_{self.__name__}.pdf')
            logger.debug(
                f'[SUCCESS -> feature_importance_plot_2(model_data= {model_data})] -> '
                f'MSG -> feature_importance_plot_2 finishing process ! -> '
                f'OUTPUT -> plots/feature_importance/feature_importance_plot1_{self.__name__}.pdf'
            )
            logger.info(
                f'[SUCCESS -> feature_importance_plot_1(model_data)] -> '
                f'MSG -> feature_importance_plot_2 starting process ! ->'
                f'OUTPUT -> plots/feature_importance/feature_importance_plot1_{self.__name__}.pdf'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> feature_importance_plot_1(model_data= {model_data})] -> '
                f"MSG -> Could execute feature_importance_plot_2 ! -> Exception: {exc}!"
            )

    def clf_report(self, model_data: Dict = None):
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
            logger.debug(
                f'[SUCCESS -> clf_report(model_data= {model_data.__dict__})] -> '
                f'MSG -> clf_report starting process ! -> '
            )
            logger.info(
                f'[SUCCESS -> clf_report(model_data)] -> '
                f'MSG -> clf_report starting process ! -> '
            )
            y_train_predicted = model_data['y_train_predicted']
            y_test_predicted = model_data['y_test_predicted']
            model_name = model_data['model_name']
            plt.figure(figsize=(5, 5))
            plt.text(
                0.01, 1.25, model_name, {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        self.y_test, y_test_predicted)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.6, model_name, {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        self.y_train, y_train_predicted)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(f'reports/{model_name}_report.pdf')
            plt.show()

            logger.debug(
                f'[SUCCESS -> clf_report(model_data= {model_data.__dict__})] -> '
                f'MSG -> clf_report starting process ! -> '
                f'OUTPUT -> reports/{model_name}_report.pdf'
            )
            logger.info(
                f'[SUCCESS -> clf_report(model_data)] -> '
                f'MSG -> clf_report starting process ! -> '
                f'OUTPUT -> reports/{model_name}_report.pdf'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> clf_report(model_data= {model_data.__dict__})] -> '
                f"MSG -> Could execute clf_report ! -> Exception: {exc}!"
            )

    @staticmethod
    def saving(model_data: Dict = None):
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
            logger.debug(
                f'[SUCCESS -> saving(model_data= {model_data.__dict__})] -> '
                f'MSG -> saving starting process ! -> '
            )
            logger.info(
                f'[SUCCESS -> saving(model_data)] -> '
                f'MSG -> saving starting process ! -> '
            )
            assert model_data
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            model_data['timestamp'] = timestamp
            name = model_data['model_name']
            model = model_data['model']
            joblib.dump(model_data, f'./models/{name}_{timestamp}.pkl')
            logger.debug(
                f'[SUCCESS -> saving(model_data= {model_data.__dict__})] -> '
                f'MSG -> saving starting process ! -> '
                f'OUTPUT -> ./models/{name}_{timestamp}.pkl'
            )
            logger.info(
                f'[SUCCESS -> saving(model_data)] -> '
                f'MSG -> saving starting process ! -> '
                f'OUTPUT -> ./models/{name}_{timestamp}.pkl'
            )
            return
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> saving(model_data= {model_data.__dict__})] -> '
                f"MSG -> Could execute saving ! -> Exception: {exc}!"
            )

    @staticmethod
    def loading(path_to_model: str) -> Dict:
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
            logger.info(
                f'[SUCCESS -> loading({path_to_model})] -> '
                f'MSG -> loading starting process ! -> '
            )
            file_name = f'{path_to_model}.pkl'
            model = joblib.load(file_name)
            logger.debug(
                f'[SUCCESS -> loading({path_to_model})] -> '
                f'MSG -> loading starting process ! -> '
                f'OUTPUT -> model from loaded {path_to_model}.pkl'
            )
            logger.info(
                f'[SUCCESS -> loading({path_to_model})] -> '
                f'MSG -> loading starting process ! -> '
                f'OUTPUT -> model: {model.__dict__}'
            )
            return model
        except BaseException as exc:
            logger.error(
                str(traceback.format_exc()).replace('\n', ' | ')
                )
            raise RuntimeError(       
                f'[ERROR -> loading({path_to_model})] -> '
                f"MSG -> Could execute loading ! -> Exception: {exc}!"
            )

    @staticmethod
    def remove_cols(cols_lst: List[str] = None,
                    cols_to_rm: List[str] = None) -> List[str]:
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
        logger.info(
            f'SUCCESS -> remove_cols(cols_lst={cols_lst}, cols_to_rm={cols_to_rm}) -> '
            f'MSG -> remove_cols starting process !'
        )
        for col in cols_to_rm:
            try:
                cols_lst.remove(col)
            except BaseException:
                logger.warning(
                    f'WARNING -> MSG -> Could not remove column: {col} !'
                )
                pass
        logger.info(
            f'SUCCESS -> remove_cols(cols_lst={cols_lst}, cols_to_rm={cols_to_rm}) -> '
            f'MSG -> remove_cols finishing process ! ->'
            f'OUTPUT -> cols_lst_old: {cols_lst} -> cols_lst_new: {cols_lst} .')
        return cols_lst

    def model_notes(self,notes:str):
        try:
            logger.info(f'Notes: {self.model_notes}')
            self.model_notes.append(notes)
            logger.info(f'Notes: {self.model_notes}')
            return self.model_notes
        except BaseException as exc:
            raise NotImplementedError(f'{exc}')

