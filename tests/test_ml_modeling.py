from lib2to3.pytree import Base
from mlvc.modeling import MlModeling
import os
import random
import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime,date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

class TestMlModeling:

    #---CreateMlModel 
    def test_init(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        assert mlm.model_data['model_name'] == 'lrc'
        assert mlm.model_data['model_algorithm'].__class__.__name__ == 'LogisticRegression'
        assert mlm.model_data['model_version'] == '0.1'

    def test_init_exception_model_name_none(self):
            with pytest.raises(AssertionError):
                mlm = MlModeling(model_name= None, model_algorithm=LogisticRegression(), model_version='0.1')  

    def test_init_exception_model_algorithm_none(self):
        with pytest.raises(AssertionError):
            mlm = MlModeling(model_name='lrc', model_algorithm=None, model_version='0.1')
    #---data_loading   
    def test_data_loading_exception(self):
        """Invalid path"""
        with pytest.raises(FileNotFoundError):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.data_loading('tests/data/data_error.csv')
            
    def test_data_loading(self):
        """Loads csv file"""
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data.csv')
        assert mlm.df.columns.to_list() == ['x','y']
    #---data_analysis     
    def test_data_analysis(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data.csv')
        stats_data = mlm.data_analysis()
        assert json.loads(stats_data['numeric_stats']) == {
            'x': {'count': 2.0, 'mean': 0.5, 'std': 0.7071067812, 'min': 0.0, '25%': 0.25, '50%': 0.5, '75%': 0.75, 'max': 1.0}, 
            'y': {'count': 2.0, 'mean': 0.5, 'std': 0.7071067812, 'min': 0.0, '25%': 0.25, '50%': 0.5, '75%': 0.75, 'max': 1.0}
            }

    def test_data_analysis_categorical_data(self):
        random.seed(10)
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data_cat.csv')
        stats_data = mlm.data_analysis()
        assert stats_data['shape'] == (2,2)

    def test_data_analysis_exception(self):
        with pytest.raises(BaseException):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.df = None
            mlm.data_analysis()
    #---isolate_categ_and_num_cols
    def test_isolate_categ_and_num_cols(self):
        """Isolate categoric and numeric columns from df"""
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", np.nan),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.df = df
        numeric_cols, categoric_cols = mlm.isolate_categ_and_num_cols()
        assert numeric_cols == ['max_speed']
        assert categoric_cols == ['class', 'order']

    def test_isolate_categ_and_num_cols_exception(self):
        with pytest.raises(BaseException):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.df = None
            mlm.isolate_categ_and_num_cols()  
    #---remove_cols
    def test_remove_cols(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        cols_list = mlm.remove_cols(['x','y','i'],['i','g'])
        assert cols_list == ['x','y']
    #---data_hist_plot
    def test_data_hist_plot(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data.csv')
        assert mlm.data_hist_plot('x') is None

    def test_data_hist_plot_exception(self):
        with pytest.raises(BaseException):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.df = None
            mlm.data_hist_plot('x')
    #---normalized_data_plot    
    def test_normalized_data_plot(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data.csv')
        assert mlm.normalized_barplots_data_plot(
            col_name = 'x', 
        ) is None
        
    def test_normalized_data_plot_exception(self):
        with pytest.raises(BaseException):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.df = None
            mlm.normalized_barplots_data_plot(
                col_name = 'x', 
            )
    #---data_dist_plot
    def test_data_dist_plot(self):
        mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
        mlm.data_loading('tests/data/data.csv')
        assert mlm.data_dist_plot(
            col_name = 'x'
        ) is None

    def test_data_dist_plot_exception(self):
        with pytest.raises(BaseException):
            mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
            mlm.df =None
            mlm.data_dist_plot(
                col_name = 'x'
            )
    #---data_heatmap_plot
    def test_data_heatmap_plot(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        output = model.data_heatmap_plot()
        assert output == 'matplotlib.figure'

    def test_data_heatmap_plot_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.df = pd.DataFrame()
            model.data_heatmap_plot()
    #---data_categoric_to_binary
    def test_data_categoric_to_binary(self):
        model = MlModel('test')
        model.data_loading('tests/data_mix.csv')
        df = model.data_categoric_to_binary(
           target_name ='is_short',
           col_name = 'height',
           base_value = 'short'
        )
        assert df.is_short.tolist() == [1, 0, 0]

    def test_data_categoric_to_binary_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.data_loading('tests/data_mix.csv')
            _= model.data_categoric_to_binary(
                target_name ='is_heavy',
                col_name = 'z',
                base_value = 1
                )
    #---data_feature_encoder
    def test_data_feature_encoder(self):
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", np.nan),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        col_name='order'
        target_col='max_speed'

        model = MlModel('test')
        model.df = df
        model.data_feature_encoder(col_name=col_name, target_col=target_col)
        assert sorted(model.df.columns.tolist()) == sorted(["class", "order", "max_speed",f"{col_name}_{target_col}"])

    def test_data_feature_encoder_exception(self):
        with pytest.raises(BaseException):
            col_name='order'
            target_col='max_speed'
            model = MlModel('test')
            model.df = None
            model.data_feature_encoder(col_name=col_name, target_col=target_col)
    #---data_build_ml_matrix
    def test_data_build_ml_matrix(self):
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", 0),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        states_key=["class", "order"]
        target_col='max_speed'

        model = MlModel('test')
        model.df = df
        ml_data = model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        assert model.y.to_list() == [389.0, 24.0, 80.2, 0, 58.0]
        assert model.X.columns.to_list() == ["class", "order"]

    def test_data_build_ml_matrix_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    ("bird", "Falconiformes", 389.0),
                    ("bird", "Psittaciformes", 24.0),
                    ("mammal", "Carnivora", 80.2),
                    ("mammal", "Primates", 0),
                    ("mammal", "Carnivora", 58),
                ],
                index=["falcon", "parrot", "lion", "monkey", "leopard"],
                columns=("class", "order", "max_speed"),
            )
            states_key=["z", "x"]
            target_col='y'

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
    #---split_test_train_data
    def test_split_test_train_data(self):

        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", 0),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        states_key=["class", "order"]
        target_col='max_speed'

        model = MlModel('test')
        model.df = df
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        data_processed = model.split_test_train_data(test_size=0.3, random_state=11)
        X_train, X_test, y_train, y_test = data_processed
        assert X_train.shape[0] > 1
        assert X_test.shape[0] > 1
        assert y_train.shape[0] > 1
        assert y_test.shape[0] > 1

    def test_split_test_train_data_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    ("bird", "Falconiformes", 389.0),
                    ("bird", "Psittaciformes", 24.0),
                    ("mammal", "Carnivora", 80.2),
                    ("mammal", "Primates", 0),
                    ("mammal", "Carnivora", 58),
                ],
                index=["falcon", "parrot", "lion", "monkey", "leopard"],
                columns=("class", "order", "max_speed"),
            )
            states_key=["class", "order"]
            target_col='max_speed'

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            data_processed = model.split_test_train_data(test_size=0.9, random_state=11)
    #---fit_predict
    def test_fit_predict(self):
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", 0),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        states_key=["class", "order"]
        target_col='max_speed'

        model = MlModel('test')
        model.df = df
        model.data_categoric_to_binary(
           target_name ='class',
           col_name = 'class',
           base_value = 'mammal'
        )
        model.data_categoric_to_binary(
           target_name ='order',
           col_name = 'order',
           base_value = 'Falconiformes'
        )
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict(
                model_name='lrc',
                model_algorithm=LogisticRegression()
            )
        
        y_train_predicted = model_data['y_train_predicted']
        y_train_predicted = y_train_predicted.tolist()
        y_test_predicted = model_data['y_test_predicted']
        y_test_predicted = y_test_predicted.tolist()
        assert y_test_predicted == [0.0, 0.0] 
        assert y_train_predicted == [389.0, 0.0, 24.0]        
        
    def test_fit_predict_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    ("bird", "Falconiformes", 389.0),
                    ("bird", "Psittaciformes", 24.0),
                    ("mammal", "Carnivora", 80.2),
                    ("mammal", "Primates", 0),
                    ("mammal", "Carnivora", 58),
                ],
                index=["falcon", "parrot", "lion", "monkey", "leopard"],
                columns=("class", "order", "max_speed"),
            )
            states_key=["class", "order"]
            target_col='max_speed'

            model = MlModel('test')
            model.df = df
            model.data_categoric_to_binary(
            target_name ='class',
            col_name = 'class',
            base_value = 'mammal'
            )
            model.data_categoric_to_binary(
            target_name ='order',
            col_name = 'order',
            base_value = 'Falconiformes'
            )
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.fit_predict(
                    model_data={'name':'lrc'},
                    model_algorithm=None
                )
    #---fit_predict_to_best_estimator
    def test_fit_predict_to_best_estimator(self):
        df = pd.DataFrame(
            [
                (100, 188, 0),
                (70, 170, 1),
                (60, 170, 0),
                (80, 188, 1),
                (67, 166, 0),
                (66, 166, 1),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
            ],
            columns=("over_weight", "height", "age"),
        )
        states_key=["height", "age"]
        target_col='over_weight'
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        model = MlModel('test')
        model.df = df
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict_to_best_estimator(
                    model_name='rfc',
                    model_algorithm=RandomForestClassifier(),
                    param_grid=param_grid,
                    folds=2,
                )
        assert model_data['model_name'] == 'rfc'
        np.testing.assert_allclose(model_data['y_train_predicted'].tolist(),[67, 100, 67, 67, 100, 70, 70, 70, 70, 100, 70, 67],atol=20)
        np.testing.assert_allclose(model_data['y_test_predicted'].tolist(),[100, 100, 100, 67, 66, 70],atol=20)

    def test_fit_predict_to_best_estimator_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [RandomForestClassifier(max_depth=4, max_features='sqrt', n_estimators=200)
                    (100, 188, 0),
                    (70, 170, 1),
                    (60, 170, 0),
                    (80, 188, 1),
                    (67, 166, 0),
                    (66, 166, 1),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                ],
                columns=("over_weight", "height", "age"),
            )
            states_key=["height", "age"]
            target_col='over_weight'
            param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
            }

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.fit_predict_to_best_estimator(
                        model_name='rfc',
                        model_algorithm=None,
                        param_grid=param_grid,
                        folds=2,
                    )
    #---tp_rate_analysis TODO: Not working       
    def test_tp_rate_analysis(self):
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 1),
                ("bird", "Psittaciformes", 1),
                ("mammal", "Carnivora", 0),
                ("mammal", "Primates", 0),
                ("mammal", "Carnivora", 0),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "heavy"),
        )
        states_key=["class", "order"]
        target_col='heavy'

        model = MlModel('test')
        model.df = df
        model.data_categoric_to_binary(
           target_name ='class',
           col_name = 'class',
           base_value = 'mammal'
        )
        model.data_categoric_to_binary(
           target_name ='order',
           col_name = 'order',
           base_value = 'Falconiformes'
        )
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data_1 = model.fit_predict(
                model_name='lrc',
                model_algorithm=LogisticRegression()
            )
        model_data_2 = model.fit_predict(
                model_name='dtc',
                model_algorithm=tree.DecisionTreeClassifier()
            )
        models = [(model_data_1,False),(model_data_2,False)] # True to exception
        output = model.tp_rate_analysis(ml_models=models)
        assert output.__name__ == 'matplotlib.pyplot'

    """def test_tp_rate_analysis_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    ("bird", "Falconiformes", 1),
                    ("bird", "Psittaciformes", 1),
                    ("mammal", "Carnivora", 0),
                    ("mammal", "Primates", 0),
                    ("mammal", "Carnivora", 0),
                ],
                index=["falcon", "parrot", "lion", "monkey", "leopard"],
                columns=("class", "order", "heavy"),
            )
            states_key=["class", "order"]
            target_col='heavy'

            model = CreateMlModel('test')
            model.df = df
            model.data_categoric_to_binary(
            target_name ='class',
            col_name = 'class',
            base_value = 'mammal'
            )
            model.data_categoric_to_binary(
            target_name ='order',
            col_name = 'order',
            base_value = 'Falconiformes'
            )
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model_data_1 = model.fit_predict(
                    model_name='lrc',
                    model_algorithm=LogisticRegression()
                )
            model_data_2 = model.fit_predict(
                    model_name='dtc',
                    model_algorithm=tree.DecisionTreeClassifier()
                )
            models = [] # True to exception
            model.tp_rate_analysis(ml_models=models)""" 
    #---feature_importance_plot_1
    def test_feature_importance_plot_1(self):
        df = pd.DataFrame(
            [
                (100, 188, 0),
                (70, 170, 1),
                (60, 170, 0),
                (80, 188, 1),
                (67, 166, 0),
                (66, 166, 1),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
            ],
            columns=("over_weight", "height", "age"),
        )
        states_key=["height", "age"]
        target_col='over_weight'
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        model = MlModel('test')
        model.df = df
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict_to_best_estimator(
                    model_name='rfc',
                    model_algorithm=RandomForestClassifier(),
                    param_grid=param_grid,
                    folds=2,
                )
        response = model.feature_importance_plot_1(model_data=model_data)
        assert response.__name__ == 'matplotlib.pyplot'

    def test_feature_importance_plot_1_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    (100, 188, 0),
                    (70, 170, 1),
                    (60, 170, 0),
                    (80, 188, 1),
                    (67, 166, 0),
                    (66, 166, 1),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                ],
                columns=("over_weight", "height", "age"),
            )
            states_key=["height", "age"]
            target_col='over_weight'
            param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
            }

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model_data = None
            model.feature_importance_plot_1(model_data=model_data)
            

    #---feature_importance_plot_2
    def test_feature_importance_plot_2(self):
        df = pd.DataFrame(
            [
                (100, 188, 0),
                (70, 170, 1),
                (60, 170, 0),
                (80, 188, 1),
                (67, 166, 0),
                (66, 166, 1),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
            ],
            columns=("over_weight", "height", "age"),
        )
        states_key=["height", "age"]
        target_col='over_weight'
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        model = MlModel('test')
        model.df = df
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict_to_best_estimator(
                    model_name='rfc',
                    model_algorithm=RandomForestClassifier(),
                    param_grid=param_grid,
                    folds=2,
                )
        response = model.feature_importance_plot_2(model_data=model_data)
        assert response.__name__ == 'shap'

    def test_feature_importance_plot_2_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    (100, 188, 0),
                    (70, 170, 1),
                    (60, 170, 0),
                    (80, 188, 1),
                    (67, 166, 0),
                    (66, 166, 1),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                ],
                columns=("over_weight", "height", "age"),
            )
            states_key=["height", "age"]
            target_col='over_weight'
            param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
            }

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model_data = None
            model.feature_importance_plot_2(model_data=model_data)
    #---clf_report
    def test_clf_report(self):
        df = pd.DataFrame(
            [
                (100, 188, 0),
                (70, 170, 1),
                (60, 170, 0),
                (80, 188, 1),
                (67, 166, 0),
                (66, 166, 1),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
                (100, 188, 1),
                (70, 170, 0),
                (60, 170, 1),
                (80, 188, 0),
                (67, 166, 1),
                (66, 166, 0),
            ],
            columns=("over_weight", "height", "age"),
        )
        states_key=["height", "age"]
        target_col='over_weight'
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        model = MlModel('test')
        model.df = df
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict_to_best_estimator(
                    model_name='rfc',
                    model_algorithm=RandomForestClassifier(),
                    param_grid=param_grid,
                    folds=2,
                )
        response = model.clf_report(model_data=model_data)
        assert response.__name__ == 'matplotlib.pyplot'
    def test_clf_report_exception(self):
        with pytest.raises(BaseException):
            df = pd.DataFrame(
                [
                    (100, 188, 0),
                    (70, 170, 1),
                    (60, 170, 0),
                    (80, 188, 1),
                    (67, 166, 0),
                    (66, 166, 1),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                    (100, 188, 1),
                    (70, 170, 0),
                    (60, 170, 1),
                    (80, 188, 0),
                    (67, 166, 1),
                    (66, 166, 0),
                ],
                columns=("over_weight", "height", "age"),
            )
            states_key=["height", "age"]
            target_col='over_weight'
            param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
            }

            model = MlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model_data = None
            model.clf_report(model_data=model_data)
    #---saving
    def test_saving(self):
        df = pd.DataFrame(
            [
                ("bird", "Falconiformes", 389.0),
                ("bird", "Psittaciformes", 24.0),
                ("mammal", "Carnivora", 80.2),
                ("mammal", "Primates", 0),
                ("mammal", "Carnivora", 58),
            ],
            index=["falcon", "parrot", "lion", "monkey", "leopard"],
            columns=("class", "order", "max_speed"),
        )
        states_key=["class", "order"]
        target_col='max_speed'

        model = MlModel('test')
        model.df = df
        model.data_categoric_to_binary(
           target_name ='class',
           col_name = 'class',
           base_value = 'mammal'
        )
        model.data_categoric_to_binary(
           target_name ='order',
           col_name = 'order',
           base_value = 'Falconiformes'
        )
        model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
        model.split_test_train_data(test_size=0.3, random_state=11)
        model_data = model.fit_predict(
                model_algorithm=LogisticRegression()
            )
        model_data['model_name'] = 'random_forest_classifier'
        output = model.saving(model_data)
        assert output.__name__ == 'joblib'

    def test_saving_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model_data = None
            model.saving(model_data)
    #---loading
    def test_loading(self):
        model = MlModel('test')
        model_data = model.loading('models/random_forest_classifier_2022')
        assert model_data['model_name'] == 'random_forest_classifier'
    def test_loading_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.loading('models/random_forest_classifier_2021')
        