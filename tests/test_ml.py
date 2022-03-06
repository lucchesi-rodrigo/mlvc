from lib2to3.pytree import Base
from mlvc.create_ml_model import CreateMlModel
import os
import pytest
import pandas as pd
import numpy as np

from datetime import datetime,date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

class TestIntegration:

    #---CreateMlModel 
    def test_init(self):
        model = CreateMlModel('test')
        assert model.__name__ == 'test'
    #---data_loading   
    def test_data_loading_exception(self):
        """Invalid path"""
        with pytest.raises(FileNotFoundError):
            model = CreateMlModel('test')
            model.data_loading('tests/extra/data.csv')

    def test_data_loading(self):
        """Loads csv file"""
        model = CreateMlModel('test')
        df = model.data_loading('tests/data.csv')
        assert df.columns.to_list() == ['x','y']
    #---data_analysis     
    def test_data_analysis(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data.csv')
        assert model.data_analysis()

    def test_data_analysis_categorical_data(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data_cat.csv')
        assert model.data_analysis()

    def test_data_analysis_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = None
            model.data_analysis()
    #---isolate_categ_and_num_cols
    def test_isolate_categ_and_num_cols(self):
        """Isolate categoric and numeric collumns from df"""
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
        model = CreateMlModel('test')
        model.df = df
        numeric_cols, categoric_cols = model.isolate_categ_and_num_cols()
        assert numeric_cols == ['max_speed']
        assert categoric_cols == ['class', 'order']

    def test_isolate_categ_and_num_cols_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = None
            model.isolate_categ_and_num_cols()  
    #---remove_cols
    def test_remove_cols(self):
        model = CreateMlModel('test')
        cols_list = model.remove_cols(['x','y','i'],['i','g'])
        assert cols_list == ['x','y']
    #---data_hist_plot
    def test_data_hist_plot(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data.csv')
        output = model.data_hist_plot('x')
        assert output == 'matplotlib.figure' 

    def test_data_hist_plot_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = pd.DataFrame()
            model.data_hist_plot('x')
    #---normalized_data_plot    
    def test_normalized_data_plot(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data.csv')
        fig = model.normalized_data_plot(
            col_name = 'x', 
            plot_type = 'bar'
        )
        assert fig

    def test_normalized_data_plot_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = pd.DataFrame()
            model.normalized_data_plot(
                col_name = 'x', 
                plot_type = 'bar'
            )
    #---data_dist_plot
    def test_data_dist_plot(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data.csv')
        model.data_dist_plot(
            col_name = 'x'
        ) 

    def test_data_dist_plot_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = pd.DataFrame()
            model.data_dist_plot(col_name = 'x')
    #---data_heatmap_plot
    def test_data_heatmap_plot(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data.csv')
        model.data_heatmap_plot()

    def test_data_heatmap_plot_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
            model.df = pd.DataFrame()
            model.data_heatmap_plot()
    #---data_categoric_to_binary
    def test_data_categoric_to_binary(self):
        model = CreateMlModel('test')
        model.data_loading('tests/data_mix.csv')
        df = model.data_categoric_to_binary(
           target_name ='is_short',
           col_name = 'height',
           base_value = 'short'
        )
        assert df.is_short.tolist() == [1, 0, 0]

    def test_data_categoric_to_binary_exception(self):
        with pytest.raises(BaseException):
            model = CreateMlModel('test')
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

        model = CreateMlModel('test')
        model.df = df
        model.data_feature_encoder(col_name=col_name, target_col=target_col)
        assert sorted(model.df.columns.tolist()) == sorted(["class", "order", "max_speed",f"{col_name}_{target_col}"])

    def test_data_feature_encoder_exception(self):
        with pytest.raises(BaseException):
            col_name='order'
            target_col='max_speed'
            model = CreateMlModel('test')
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

        model = CreateMlModel('test')
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

            model = CreateMlModel('test')
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

        model = CreateMlModel('test')
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

            model = CreateMlModel('test')
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

        model = CreateMlModel('test')
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

            model = CreateMlModel('test')
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
        models = [(model_data_1,False),(model_data_2,False)] # True to exception
        model.tp_rate_analysis(ml_models=models)

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

        model = CreateMlModel('test')
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
        model.feature_importance_plot_1(model_data=model_data)

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

            model = CreateMlModel('test')
            model.df = df
            model.data_build_ml_matrix(target_col=target_col,states_key=states_key)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model.split_test_train_data(test_size=0.3, random_state=11)
            model_data = None
            model.feature_importance_plot_1(model_data=model_data)
    #---feature_importance_plot_2
    def test_feature_importance_plot_2(self):
        pass
    def test_feature_importance_plot_2_exception(self):
        pass
    #---clf_report
    def test_clf_report(self):
        pass
    def test_clf_report_exception(self):
        pass
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
        model_data = model.fit_predict(
                model_data={'name':'lrc'},
                model_algorithm=LogisticRegression()
            )
        model_data['name'] = 'random_forest_classifier'
        model.saving(model_data)
    def test_saving_exception(self):
        pass
    #---loading
    def test_loading(self):
        pass
    def test_loading_exception(self):
        pass