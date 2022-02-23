from lib2to3.pytree import Base
from mlvc.ml_model import MlModel
import pytest
import pandas as pd
import numpy as np
class TestIntegration:

    def test_init(self):
        model = MlModel('test')
        assert model.__name__ == 'test'
        
    def test_data_loading_valid_path(self):
        model = MlModel('test')
        df = model.data_loading('tests/data.csv')
        # import pdb;pdb.set_trace()
        assert df.columns.to_list() == ['x','y']

    def test_data_loading_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            model = MlModel('test')
            model.data_loading('tests/extra/data.csv')
            
    def test_data_statistics(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        assert model.data_statistics()

    def test_data_statistics_categorical_data(self):
        model = MlModel('test')
        model.data_loading('tests/data_cat.csv')
        assert model.data_statistics()


    def test_data_hist_plot(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        output = model.data_hist_plot('x')
        assert output == 'matplotlib.figure' 

    def test_normalized_data_plot(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        fig = model.normalized_data_plot(
            col_name = 'x', 
            plot_type = 'bar'
        )
        assert fig

    def test_data_dist_plot(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        fig = model.data_dist_plot(
            col_name = 'x'
        )
        assert fig 

    def test_data_heatmap_plot(self):
        model = MlModel('test')
        model.data_loading('tests/data.csv')
        fig = model.data_heatmap_plot()
        assert fig


    def test_data_categoric_to_binary(self):
        model = MlModel('test')
        model.data_loading('tests/data_mix.csv')
        df = model.data_categoric_to_binary(
           target_name ='is_short',
           col_name = 'height',
           base_value = 'short'
        )
        #import pdb;pdb.set_trace()
        assert df.is_short.tolist() == [1, 0, 0]

    def test_data_categoric_to_binary(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.data_loading('tests/data_mix.csv')
            _= model.data_categoric_to_binary(
                target_name ='is_heavy',
                col_name = 'z',
                base_value = 1
                )

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

    def test_data_build_ml_matrix(self):
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
        