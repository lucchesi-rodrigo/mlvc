from lib2to3.pytree import Base
from mlvc.ml_model import MlModel
import pytest
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


    def test_data_build_ml_matrix(self):
        model = MlModel('test')
        model.data_loading('tests/data_mix.csv')
        ml_data = model.data_build_ml_matrix(
           target_col ='height',
           states_key = ['weight'],
        )
        #import pdb;pdb.set_trace()
        assert list(ml_data.keys()) == ['height', 'weight']