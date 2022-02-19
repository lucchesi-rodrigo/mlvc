from mlvc.ml_model import MlModel
import pytest
class TestIntegration:
    def test_init(self):
        model = MlModel('test')
        assert model.__name__ == 'test'
    def test_load_data_valid_path(self):
        model = MlModel('test')
        df = model.load_data('tests/data.csv')
        # import pdb;pdb.set_trace()
        assert df.columns.to_list() == ['x','y']
    def test_load_data_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            model = MlModel('test')
            model.load_data('tests/extra/data.csv')

    def test_statistics(self):
        model = MlModel('test')
        model.load_data('tests/data.csv')
        assert model.df_statistics()
    
    def test_df_histogram(self):
        model = MlModel('test')
        model.load_data('tests/data.csv')
        assert model.df_hist_plot('x')


#     def test_load_data_valid_path(self):
# model = MlModel('test')
# model.load_data('data.csv')
# import pdb;pdb.set_trace()
# # assert df.columns.to_list() in ['x','y'] 