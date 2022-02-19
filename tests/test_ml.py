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
    def test_df_hist(self):
        model = MlModel('test')
        model.load_data('tests/data.csv')
        assert model.df_hist_plot('x')
    def test_df_hist_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.load_data('tests/data.csv')
            model.df_bar_plot('z')
    def test_df_bar_plot(self):
        model = MlModel('test')
        model.load_data('tests/data.csv')
        assert model.df_bar_plot(
            col_name = 'x', 
            plot_type = 'bar'
        )
    def test_df_bar_plot_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.load_data('tests/data.csv')
            model.df_bar_plot(
                col_name = 'z', 
                plot_type = 'bar'
            )    
    def test_df_heatmap_plot(self):
        model = MlModel('test')
        model.load_data('tests/data.csv')
        assert model.df_heatmap_plot()
    def test_df_heatmap_plot_exception(self):
        with pytest.raises(BaseException):
            model = MlModel('test')
            model.load_data('tests/data_empty.csv')
            model.df_heatmap_plot()

    def test_df_col_categoric_to_binary(self):
        model = MlModel('test')
        model.load_data('tests/data_cat.csv')
        df = model.df_col_categoric_to_binary(
           target_name ='is_short',
           col_name = 'height',
           condition = 'short'
        )
        #import pdb;pdb.set_trace()
        assert df.is_short.to_list() == [1, 0, 0]

#     def test_load_data_valid_path(self):
# model = MlModel('test')
# model.load_data('data.csv')
# import pdb;pdb.set_trace()
# # assert df.columns.to_list() in ['x','y'] 