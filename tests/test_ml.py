import ml_version_control as mlvc 
# import pytest
# class TestIntegration:
#     def test_init(self):
#         model = MLVersionControl('test')
#         print(model)
#     def test_load_data_valid_path(self):
model = mlvc.MLVersionControl('test')
model.load_data('data.csv')
import pdb;pdb.set_trace()
# assert df.columns.to_list() in ['x','y'] 