import ml_version_control as mlvc 

# 1 test
model = mlvc.MLVersionControl('test')
model.load_data('tests/data.csv')
stats = model.df_statistics()
import pdb;pdb.set_trace()
