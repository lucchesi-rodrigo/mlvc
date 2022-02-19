from mlvc.ml_model import MlModel

if __name__ == '__main__':

    model = MlModel('Test')
    model.eda(
        df_path='tests/data_cat.csv',
        target_name='is_short',
        col_name=['is_short','height'],
        condition='short',
        plot_type=['bar']
        )