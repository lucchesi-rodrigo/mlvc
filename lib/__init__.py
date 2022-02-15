"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string


# import libraries
import os
import logging as log
import pandas as pd
import churn_library_solution as cls
import matplotlib.pyplot as plt
import seaborn as sns

log.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class MLVersionControl:
    def __init__(self, name):
        self.name = name

    def load_data(self, df_path: str) -> pd.DataFrame:
        """
        Returns dataframe for the csv found at path

        Parameters
        ----------
        df_path: str
            A path to the csv file

        Returns:
        --------
        df: pd.DataFrame
            A pandas dataframe

        Examples:
        ---------
            >>> df = load_data('path/file.csv')
        """
        try:
            self.df = pd.read_csv(df_path,index=False)
            log.info(
                f'SUCCESS: import_data({df_path}) -> msg: dataframe read successfully -> df:' f'{df.head().to_dict()}'
                )
            return self.df
        except FileNotFoundError as exc:
            log.error(f'ERROR: import_data({df_path}) -> Exception: {exc}')

    def perform_eda_pipeline(self):
        """
        Perform eda pipeline on df and save figures to images folder

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used in ml project

        Returns:
        --------
        None

        Examples:
        ---------
            >>> perform_eda(df)
        """

        plt.figure(figsize=(20,10))

    def df_statitics(self):
        """
        Perform eda pipeline on df and save figures to images folder

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used in ml project

        Returns:
        --------
        None

        Examples:
        ---------
            >>> perform_eda(df)
        """
        self.stats_data = {
            shape: self.df.shape,
            null_values: self.df.isnull().sum(),
            numeric_stats: self.df.describe()
        }
        log.info(
            f''
            )

    def ml_target_analysis(self, target_name: str, col_name: str,condition: str) -> None:
        """
        Perform eda pipeline on df and save figures to images folder

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used in ml project

        Returns:
        --------
        None

        Examples:
        ---------
            >>> perform_eda(df)
        """
        #self.df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        self.df[target_name] = self.df[col_name].apply(lambda val: 0 if val == condition else 1)

    def df_hist(self,col_name: str) -> None:
        #self.df['Churn'].hist();
        #self.df['Customer_Age'].hist();
        self.df[col_name].hist();

    def df_bar(self,col_name: str, plot_type: str) -> None:
        #df.Marital_Status.value_counts('normalize').plot(kind='bar');
        self.df[col_name].value_counts('normalize').plot(kind=plot_type);

    def df_heatmap(self,col_name: str) -> None:
        #sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)

    def encoder_helper(self, category, target_col, response):
        """"""
        category_lst = []
        category_groups = self.df.groupby(category).mean()[target_col]

        for val in self.df[category]:
            category_lst.append(category_groups.loc[val])

        self.df[f'{category}_{target_col}'] = category_lst

    def target_instance_matrix(self, target_col):
        self.y = self.df[target_col]
        self.X = pd.DataFrame()

        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                    'Income_Category_Churn', 'Card_Category_Churn']

        self.X[keep_cols] = self.df[keep_cols]
        return {'X':self.X,'y':self.y}

    def train_test(self):
        """"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size= 0.3,
            random_state=42)

    def best_models(self,classifier_model: Object):
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

    def model_performance(self):
        # Check book pipeline names
        # scores
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))

        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))

    def classification_report_image(
        self,y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
            """
            produces classification report for training and testing results and stores report as image
            in images folder
            input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

            output:
                    None
            """
        pass


    def feature_importance_plot(self,model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        pass

    def train_models(self,X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        pass
