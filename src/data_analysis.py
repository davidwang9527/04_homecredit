import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from data_prepare import get_logger

#check duplicates,describe info, pearson correlations and PCA analysis
def data_analysis(app_train_dataframe):
    logger=get_logger()
    
    #check duplicates
    logger.info("Checking duplicates......")
    if 'SK_ID_BUREAU' in app_train_dataframe.columns:
        duplicated_index_columns=['SK_ID_CURR','SK_ID_BUREAU']
    else:
        duplicated_index_columns=['SK_ID_CURR']

    df=app_train_dataframe[app_train_dataframe.duplicated(duplicated_index_columns)]
    
    if df.shape[0]!=0:
       logger.info("There're some duplicates in the training set!")
       return False
    else: 
      logger.info("There's no duplicate in the training set!")
    
    #check describe info
    #app_train_dataframe.describe(include='all').to_excel('./doc/application_describe.xlsx')
    #logger.info("Describe info has been written to ./doc/application_describe.xlsx")

    #check pearson correlations
    #correlation_dataframe = app_train_dataframe.copy()
    #correlation_dataframe.corr().to_excel('./doc/application_correlation.xlsx')
    #logger.info("Correlation info has been written to ./doc/application_correlation.xlsx")
    
    #check PCA analysis
    logger.info("PCA Understanding......")
    exclude_columns=['SK_ID_CURR']
    if 'SK_ID_BUREAU' in app_train_dataframe.columns:
          exclude_columns=exclude_columns + ['SK_ID_BUREAU']
    if 'TARGET' in app_train_dataframe.columns:
          exclude_columns=exclude_columns+['TARGET']
    features_dataframe=app_train_dataframe.drop(columns=exclude_columns)

    X=features_dataframe.values
    
    pca = PCA()
    pca.fit(X)
    logger.info('PCA explained variance ratio: %s'% str(pca.explained_variance_ratio_))
    logger.info('PCA explained variance ratio cumsum: %s'% str(pca.explained_variance_ratio_.cumsum()))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('./doc/pca.png')
    plt.close()
    logger.info("pca.png has been written to ./doc/pca.png")
    
    return True
