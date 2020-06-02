import numpy as np
import pandas as pd
import data_prepare
import pickle

#Phase01 : loading parameters
logger=data_prepare.get_logger()
logger.info("Phase 01:loading parameters......")
category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict=data_prepare.get_parameters()

#Phase 02: loading data sets
logger.info("Phase 02:loading data sets......")
app_predict_dataframe=data_prepare.load_data(app_file="./data/application_test.csv",bureau_file="./data/bureau.csv")
app_predict_dataframe=data_prepare.derive_data(app_predict_dataframe)
ori_app_predict_dataframe=app_predict_dataframe.copy()
logger.info("Phase 01 completed successfully")

#Phase 03: Pre-Processing test features
logger.info("Phase 03:data_prepareing test features......")
app_predict_dataframe=data_prepare.group_days_to_yearly_bins(app_predict_dataframe)
app_predict_dataframe,category_dict=data_prepare.set_categorical_columns(app_predict_dataframe,category_dict=category_dict)
app_predict_dataframe,clipping_dict=data_prepare.clip_outliers(app_predict_dataframe,clipping_dict=clipping_dict)
app_predict_dataframe,missing_dict=data_prepare.fill_missing_values(app_predict_dataframe,missing_dict=missing_dict)
app_predict_dataframe=data_prepare.one_hot_encoding(app_predict_dataframe)
app_predict_dataframe,nonzero_columns=data_prepare.keep_non_zero_columns(app_predict_dataframe,nonzero_columns=nonzero_columns)
app_predict_dataframe,scalerDict=data_prepare.scaling(app_predict_dataframe,scalerDict=scalerDict)
app_predict_features_array,pca=data_prepare.feature_reduction(app_predict_dataframe,pca)
predict_features_array,npscalerDict=data_prepare.npscaling(app_predict_features_array,npscalerDict=npscalerDict)
del app_predict_dataframe
logger.info("Phase 03 completed successfully")

#Phase 04:Predicting
logger.info('Phase 04:Predicting......')
try:
    with open('./model/best_model.pkl', "rb") as f:
        clf=pickle.load(f)
except:
    logger.error('There is no model yet, please train the model before trying to predict!')
    exit()

pp=clf.predict_proba(app_predict_features_array)
app_predict_dataframe=ori_app_predict_dataframe[['SK_ID_CURR']]
app_predict_dataframe=app_predict_dataframe.assign(TARGET=pd.Series(pp[:,1]))
predict_filename="./predict/home_credit_predict.csv"
app_predict_dataframe.to_csv(predict_filename,index=False)
logger.info('Phase 04:Predicting Completed successfully')
