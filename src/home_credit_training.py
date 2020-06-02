import data_prepare
import data_analysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
import data_prepare
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger=data_prepare.get_logger()

#If the features and targets are already prepared, restore them and train the model directly. 
# Otherwise, the features and targets will be prepared from the scratch. 
training_features_array=data_prepare.get_object('./data/training_features_array.pkl','training_features_array')
training_targets_array=data_prepare.get_object('./data/training_targets_array.pkl','training_targets_array')
test_features_array=data_prepare.get_object('./data/test_features_array.pkl','test_features_array')
test_targets_array=data_prepare.get_object('./data/test_targets_array.pkl','test_targets_array')

if training_features_array is None or training_targets_array is None or test_features_array is None or test_targets_array is None:
    #Phase 01: loading data sets
    logger.info("Phase 01:loading data sets......")
    app_train_dataframe=data_prepare.load_data(app_file="./data/application_train.csv",bureau_file="./data/bureau.csv")
    app_train_dataframe=app_train_dataframe.head(50000)
    app_train_dataframe=data_prepare.derive_data(app_train_dataframe)
    targets_dataframe=data_prepare.get_targets(app_train_dataframe)
    features_dataframe=data_prepare.get_features(app_train_dataframe)
    training_features_dataframe,test_features_dataframe,training_targets_dataframe,test_targets_dataframe=data_prepare.split_dataSets(features_dataframe,targets_dataframe)
    del app_train_dataframe,targets_dataframe,features_dataframe
    logger.info("Phase 01 completed successfully")

    #Phase 02: Pre-Processing training features
    logger.info("Phase 02:prepareing training features......")
    training_features_dataframe=data_prepare.group_days_to_yearly_bins(training_features_dataframe)
    training_features_dataframe,category_dict=data_prepare.set_categorical_columns(training_features_dataframe,category_dict={})
    training_features_dataframe,clipping_dict=data_prepare.clip_outliers(training_features_dataframe,clipping_dict={})
    training_features_dataframe,missing_dict=data_prepare.fill_missing_values(training_features_dataframe,missing_dict={})
    training_features_dataframe=data_prepare.one_hot_encoding(training_features_dataframe)
    training_features_dataframe,nonzero_columns=data_prepare.keep_non_zero_columns(training_features_dataframe,nonzero_columns=None)
    training_features_dataframe,scalerDict=data_prepare.scaling(training_features_dataframe,scalerDict={})
    data_analysis.data_analysis(training_features_dataframe)
    training_features_array,pca=data_prepare.feature_reduction(training_features_dataframe,None)
    del training_features_dataframe
    #!below two lines are exclusive,so one of them needs to be commented
    training_targets_array=training_targets_dataframe.values.ravel()
    #training_features_array,training_targets_array,smote=data_prepare.smoteenn_sampling(training_features_array,training_targets_dataframe.values.ravel(),None)
    training_features_array,npscalerDict=data_prepare.npscaling(training_features_array,npscalerDict={})
    data_prepare.set_parameters(category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict)
    logger.info("Phase 02 completed successfully")

    #Phase 03: Pre-Processing test features
    logger.info("Phase 03:data_prepareing test features......")
    test_features_dataframe=data_prepare.group_days_to_yearly_bins(test_features_dataframe)
    test_features_dataframe,category_dict=data_prepare.set_categorical_columns(test_features_dataframe,category_dict=category_dict)
    test_features_dataframe,clipping_dict=data_prepare.clip_outliers(test_features_dataframe,clipping_dict=clipping_dict)
    test_features_dataframe,missing_dict=data_prepare.fill_missing_values(test_features_dataframe,missing_dict=missing_dict)
    test_features_dataframe=data_prepare.one_hot_encoding(test_features_dataframe)
    test_features_dataframe,nonzero_columns=data_prepare.keep_non_zero_columns(test_features_dataframe,nonzero_columns=nonzero_columns)
    test_features_dataframe,scalerDict=data_prepare.scaling(test_features_dataframe,scalerDict=scalerDict)
    test_features_array,pca=data_prepare.feature_reduction(test_features_dataframe,pca)
    del test_features_dataframe
    test_features_array,npscalerDict=data_prepare.npscaling(test_features_array,npscalerDict={})
    logger.info("Phase 03  completed successfully")

    #Phase 04: Check the similarity between test and train data
    logger.info("Phase 04  Check the similarity between test and train data......")
    data_prepare.check_similarity(training_features_array,test_features_array)
    #Save data so that we can skip the data_preparing steps
    test_targets_array=test_targets_dataframe.values.ravel()
    data_prepare.save_object('./data/training_features_array.pkl',training_features_array,'training_features_array')
    data_prepare.save_object('./data/training_targets_array.pkl',training_targets_array,'training_targets_array')
    data_prepare.save_object('./data/test_features_array.pkl',test_features_array,'test_features_array')
    data_prepare.save_object('./data/test_targets_array.pkl',test_targets_array,'test_targets_array')
    logger.info("Phase 04  completed successfully")

#Phase 05:Training and Validation
logger.info('Phase 05:Modelling......')
training_features,validation_features,training_targets,validation_targets=train_test_split(training_features_array,training_targets_array,train_size=0.8,test_size=0.2,random_state=1,shuffle=True,stratify=training_targets_array)
train_roc_auc_score_best=0
valid_roc_auc_score_best=0
test_roc_auc_score_best=0
for model_name in ('LinearRegressionClassifer','RandomForestClassifier','LightGBM'):
    if model_name=='LinearRegressionClassifer':
        logger.info("SGD for Linear Regression......")
        clf=SGDClassifier(loss='log',learning_rate='invscaling',eta0=0.0001,tol=10e-8,class_weight='balanced',n_jobs=-1)
        param1_name='alpha'
        param1_range=[0.0003]
        param2_name='eta0'  
        param2_range=[0.003]
    elif model_name=='RandomForestClassifier':
        logger.info("Random Forest Tree......")
        clf=RandomForestClassifier(max_features=20,class_weight='balanced',random_state=50,oob_score = True,n_jobs=-1)
        param1_name='n_estimators'
        param1_range=np.arange(1000,1001,2000)
        param2_name='min_samples_leaf'
        param2_range=np.arange(100,201,50)
    elif model_name=='LightGBM':
        clf = LGBMClassifier(n_estimators=10000,learning_rate=0.02,num_leaves=34,colsample_bytree=0.9497036,subsample=0.8715623,max_depth=8,reg_alpha=0.041545473,reg_lambda=0.0735294,min_split_gain=0.0222415,min_child_weight=39.3259775,n_jobs=-1)
        param1_name='n_estimators'
        param1_range=np.arange(10000,10001,1)
        param2_name='learning_rate'
        param2_range=[0.02]
    else:
        logger.error("Wrong model!")
        exit()

    logger.info('param1_name:{},param1_range:{},param2_name:{},param2_range:{}'.format(param1_name,param1_range,param2_name,param2_range))

    for p1 in param1_range:
        for p2 in param2_range:
            params={param1_name:p1,param2_name:p2}
            clf.set_params(**params)
            clf.fit(training_features,training_targets.ravel())

            train_y_pred_proba =clf.predict_proba(training_features)[:,1]
            train_roc_auc_score=roc_auc_score(training_targets.ravel(), train_y_pred_proba)
            validation_y_pred_proba =clf.predict_proba(validation_features)[:,1]
            validation_roc_auc_score=roc_auc_score(validation_targets.ravel(), validation_y_pred_proba)
            test_y_pred_proba =clf.predict_proba(test_features_array)[:,1]
            test_roc_auc_score=roc_auc_score(test_targets_array.ravel(), test_y_pred_proba)

            logger.info('{}={},{}={},train_roc_auc_score:{},validation_roc_auc_score:{},test_roc_auc_score:{}'.format(param1_name,p1,param2_name,p2,train_roc_auc_score,validation_roc_auc_score,test_roc_auc_score))
            
            #save the best model
            if test_roc_auc_score > test_roc_auc_score_best:
                logger.info('New test roc_auc_score benchmark:test_roc_auc_score:{},test_roc_auc_score_best:{}'.format(test_roc_auc_score,test_roc_auc_score_best))
                test_roc_auc_score_best = test_roc_auc_score
                model_filename="./model/best_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(clf, f)
                    logger.info("Model has been written to " + model_filename)
