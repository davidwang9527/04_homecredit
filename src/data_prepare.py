import datetime
import logging
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

#get logger which writes to both ./model/homeCredit{YYYYMMDD}.log and stdout
def get_logger():
    log_filename='./log/homeCredit{}.log'.format(datetime.date.today().strftime('%Y%m%d'))
    logging.basicConfig(
		level=logging.INFO, 
		format='[%(asctime)s] %(levelname)s - %(message)s',
		handlers=[logging.FileHandler(filename=log_filename),logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()
    return logger

#Load appfile and bereau data into dataframe
def load_data(app_file,bureau_file):
    logger=get_logger()
    logger.info("Loading data......")
    app_dataframe= pd.read_csv(app_file)
    drop_columns=['FLAG_MOBIL','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT']
    app_dataframe=app_dataframe.drop(columns=drop_columns)
    app_dataframe=app_dataframe.reindex(np.random.permutation(app_dataframe.index))
    return app_dataframe

#derive some columns
def derive_data(app_dataframe):
    app_dataframe['ANNUITY_INCOME_PERC'] = app_dataframe['AMT_ANNUITY'] / app_dataframe['AMT_INCOME_TOTAL']
    app_dataframe['PAYMENT_RATE'] = app_dataframe['AMT_ANNUITY'] / app_dataframe['AMT_CREDIT']
    return app_dataframe

#get Targets
def get_targets(app_dataframe):
    logger=get_logger()
    logger.info("getting targets......")
    targets_dataframe = pd.DataFrame()
    targets_dataframe["TARGET"]=app_dataframe['TARGET'].copy()
    return targets_dataframe

#get Features
def get_features(app_dataframe):
    logger=get_logger()
    logger.info("getting Features......")
    features_dataframe=app_dataframe.drop(columns='TARGET')
    return features_dataframe

#split data Sets into training set and testing set
def split_dataSets(features_dataframe,targets_dataframe):
    logger=get_logger()
    logger.info("splitting data into training and test data sets......")
    training_features_dataframe,test_features_dataframe,training_targets_dataframe,test_targets_dataframe=train_test_split(features_dataframe,targets_dataframe,train_size=0.8,test_size=0.2,shuffle=True,stratify=targets_dataframe)
    return (training_features_dataframe,test_features_dataframe,training_targets_dataframe,test_targets_dataframe)

#convert days to years and then cut into discrete bins
def group_days_to_yearly_bins(app_dataframe):
    logger=get_logger()
    logger.info("Converting days to years and then cutting them into discrete bins......")
    app_dataframe['DAYS_BIRTH']=np.rint(np.abs(app_dataframe['DAYS_BIRTH'])/365)
    app_dataframe['DAYS_EMPLOYED']=np.rint(np.abs(app_dataframe['DAYS_EMPLOYED'])/365)
    app_dataframe['DAYS_REGISTRATION']=np.rint(np.abs(app_dataframe['DAYS_REGISTRATION'])/365)
    app_dataframe['DAYS_ID_PUBLISH']=np.rint(np.abs(app_dataframe['DAYS_ID_PUBLISH'])/365)
    app_dataframe['DAYS_LAST_PHONE_CHANGE']=np.rint(np.abs(app_dataframe['DAYS_LAST_PHONE_CHANGE'])/365)
    
    app_dataframe['DAYS_BIRTH']= pd.cut(app_dataframe['DAYS_BIRTH'], range(18, 78, 1), right=False)
    app_dataframe['DAYS_EMPLOYED']= pd.cut(app_dataframe['DAYS_EMPLOYED'], range(0, 60, 1), right=False)
    app_dataframe['DAYS_REGISTRATION']= pd.cut(app_dataframe['DAYS_REGISTRATION'], range(0, 60, 1), right=False)
    app_dataframe['DAYS_ID_PUBLISH']= pd.cut(app_dataframe['DAYS_ID_PUBLISH'], range(0, 60, 1), right=False)
    app_dataframe['DAYS_LAST_PHONE_CHANGE']= pd.cut(app_dataframe['DAYS_LAST_PHONE_CHANGE'], range(0, 60, 1), right=False)   
   
    return app_dataframe

#Change dtype 'object' and some other numerics to categories
def set_categorical_columns(app_dataframe,category_dict):
    logger=get_logger()
    logger.info("Setting Categorical Columns......")
    if 'colname_all_categories' not in category_dict:
        #Get the list which should be changed to categorical
        colname_obj=list(app_dataframe.select_dtypes(include= ['object']).columns)
        #Set Days and Real Numbers as Category
        colname_days_Numbers=['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','FLAG_EMP_PHONE',
                   'FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                   'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY',
                   'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5',
                   'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12',
                   'FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
                   'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
        colname_all_categories=colname_obj+colname_days_Numbers
        category_dict['colname_all_categories']=colname_all_categories
    else:
        colname_all_categories=category_dict['colname_all_categories']
        
    for colname in colname_all_categories:
            if colname not in category_dict:
                app_dataframe[colname]=app_dataframe[colname].astype('category')
                app_dataframe[colname]=app_dataframe[colname].copy().cat.add_categories(new_categories='NA').fillna('NA')
                category_dict[colname]=pd.api.types.CategoricalDtype(app_dataframe[colname].cat.categories)
            else:
                app_dataframe[colname]=app_dataframe[colname].astype(category_dict[colname])
                if app_dataframe[colname].isnull().sum()>0:
                    app_dataframe[colname]=app_dataframe[colname].copy().fillna('NA')
    return (app_dataframe,category_dict)

#Clipping outliers
def clip_outliers(app_dataframe,clipping_dict):
    logger=get_logger()
    logger.info("Clipping Outliers......")
    cat_columns =  app_dataframe.select_dtypes(include= ['category']).columns
    if 'SK_ID_BUREAU' in app_dataframe.columns:
        exclude_non_cat_columns=['SK_ID_CURR','SK_ID_BUREAU']
    else:
        exclude_non_cat_columns=['SK_ID_CURR']

    non_cat_columns  =  app_dataframe.columns.drop(cat_columns).drop(exclude_non_cat_columns)
    for colname in non_cat_columns:
        if colname+'_lower' not in clipping_dict: 
            q=np.nanpercentile(app_dataframe[colname],(25.0,75.0))
            clipping_dict[colname+'_lower']=q[0]-(q[1]-q[0])*1.5
            clipping_dict[colname+'_upper']=q[1]+(q[1]-q[0])*1.5
        app_dataframe[colname].clip(lower=clipping_dict[colname+'_lower'],upper=clipping_dict[colname+'_upper'],inplace=True)
    return (app_dataframe,clipping_dict)

#Filling Missing Values
def fill_missing_values(app_dataframe,missing_dict):
    logger=get_logger()
    logger.info("Fill Missing Values......")
 
    for colname in app_dataframe.columns.values:
        if pd.api.types.is_categorical_dtype(app_dataframe[colname])==True:
            pass
        else:
            if app_dataframe[colname].isnull().sum()>0: 
                if colname not in missing_dict: 
                    missing_dict[colname]=app_dataframe[colname].median()
                app_dataframe[colname].fillna(missing_dict.get(colname),inplace=True)
    s=app_dataframe.isnull().sum()
    if(s[s>0].shape[0]>0):
        logger.error('There are some nulls in data,Please fix them before going on')
        app_dataframe.isnull().sum().to_excel('./doc/application_null.xlsx')
        logger.error("Missing info has been written to ./doc/application_null.xlsx")
    return (app_dataframe,missing_dict)

#oneHotEncoding
def one_hot_encoding(app_dataframe):
    logger=get_logger()
    logger.info("Change Categorical data into One-Hot-Encoding......")    
    app_dataframe=pd.get_dummies(app_dataframe)
    return (app_dataframe)

#keep only non-zero columns
def keep_non_zero_columns(app_dataframe,nonzero_columns):
    logger=get_logger()
    logger.info("Keep only Non Zero Columns......")    
    if nonzero_columns is None:
        logger.info("get nonzerocolumn serial......")
        nonzero_columns=(app_dataframe != 0).any(axis=0)
    app_dataframe=app_dataframe.loc[:,nonzero_columns]
    return (app_dataframe,nonzero_columns)

#Scaling
def scaling(app_dataframe,scalerDict):
    logger=get_logger()
    logger.info("Feature Scaling......")
 
    if 'SK_ID_BUREAU' in app_dataframe.columns:
        exclude_non_scale_columns=['SK_ID_CURR','SK_ID_BUREAU']
    else:
        exclude_non_scale_columns=['SK_ID_CURR']

    scaling_columns  =  app_dataframe.columns.drop(exclude_non_scale_columns)
    
    for colname in scaling_columns:
        if colname not in scalerDict:
            #logger.info("Feature scaling-fitting on column "+colname)
            m=np.mean(app_dataframe[colname].values)
            s=np.sqrt(np.mean((app_dataframe[colname].values - m)**2))
            scalerDict[colname]=(m,s)
        #logger.info("Feature Scaling-transform on column "+colname)
        m,s=scalerDict[colname]
        if m==0 and s==0:
            logger.info("column {},both mean and std is zero,skipped".format(colname))
        elif s==0:
            logger.info("column {},mean is not zero but stddev is zero,skipped".format(colname))
        else:
            app_dataframe[colname]=(app_dataframe[colname].values-m)/s
    return (app_dataframe,scalerDict)

#Feature Reduction
def feature_reduction(features_dataframe,pca):
    logger=get_logger()
    logger.info("Applying PCA for feature Reduction......")    
    
    if 'SK_ID_BUREAU' in features_dataframe.columns:
        exclude_columns=['SK_ID_CURR','SK_ID_BUREAU']
    else:
        exclude_columns=['SK_ID_CURR']

    features_dataframe=features_dataframe.drop(columns=exclude_columns)
    
    if pca is None:
        #406 for 100%
        numPCA=406
        logger.info("PCA fitting,n_components={}......".format(numPCA))    	
        pca=PCA(n_components=numPCA)
        pca.fit(features_dataframe.values)
    logger.info("PCA transforming......")
    features_array=pca.transform(features_dataframe.values)
    logger.info('PCA explained variance ratio: %s'% str(pca.explained_variance_ratio_))
    
    return (features_array,pca)

#SMOTEENN Sampling to keep the data balanced
def smoteenn_sampling(features_dataframe_array,targets_dataframe_array,smote):
    logger=get_logger()
    logger.info("SMOTEENN Sampling to keep data balance......")
    if smote is None:
        logger.info("SMOTEENN fitting......")
        smote=SMOTE(k_neighbors=3)
        smote.fit(features_dataframe_array, targets_dataframe_array)
    logger.info("SMOTEENN sampling......")
    X_resampled, y_resampled = smote.sample(features_dataframe_array, targets_dataframe_array)
    return (X_resampled, y_resampled,smote)

#Scaling for numpy array
def npscaling(nparrary,npscalerDict):
    logger=get_logger()
    logger.info("Feature Scaling for Ndarray......")
    
    colnum=nparrary.shape[1]
    for i in range(colnum):
        if i not in npscalerDict:
            #logger.info("Feature scaling - fitting on column {}".format(i))
            m=np.mean(nparrary[:,i])
            s=np.sqrt(np.mean((nparrary[:,i] - m)**2))
            npscalerDict[i]=(m,s)
        #logger.info("Feature Scaling - transforming on column {}".format(i))
        m,s=npscalerDict[i]
        if m==0 and s==0:
            logger.info("column {},both mean and std is zero,skipped".format(i))
        elif s==0:
            logger.error("column {},mean is not zero but stddev is zero,which means all values are same value,this column should be deleted from dataset!".format(i))
            exit()
        else:
            nparrary[:,i]=(nparrary[:,i]-m)/s
    return (nparrary,npscalerDict)

#Set parameters
def set_parameters(category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict):
    logger=get_logger()
    transform_parameters=(category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict)
    transform_filename="./model/transform_parameters.pkl"
    with open(transform_filename, "wb") as f:
        pickle.dump(transform_parameters, f)
        logger.info("transform parameters have been written to "+transform_filename)

#get parameters
def get_parameters():
    logger=get_logger()
    transform_filename="./model/transform_parameters.pkl"
    with open(transform_filename, "rb") as f:
        category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict=pickle.load(f)
        logger.info("transform parameters have been loaded from "+transform_filename)
    return (category_dict,clipping_dict,missing_dict,scalerDict,nonzero_columns,pca,npscalerDict)

#Check Similarity between test and train data
def check_similarity(training_features_array,test_features_array):
    logger=get_logger()
    logger.info("checking Similarity between train sets and test sets......")
    
    m_train=np.mean(training_features_array)
    s_train=np.sqrt(np.mean((training_features_array - m_train)**2))

    m_test=np.mean(test_features_array)
    s_test=np.sqrt(np.mean((test_features_array - m_test)**2))

    logger.info("train mean:{},trarin stddev:{},test mean:{},test stddev:{}".format(np.mean(m_train),np.mean(s_train),np.mean(m_test),np.mean(s_test)))

    if np.mean(s_test)/np.mean(s_train)>1.01:
        logger.error('Train and Test data set have too much difference!')

#save object
def save_object(filename,ob,obname):
    logger=get_logger()
    with open(filename, "wb") as f:
        pickle.dump(ob, f)
        logger.info("object {} have been written to {}".format(filename,obname))

#save object
def get_object(filename,obname):
    logger=get_logger()
    try:
        with open(filename, "rb") as f:
            ob=pickle.load(f)
            logger.info("object {} have been loaded from {}".format(filename,obname))
        return ob
    except:
        return None
