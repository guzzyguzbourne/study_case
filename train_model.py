"""
Created on Tue Apr  5 01:56:44 2022

@author: gznerdem
"""
import os
import pandas as pd
import numpy as np
import joblib

#PATH1 ="/Users/gzner/KLARNA_CASE_STUDY/artifacts"
DATASET_PATH = "../data/dataset.csv"
TARGET = "default"
REQUIRED_COLUMNS = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m', 'age',
       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'has_paid',
       'max_paid_inv_0_12m', 'max_paid_inv_0_24m',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_ok_12_24m', 'num_arch_ok_0_12m', 'num_arch_rem_0_12m',
       'num_arch_written_off_0_12m', 'num_unpaid_bills',
       'status_last_archived_0_24m', 'status_2nd_last_archived_0_24m',
       'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
       'status_max_archived_0_12_months', 'status_max_archived_0_24_months',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m', 'sum_paid_inv_0_12m', 'time_hours',
       'merchant_group', 'merchant_category']


def load_data(path):
    """Load data """
    print('Loading the data')
    df = pd.read_csv(path, delimiter=';')
    print('Successfully uploaded the data')
    return df


def prepare_data(df):
    """function to prepare data in order to pass as an input to our model"""

    print('Preparating the data')
    # 1. Check for default column in test data
    if TARGET in df.columns:
        df = df[df["default"].isna()].drop(["default"], axis=1).reset_index(drop=True)
    
   

    # 2. Check for all required columns
    for col_name in REQUIRED_COLUMNS:
        if col_name not in df.columns:
            print('Writing the data to csv file where required column values are missing')
            df[df[REQUIRED_COLUMNS].isnull().any(axis=1)].to_csv(
                '../artifacts/required_columns_values_missing.csv')
            raise Exception('Required column  is missing:{}', format(col_name))

            
   
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    #3. Create 2 new columns and prepare others in test data
    
    df["num_arch_ok_0_24m"] = df.num_arch_ok_0_12m + df.num_arch_ok_12_24m
    df["sum_capital_paid_account_0_24m"] = df.sum_capital_paid_account_0_12m + df.sum_capital_paid_account_12_24m
    
    df = df.assign(merchant_category_upt = np.where(df['merchant_category'].isin(['Diversified electronics', 'Prints & Photos',
       'Children Clothes & Nurturing products', 'Pet supplies',
       'Electronic equipment & Related accessories', 'Hobby articles',
       'Jewelry & Watches', 'Prescription optics', 'Body & Hair Care',
       'Automotive Parts & Accessories',
       'Diversified Health & Beauty products',
       'Diversified Home & Garden products', 'Decoration & Art',
       'Video Games & Related accessories', 'Cosmetics', 'Dating services',
       'Children toys', 'Diversified erotic material',
       'Tools & Home improvement', 'Furniture', 'Pharmaceutical products',
       'Personal care & Body improvement', 'Fragrances',
       'Adult Shoes & Clothing', 'Digital services', 'Food & Beverage',
       'Travel services', 'Costumes & Party supplies', 'Music & Movies',
       'Wheels & Tires', 'Collectibles', 'Kitchenware', 'Underwear',
       'Household electronics (whitegoods/appliances)',
       'Erotic Clothing & Accessories', 'Non',
       'Musical Instruments & Equipment', 'Tobacco', 'Safety products',
       'Diversified Jewelry & Accessories', 'Car electronics', 'Sex toys',
       'Plants & Flowers', 'Bags & Wallets',
       'Office machines & Related accessories (excl. computers)',
       'Cleaning & Sanitary', 'Event tickets', 'Wine, Beer & Liquor',
       'Education']),'Other', df['merchant_category']))
    
    
    df = df.assign(merchant_group_upt = np.where(df['merchant_group'].isin(['Home & Garden', 'Electronics', 'Intangible products',
       'Jewelry & Accessories', 'Automotive Products', 'Erotic Materials',
       'Food & Beverage']),'Other', df['merchant_group']))
    
    df['has_paid'] = df['has_paid'].astype('object')
    
    # 4. Select columns of training data    
    model_columns = joblib.load("../artifacts/Prob_Default_Klarna_InputNames3.pkl")
    uuid = df["uuid"]
    df = df[model_columns]
    for col in df.filter(regex='status').columns:
        df[col] = df[col].astype('object')

    # 5. Impute data for numerical columns :
    
    df_numerical = df[df.select_dtypes(exclude = "object").columns]
    
    # load iterative imputer from disk
    imputer = joblib.load("../artifacts/iterative_imputer.pkl")
    df_num_imp = imputer.transform(df_numerical)
    df_num_imp = pd.DataFrame(df_num_imp)
    df_num_imp.columns = df_numerical.columns


    # 5.Categorical column data preparation
    
    df_categorical = df[df.select_dtypes(include = "object").columns]
    
    # load one hot encoder from disk
    
    simp_imp = joblib.load("../artifacts/simple_imputer.pkl")
    one_hot_encoder = joblib.load("../artifacts/one_hot_encoder.pkl")
    df_one_hot = one_hot_encoder.transform(df_categorical)
    df_one_hot = simp_imp.transform(pd.DataFrame(df_one_hot.toarray()))
    df_one_hot = pd.DataFrame(df_one_hot)
    df_one_hot.reset_index()
    df_final = pd.concat([uuid, df_num_imp,
                          df_one_hot], axis=1)

    print('Data preparation finished successfully')
    return df_final


def predict_default(df_final):
    """This function is used to predict the default and write the result to ../artifacts/prediction.csv file"""

    # load model from disk
    sgd_model = joblib.load("../artifacts/default_predictor_sgd.pkl")

    if 'uuid' in df_final.columns:

        df_final['default_prediction'] = sgd_model.predict(df_final.drop(['uuid'], axis=1))
        prediction = df_final[['uuid', 'default_prediction']]

    else:
        df_final['default_prediction'] = sgd_model.predict(df_final)
        prediction = df_final['default_prediction']

    prediction.to_csv('../artifacts/prediction.csv')

    print('Successfully predicted the data, please check: ../artifacts/prediction.csv')
    return prediction
