"""
Load data into configuration dictionary for use on later end models.
"""



import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
import pickle
import random
import tqdm



savepath = 'D:\yzy\paper3-CL\Codes\Preprocessing\Results_round2/processed_data_y_df_x_train_test_norm.pkl'
with open(savepath, 'rb') as file:
    train_df, test_df, all_x_trainset, X_test, \
    mask_x_train_selectedFeatures, mask_x_test_selectedFeatures, feature_list, norm_min, norm_max = pickle.load(
        file)
'''
['patient_ID', 'time_to_start', 'death_label_t', 'death_label_e',
 'class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes',
 'class_infect', 'class_vascular', 'class_kidney_failure', 'class_liver',
 'class_k', 'class_digestive', 'class_bone', 'class_breath',
 'class_nerves', 'class_thrombocytopenia', 'class_malnutrition',
 'class_renal_amyloidosis', 'train_test_flag']
 '''
# classificationOutcomes  = ['class_hypertension', 'class_anemia', 'class_heart','class_diabetes','class_infect',]
classificationOutcomes = ['class_hypertension', 'class_anemia', 'class_heart','class_diabetes','class_infect', 'class_bone']
InterestedOutcomes = classificationOutcomes + ['death_label_t', 'death_label_e']

all_label_trainset = train_df[InterestedOutcomes].values
y_test = test_df[InterestedOutcomes].values
print("np.isnan(all_label_trainset)", np.sum(np.isnan(all_label_trainset)))
print("np.isnan(y_test)", np.sum(np.isnan(y_test)))
print("np.isnan(train_df)", train_df.isnull().sum())
print("np.isnan(test_df)", test_df.isnull().sum())
print("np.isnan(all_x_trainset)", np.sum(np.isnan(all_x_trainset)))
print("np.isnan(X_test)", np.sum(np.isnan(X_test)))


######TODO：划分训练集和验证集

flag = False
for ii in range(1):
    seed = random.randrange(20, 10000, 32)

    if flag:
        break
    X_train, X_val, y_train_df, y_val_df, mask_train, mask_val = train_test_split(
        all_x_trainset, train_df, mask_x_train_selectedFeatures, train_size=0.7, shuffle=True, random_state=9940)

    # X_train, X_val, y_train_df, y_val_df, mask_train, mask_val = train_test_split(
    #     all_x_trainset, train_df, mask_x_train_selectedFeatures, train_size=0.7, shuffle=True, random_state=9940,
    #     stratify=all_label_trainset[:, -1])

    p1 = ks_2samp(y_train_df['class_hypertension'].values, y_val_df['class_hypertension'].values)[1]
    p2 = ks_2samp(y_train_df['class_anemia'].values, y_val_df['class_anemia'].values)[1]
    p3 = ks_2samp(y_train_df['class_heart'].values, y_val_df['class_heart'].values)[1]
    p4 = ks_2samp(y_train_df['class_diabetes'].values, y_val_df['class_diabetes'].values)[1]
    p5 = ks_2samp(y_train_df['class_infect'].values, y_val_df['class_infect'].values)[1]
    p6 = ks_2samp(y_train_df['class_bone'].values, y_val_df['class_bone'].values)[1]

    print("p1:", p1)
    print("p2:", p2)
    print("p3:", p3)
    print("p4:", p4)
    print("p5:", p5)
    print("p6:", p6)
    print("seed", seed)
    if (p1 > 0.05) and (p2 > 0.05) and (p3 > 0.05) and (p4 > 0.05) and (p5 > 0.05) and (p6 > 0.05) :
        print("Success-----------------")
        print("p1:", p1)
        print("p2:", p2)
        print("p3:", p3)
        print("p4:", p4)
        print("p5:", p5)
        print("p6:", p6)
        print("seed",seed)
        flag = True
