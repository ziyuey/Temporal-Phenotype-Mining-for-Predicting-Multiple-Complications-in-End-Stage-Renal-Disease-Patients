
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
import xgboost as xgb
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

# 引入我们的pickle数据

savepath = '/data/data_yangzy/BP_paper/Codes/Preprocessing/Results_round2/processed_data_y_df_x_train_test_norm.pkl'
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
for item in [
 'class_hypertension', 'class_anemia', 'class_heart', 'class_diabetes',
 'class_infect', 'class_vascular', 'class_kidney_failure', 'class_liver',
 'class_k', 'class_digestive', 'class_bone', 'class_breath',
 'class_nerves', 'class_thrombocytopenia', 'class_malnutrition',
 'class_renal_amyloidosis']:

    print("InterestedOutcome is:",item)
    InterestedOutcomes = [item]

    all_label_trainset = train_df[InterestedOutcomes].values
    y_test = test_df[InterestedOutcomes].values
    print("np.isnan(all_label_trainset)", np.sum(np.isnan(all_label_trainset)))
    print("np.isnan(y_test)", np.sum(np.isnan(y_test)))
    seed: int = 1024
    X_train, X_val, y_train_df, y_val_df, mask_train, mask_val = train_test_split(
        all_x_trainset, train_df, mask_x_train_selectedFeatures, train_size=0.7, shuffle=True, random_state=seed,
        stratify=all_label_trainset[:, 0])
    y_train = y_train_df[InterestedOutcomes].values
    y_val = y_val_df[InterestedOutcomes].values
    # y_train = convertToTwoClass(y_train_df[InterestedOutcomes].values)
    # print("distribution of y_train:", np.mean(y_train, axis=0))
    # y_val = convertToTwoClass(y_val_df[InterestedOutcomes].values)
    # print("distribution of y_val:", np.mean(y_val, axis=0))
    # print("np.isnan(y_train)",np.sum(np.isnan(y_train)))
    # print("np.isnan(y_val)", np.sum(np.isnan(y_val)))

    ######################################featureset
    feature_list = ['blood_flow_rate', 'dialysate_flow_rate', 'time_setting',
                                 'concentration_of_ca', 'concentration_of_na', 'concentration_of_k', 'n_sessions_per_week',
                                 'sex', 'height',
                                 'transplant', 'pd',
                                 'white_blood_cell', 'hemoglobin', 'platelet', 'k',
                                 'cl', 'na', 'p', 'ca', 'TP', 'ALB', 'GLB', 'AG', 'ALT', 'AST', 'ALP', 'TBIL', 'DBIL',
                                 'IBIL', 'GGT',
                                 'creatinine',
                                 'blood_urea_uitrogen', 'uric_acid', 'glucose', 'serum_iron', 'TIBC', 'transferritin',
                                 'ferritin',
                                 'Triglyceride',
                                 'Total_cholesterol', 'HDL', 'LDL', 'VLDL', 'bloodph', 'pCO2', 'pO2', 'bicarbonate', 'BE',
                                 'SatO2',
                                 'dry_weight',
                                 'preweight', 'postweight', 'target_amoounts_UF', 'IWG_new', 'HD_type_  HDF',
                                 'HD_type_CRRT',
                                 'HD_type_HD', 'HD_type_HD高通量', 'HD_type_HF', 'access_route_临时插管', 'access_route_直接穿刺',
                                 'access_route_移植血管内瘘',
                                 'access_route_自体动静脉内瘘', 'access_route_长期插管', 'use_anticoagulant_4%枸橼酸',
                                 'use_anticoagulant_万脉舒',
                                 'use_anticoagulant_克赛', 'use_anticoagulant_吉派啉', 'use_anticoagulant_希弗全',
                                 'use_anticoagulant_无肝素',
                                 'use_anticoagulant_普洛静', 'use_anticoagulant_普通肝素', 'use_anticoagulant_枸橼酸',
                                 'use_anticoagulant_法安明',
                                 'use_anticoagulant_肝素＋鱼精蛋白', 'use_anticoagulant_速比凝＋肝素', 'use_anticoagulant_速避凝',
                                 'use_anticoagulant_齐征', 'age',
                                 'startHD_age', 'HD_history_length', 'BP_lasting_time', 'IDH1_label', 'IDH2_label',
                                 'IDH_start_time',
                                 'IDH_lasting_time', 'SBP_min', 'SBP_max', 'SBP_delta', 'SBP_std', 'SBP_mean', 'DBP_min',
                                 'DBP_max',
                                 'DBP_delta',
                                 'DBP_std', 'DBP_mean', 'MAP_min', 'MAP_max', 'MAP_delta', 'MAP_std', 'MAP_mean']
    selected_feature_list = ['blood_flow_rate', 'dialysate_flow_rate', 'time_setting',
                                 'concentration_of_ca', 'concentration_of_na', 'concentration_of_k', 'n_sessions_per_week',
                                 'sex', 'height',
                                 'transplant', 'pd',
                                 'white_blood_cell', 'hemoglobin', 'platelet', 'k',
                                 'cl', 'na', 'p', 'ca', 'TP', 'ALB', 'GLB', 'AG', 'ALT', 'AST', 'ALP', 'TBIL', 'DBIL',
                                 'IBIL', 'GGT',
                                 'creatinine',
                                 'blood_urea_uitrogen', 'uric_acid', 'glucose', 'serum_iron', 'TIBC', 'transferritin',
                                 'ferritin',
                                 'Triglyceride',
                                 'Total_cholesterol', 'HDL', 'LDL', 'VLDL', 'bloodph', 'pCO2', 'pO2', 'bicarbonate', 'BE',
                                 'SatO2',
                                 'dry_weight',
                                 'preweight', 'postweight', 'target_amoounts_UF', 'IWG_new', 'HD_type_  HDF',
                                 'HD_type_CRRT',
                                 'HD_type_HD', 'HD_type_HD高通量', 'HD_type_HF', 'access_route_临时插管', 'access_route_直接穿刺',
                                 'access_route_移植血管内瘘',
                                 'access_route_自体动静脉内瘘', 'access_route_长期插管', 'use_anticoagulant_4%枸橼酸',
                                 'use_anticoagulant_万脉舒',
                                 'use_anticoagulant_克赛', 'use_anticoagulant_吉派啉', 'use_anticoagulant_希弗全',
                                 'use_anticoagulant_无肝素',
                                 'use_anticoagulant_普洛静', 'use_anticoagulant_普通肝素', 'use_anticoagulant_枸橼酸',
                                 'use_anticoagulant_法安明',
                                 'use_anticoagulant_肝素＋鱼精蛋白', 'use_anticoagulant_速比凝＋肝素', 'use_anticoagulant_速避凝',
                                 'use_anticoagulant_齐征', 'age',
                                 'startHD_age', 'HD_history_length', 'BP_lasting_time', 'IDH1_label', 'IDH2_label',
                                 'IDH_start_time',
                                 'IDH_lasting_time', 'SBP_min', 'SBP_max', 'SBP_delta', 'SBP_std', 'SBP_mean', 'DBP_min',
                                 'DBP_max',
                                 'DBP_delta',
                                 'DBP_std', 'DBP_mean', 'MAP_min', 'MAP_max', 'MAP_delta', 'MAP_std', 'MAP_mean']
    # positions = [feature_list.index(x) if x in feature_list else None for x in selected_feature_list]
    # print('positions',positions)
    # x_train_selectedFeatures = X_train[:, :, positions]
    # x_val_selectedFeatures = X_val[:, :, positions]
    # x_test_selectedFeatures = X_test[:, :, positions]



    # all_trainset = pd.DataFrame(np.mean(X_train,axis=1),columns=feature_list)
    # all_testset = pd.DataFrame(np.mean(X_test,axis=1),columns=feature_list)
    # all_valset = pd.DataFrame(np.mean(X_val,axis=1),columns=feature_list)

    all_trainset = pd.DataFrame(np.reshape(X_train,[X_train.shape[0],-1]))
    all_testset = pd.DataFrame(np.reshape(X_test,[X_test.shape[0],-1]))
    all_valset = pd.DataFrame(np.reshape(X_val,[X_val.shape[0],-1]))
    # all_trainset['IDH_label'] = y_train
    # all_testset['IDH_label'] = y_test
    # all_valset['IDH_label'] = y_val


    # 1 训练xgboost模型
    # Set the hyperparameters for the XGBoost model
    # dtrain = xgb.DMatrix(all_trainset[selected_feature_list].values, label=all_trainset['IDH_label'].values)
    # dtest = xgb.DMatrix(all_testset[selected_feature_list].values, label=all_testset['IDH_label'].values)
    # dval = xgb.DMatrix(all_valset[selected_feature_list].values, label=all_valset['IDH_label'].values)
    dtrain = xgb.DMatrix(all_trainset.values, label=y_train)
    dtest = xgb.DMatrix(all_testset.values, label=y_test)
    dval = xgb.DMatrix(all_valset.values, label=y_val)
    params = {
        'objective': 'multi:softmax',  # Specify the objective as multi-class classification
        'num_class': 2,  # Specify the number of classes
        'tree_method': 'auto',  # Use the fastest method for building trees
        #####tuning parameters
        'max_depth':25,
        'min_child_weight': 5,
        'eta':0.2,
        'subsample' : 0.5
    }


    # Train the XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=10)
    # Predict the class labels on the test data
    print("Validation on the test set..")
    y_pred = model.predict(dtest)
    # 计算预测结果的准确率
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    roc = metrics.roc_auc_score(y_test, y_pred)
    print("AUC Score:", roc)

    # 计算预测结果的 F1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score:", f1)

    print("Validation on the val set..")
    y_pred = model.predict(dval)
    # 计算预测结果的准确率
    acc = accuracy_score(y_val, y_pred)
    print("Accuracy:", acc)

    roc = metrics.roc_auc_score(y_val, y_pred)
    print("AUC Score:", roc)

    # 计算预测结果的 F1 score
    f1 = f1_score(y_val, y_pred, average='macro')
    print("F1 Score:", f1)

    # Print feature importance
    def get_xgb_imp(xgb, feat_names):
        imp_vals = xgb.get_fscore()
        imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
        return sorted(imp_dict.items(), key = lambda kv:(kv[1], kv[0]))

    print(get_xgb_imp(model,selected_feature_list))
