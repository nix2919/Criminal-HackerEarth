from pandas import read_csv, DataFrame, get_dummies, Series
from numpy import nanmean
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# from boruta import BorutaPy
import xgboost as xgb
from random import sample
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold
from sklearn.feature_selection import SelectFpr, chi2, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy import arange, delete


def pipeline_params():
    rf = RandomForestClassifier(n_jobs=-1)
    et = ExtraTreesClassifier(n_jobs=-1)
    ada = AdaBoostClassifier(base_estimator=et)
    xg = xgb.XGBClassifier(n_jobs=-1)
    gb = GradientBoostingClassifier()
    lr = LogisticRegression()
    rfe = RFE(rf, step=0.2)
    select = SelectFromModel(rf)
    kbest = SelectKBest(chi2)
    pipe = Pipeline([('feat_sel', rfe), ('model', rf)])
    feat_sel_params = [
        {
            'feat_sel': [kbest],
            'feat_sel__k': [30]},
        {
            'feat_sel': [rfe],
            'feat_sel__estimator': [rf],  # rf, et, gb, xg
            'feat_sel__n_features_to_select': [30]},
        {
            'feat_sel': [select],
            'feat_sel__estimator': [rf]}  # rf, et, gb, xg
    ]
    model_params = [
        {
            'model': [lr]},
        {
            'model': [gb],
            'model__n_estimators': [1000],  # 500, 1000, 2000, 4000
            'model__learning_rate': [0.1]},  # 0.01, 0.04, 0.1, 0.5, 1
        # {
        #     'model': [ada],
        #     'model__n_estimators': [1000],  # 500, 1000, 2000, 4000
        #     'model__learning_rate': [0.1, 0.2, 0.5],  # 0.01, 0.04, 0.1, 0.5, 1
        #     'model__random_state': [9]},
        {
            'model': [xg],
            'model__objective': ['binary:logistic'],
            'model__learning_rate': [0.1],  # Learning rate alpha
            'model__gamma': [0.5],   # minimum eval_score deduction at each split
            'model__min_child_weight': [4],  # minimum number of data points in a split
            'model__subsample': [0.9],  # sample size row-wise during bootstrap
            'model__colsample_bytree': [0.7],  # column-wise sample size
            'model__n_estimators': [1000]},   # number of trees to build
        {
            'model': [rf],
            'model__n_estimators': [1000],  # 500, 1000, 2000, 4000
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['sqrt'],  # , 'log2'
            'model__min_samples_leaf': [4],  # 3, 5, 7, 9
            'model__max_depth': [12]},  # 8, 10, 14
        {
            'model': [et],
            'model__n_estimators': [1000],  # 500, 1000, 2000, 4000
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['sqrt'],  # , 'log2'
            'model__min_samples_leaf': [4],  # 3, 5, 7
            'model__max_depth': [12]}  # 8, 10, 14
    ]
    params = []
    for feat_sel in feat_sel_params:
        for model in model_params:
            # Merge dictionaries and append to list
            params.append({**feat_sel, **model})

    return pipe, feat_sel_params, model_params, params


def feat_model_combo():
    rfe_rf = RFE(RandomForestClassifier(n_estimators=100), step=0.2, n_features_to_select=30)
    select_rf = SelectFromModel(RandomForestClassifier(n_estimators=100))
    rfe_et = RFE(ExtraTreesClassifier(n_estimators=100), step=0.2, n_features_to_select=30)
    select_et = SelectFromModel(ExtraTreesClassifier(n_estimators=100))

    rf_gini = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=11, max_features='sqrt',
                                     min_samples_leaf=4, n_jobs=-1)
    rf_ent = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=11, max_features='sqrt',
                                    min_samples_leaf=4, n_jobs=-1)

    et_gini = ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=11, max_features='sqrt',
                                   min_samples_leaf=4, n_jobs=-1)
    et_ent = ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=11, max_features='sqrt',
                                  min_samples_leaf=4, n_jobs=-1)

    xg = xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.5,
                                    learning_rate=0.2, min_child_weight=5, subsample=0.9, n_jobs=-1)

    lr = LogisticRegression()

    combo = [
        # RFE (rf + et + xgb)

        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                              min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': rfe_rf,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': rfe_et,
        #  'model': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.39,
                                    learning_rate=0.07, min_child_weight=3, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.69,
                                    learning_rate=0.07, min_child_weight=5, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.93,
                                    learning_rate=0.07, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.69,
                                    learning_rate=0.75, min_child_weight=2, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.93,
                                    learning_rate=0.75, min_child_weight=7, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.69,
                                    learning_rate=0.075, min_child_weight=9, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.39,
                                    learning_rate=0.705, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.96,
                                    learning_rate=0.075, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.93,
                                    learning_rate=0.42, min_child_weight=3, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.69,
                                    learning_rate=0.23, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.39,
                                    learning_rate=0.52, min_child_weight=5, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.96,
                                    learning_rate=0.32, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.93,
                                    learning_rate=0.73, min_child_weight=5, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.69,
                                    learning_rate=0.38, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.93,
                                    learning_rate=0.93, min_child_weight=3, subsample=0.9, n_jobs=-1)},
        {'feat_sel': rfe_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.69,
                                    learning_rate=0.33, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        # SelectFrom Model  (rf + et + xgb)
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': select_rf,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=4, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=6, n_jobs=-1)},
        # {'feat_sel': select_et,
        #  'model': ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=14, max_features='sqrt',
        #                                  min_samples_leaf=8, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.79,
                                    learning_rate=0.05, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.49,
                                    learning_rate=0.05, min_child_weight=2, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.79,
                                    learning_rate=0.05, min_child_weight=8, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.94,
                                    learning_rate=0.05, min_child_weight=8, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.97,
                                    learning_rate=0.05, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.49,
                                    learning_rate=0.05, min_child_weight=2, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.79,
                                    learning_rate=0.05, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.94,
                                    learning_rate=0.05, min_child_weight=7, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.97,
                                    learning_rate=0.15, min_child_weight=9, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.49,
                                    learning_rate=0.15, min_child_weight=6, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.97,
                                    learning_rate=0.15, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_rf,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.49,
                                    learning_rate=0.15, min_child_weight=3, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.97,
                                    learning_rate=0.25, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.5, gamma=0.49,
                                    learning_rate=0.25, min_child_weight=7, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.49,
                                    learning_rate=0.25, min_child_weight=4, subsample=0.9, n_jobs=-1)},
        {'feat_sel': select_et,
         'model': xgb.XGBClassifier(n_estimators=1000, objective='binary:logistic', colsample_bytree=0.7, gamma=0.97,
                                    learning_rate=0.25, min_child_weight=5, subsample=0.9, n_jobs=-1)}
    ]
    return combo


def predict_mixed(combo, X_train, y_train, X_test, y_test, prob=False):
    preds = DataFrame()
    preds['y_true'] = y_test  # Check for y and y_test
    trained_models = {}
    features = X_train.columns.values

    for i, one in zip(range(1, len(combo) + 1), combo):
        print('\n')
        trained_models[i] = {}

        feat_sel = one['feat_sel']
        feat_sel.fit(X_train, y_train)

        imp_feat = [features[i] for i in feat_sel.get_support(indices=True)]
        if len(imp_feat)>30:
            imp_feat = imp_feat[:30]
        print(len(imp_feat))
        trained_models[i]['feats'] = imp_feat

        model = one['model']
        model.fit(X_train[imp_feat], y_train)
        trained_models[i]['model'] = model

        # print(feat_sel, model)

        if prob:
            pred = model.predict_proba(X_test[imp_feat])  # Check for X and X_test
            # preds[key+'_0'] = [i[0] for i in pred]
            preds['pred' + str(i)] = [j[1] for j in pred]
        else:
            preds['pred' + str(i)] = model.predict(X_test[imp_feat])  # Check for X and X_test
            print('MCC:', matthews_corrcoef(preds['y_true'], preds['pred' + str(i)]))
            print('Acc:', accuracy_score(preds['y_true'], preds['pred' + str(i)]))
            print('Confusion Matrix\n', confusion_matrix(preds['y_true'], preds['pred' + str(i)]))

    if prob:
        preds.to_excel('mixed_ensemble_proba.xlsx', index=False)
    else:
        preds.to_excel('mixed_ensemble_preds.xlsx', index=False)

    return preds, trained_models


def get_models():
    rf = RandomForestClassifier(n_estimators=100)
    et = ExtraTreesClassifier(n_estimators=100)
    xg = xgb.XGBClassifier()
    ada = AdaBoostClassifier(base_estimator=et)
    gb = GradientBoostingClassifier(n_estimators=100)
    lr = LogisticRegression()

    models = {
        'rf': rf,
        'lr': lr,
        'xg': xg,
        'et': et,
        'gb': gb,
        'ada': ada
    }

    return models


def predict_uni(models, X_train, y_train, X_test, y_test, prob=False):
    preds = DataFrame()
    preds['y_true'] = y_test  # Check for y and y_test

    trained_models = {}

    for key, model in models.items():
        print('\n', key)

        model.fit(X_train, y_train)
        trained_models[key] = model

        if prob:
            pred = model.predict_proba(X_test)  # Check for X and X_test
            # preds[key+'_0'] = [i[0] for i in pred]
            preds[key + '_1'] = [i[1] for i in pred]
        else:
            preds[key] = model.predict(X_test)  # Check for X and X_test
            print('MCC:', matthews_corrcoef(preds['y_true'], preds[key]))
            print('Acc:', accuracy_score(preds['y_true'], preds[key]))
            print('Confusion Matrix\n', confusion_matrix(preds['y_true'], preds[key]))

    if prob:
        preds.to_excel('uni_ensemble_proba.xlsx', index=False)
    else:
        preds.to_excel('uni_ensemble_preds.xlsx', index=False)

    return preds, trained_models


def calculate_vif(X, thresh=100):
    cols = X.columns
    variables = arange(X.shape[1])
    dropped = True
    while dropped:
        dropped = False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = delete(variables, maxloc)
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]
