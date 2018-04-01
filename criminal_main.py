from functions import *
from numpy import inf, nan, nanmean

from warnings import simplefilter
simplefilter("ignore")

path = 'C:/Users/nikhi/Desktop/Practice/HackerEarth-Criminal/Criminal-HackerEarth/'

def crime():
    # Train data
    train = read_csv(path + 'train.csv', na_values=-1)
    print(train.head(2))

    train.drop('PERID', axis=1, inplace=True)
    train = train.replace([inf, -inf], nan).dropna()

    # Class imbalance
    print('Target Class\n', train['Criminal'].value_counts())
    cols = train.columns.values

    # Number of unique values
    print('\nNumber of unique values in each column')
    overall = train.shape[0]
    train_criminal = train[train['Criminal'] == 1]
    criminal = train_criminal.shape[0]
    for col in cols:
        if len(train[col].unique()) > 10:
            continue
        print('\n', col)
        temp = DataFrame({'Overall': train[col].value_counts() / overall,
                          'Criminal': train_criminal[col].value_counts() / criminal})
        print(temp[['Overall', 'Criminal']])

    # Feature Engineering
    train['NC17'] = train['NRCH17_2'] / train['IRHHSIZ2']
    del train['NRCH17_2']
    del train['IRHHSIZ2']

    hlnv_cols = [col for col in train.columns.values if "HLNV" in col]
    print(hlnv_cols)
    train['HLNV'] = train[hlnv_cols].apply(lambda row: round(sum(row.values)), axis=1)
    train = train.drop(hlnv_cols, axis=1)

    hlcall_cols = [col for col in train.columns.values if "HLCALL" in col]
    print(hlcall_cols)
    train['HLCALL'] = train[hlcall_cols].apply(lambda row: round(sum(row.values)), axis=1)
    train = train.drop(hlcall_cols, axis=1)

    train['HLCNOTMO'] = train['HLCNOTMO'].apply(lambda x: 1 if x > 90 else 0)
    train['HLCLAST'] = train['HLCLAST'].apply(lambda x: 1 if x > 90 else 0)

    target = ['Criminal']
    num_cols = ['NC17', 'IRKI17_2', 'IRHH65_2', 'IRWELMOS', 'ANALWT_C']
    cat_cols = [col for col in train.columns.values if col not in (num_cols + target)]
    print(len(train.columns.values), len(num_cols), len(cat_cols))

    freqs = {}
    for col in cat_cols:
        # Frequency columns
        print(f">> Calculating frequency for: {col}")
        # Get counts, sums and frequency of is_attributed
        df = DataFrame({
            'sums': train.groupby(col)['Criminal'].sum(),
            'counts': train.groupby(col)['Criminal'].count()
        })
        df.loc[:, 'freq'] = df.sums / df.counts

        # If we have less than 3 observations, e.g. for an IP, then assume freq of 0
        df.loc[df.counts <= 3, 'freq'] = 0

        # Saving in dictionary
        freqs[col] = df

        # Add to X_total
        train[col + '_freq'] = train[col].map(df['freq'])

        # One Hot Encoding
        print(f">> One-hot encoding for: {col}")
        # train[col] = train[col].astype('category',copy=False)
        temp = get_dummies(train[col])
        temp.columns = [col + '_' + str(i).split('.')[0] for i in temp.columns]
        train = train.join(temp)
        train = train.drop(col, axis=1)

    y = train['Criminal']
    X = train[[col for col in train.columns.values if col not in ['PERID', 'Criminal']]]

    # Splitting Train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.001, random_state=45)
    print(X_train.shape, X_test.shape)

    # Feature Selection
    print('Working on Feature Selection...')
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, y_train)
    feat_imp = Series(rf.feature_importances_, index=X_train.columns.values).sort_values(ascending=False)
    # feat_imp[:30].plot(kind='bar', title='Feature Importance with Random Forest', figsize=(15, 8))
    # plt.ylabel('Feature Importance values')
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig('FeatImportance.png')
    # plt.show()
    imp_feats = list(feat_imp[:280].index)
    print(imp_feats)

    X = X[imp_feats]
    X_train = X_train[imp_feats]
    X_test = X_test[imp_feats]

    # Pipeline
    pipe, feat_params, model_params, all_params = pipeline_params()
    #
    # grid = GridSearchCV(estimator=pipe, param_grid=all_params, scoring=make_scorer(matthews_corrcoef), verbose=20,
    #                     n_jobs=-1)
    # grid.fit(X_train, y_train)
    #
    # cv_result_pipe = DataFrame(grid.cv_results_).sort_values('rank_test_score').to_csv(path + 'cv_result_pipe.csv',
    #                                                                                    index=False)
    # print(grid.best_score_)
    # print(grid.best_estimator_)
    #
    # imp_feats = X_train.columns.values[grid.best_params_['feat_sel'].get_support(indices=True)]
    # print(imp_feats)
    #
    # # Results on X_test
    # best_model = grid.best_estimator_.fit(X_train[imp_feats], y_train)
    # y_pred = best_model.predict(X_test[imp_feats])
    #
    # print('MCC:', matthews_corrcoef(y_test, y_pred))
    # print('Acc:', accuracy_score(y_test, y_pred))
    # print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))

    # Ensemble
    models_combo = feat_model_combo()
    pred_all, trained_models = predict_mixed(models_combo, X_train, y_train, X, y,  # Check for _test
                                             False)  # Probability True/False

    pred_all['final'] = pred_all.loc[:, 'pred1':'pred32'].apply(lambda row: round(nanmean(row.values)), axis=1)

    print('MCC:', matthews_corrcoef(pred_all['y_true'], pred_all['final']))
    print('Acc:', accuracy_score(pred_all['y_true'], pred_all['final']))
    print('Confusion Matrix\n', confusion_matrix(pred_all['y_true'], pred_all['final']))

    # Meta-learner
    try:
        del pred_all['final']
    except:
        pass
    print(pred_all.head(2))

    y_m = pred_all['y_true']
    X_m = pred_all.drop('y_true', axis=1)
    print(X_m.head(2))

    # Splitting Train test
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, stratify=y_m, test_size=0.001, random_state=20)
    print('\n', X_train_m.shape, X_test_m.shape)

    meta_pipe = Pipeline([('model', RandomForestClassifier())])
    meta_grid = GridSearchCV(estimator=meta_pipe, param_grid=model_params, scoring=make_scorer(matthews_corrcoef),
                             verbose=20, n_jobs=-1)
    meta_grid.fit(X_train_m, y_train_m)
    meta_model = meta_grid.best_estimator_.fit(X_train_m, y_train_m)
    meta_pred = meta_model.predict(X_test_m)

    print('MCC:', matthews_corrcoef(y_test_m, meta_pred))
    print('Acc:', accuracy_score(y_test_m, meta_pred))
    print('Confusion Matrix\n', confusion_matrix(y_test_m, meta_pred))

    # Final results on Test data
    test = read_csv(path + 'test.csv', na_values=-1)
    perid = test['PERID']
    test.drop('PERID', axis=1, inplace=True)

    test = test.replace([inf, -inf, nan], 0).fillna(0)

    test['NC17'] = test['NRCH17_2'] / test['IRHHSIZ2']
    del test['NRCH17_2']
    del test['IRHHSIZ2']

    hlnv_cols = [col for col in test.columns.values if "HLNV" in col]
    print(hlnv_cols)
    test['HLNV'] = test[hlnv_cols].apply(lambda row: round(sum(row.values)), axis=1)
    test = test.drop(hlnv_cols, axis=1)

    hlcall_cols = [col for col in test.columns.values if "HLCALL" in col]
    print(hlcall_cols)
    test['HLCALL'] = test[hlcall_cols].apply(lambda row: round(sum(row.values)), axis=1)
    test = test.drop(hlcall_cols, axis=1)

    test['HLCNOTMO'] = test['HLCNOTMO'].apply(lambda x: 1 if x > 90 else 0)
    test['HLCLAST'] = test['HLCLAST'].apply(lambda x: 1 if x > 90 else 0)

    num_cols = ['NC17', 'IRKI17_2', 'IRHH65_2', 'IRWELMOS', 'ANALWT_C']
    cat_cols = [col for col in test.columns.values if col not in num_cols]

    for col in cat_cols:
        # Frequency columns
        print(f">> Calculating frequency for: {col}")
        test[col + '_freq'] = test[col].map(freqs[col]['freq'])

        # One Hot Encoding
        print(f">> One-hot encoding for: {col}")
        test[col] = test[col].astype('category', copy=False)
        temp = get_dummies(test[col])
        temp.columns = [col + '_' + str(i).split('.')[0] for i in temp.columns]
        test = test.join(temp)
        test = test.drop(col, axis=1)
    print(test.head(2))

    test = test.fillna(0)
    meta_test = DataFrame()

    for i, dic in trained_models.items():
        meta_test['pred' + str(i)] = dic['model'].predict(test[dic['feats']])
    print(meta_test.head(2))

    meta_test['Criminal'] = meta_model.predict(meta_test)
    meta_test['PERID'] = perid
    print(meta_test.head(2))

    # Submission File
    meta_test[['PERID', 'Criminal']].to_csv(path + 'final_14.csv', index=False)
    nc = sum(meta_test['Criminal'])
    print('Criminal  number: {}, percent: {}'.format(nc, nc / meta_test.shape[0]))


if __name__ == '__main__':
    crime()
