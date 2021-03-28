import argparse
import pickle
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random

def read_dataset(filename, feature_cols, cat_feature_cols):
    df = pd.read_csv(filename)
    print(df.head())
    print(df.dtypes)

    # for f in cat_feature_cols:
    #     df[f] = df[f].astype('category')
        # df[f] = df[f].cat.codes

    replace_map = {'Gender': {'Male': 0, 'Female': 1},
                   'Company Type': {'Service': 0, 'Product': 1},
                   'WFH Setup Available': {'Yes': 0, 'No': 1}}

    df_repl = df.copy()
    df_repl.replace(replace_map, inplace=True)
    print(df_repl.head)

    X = df_repl.loc[:, feature_cols]
    print(X.shape)
    y = df_repl['Burn Rate']
    print(y.shape)
    return X,y

def dtrees(X_fit, y_fit, X_eval, y_eval, features, dt_file):
    #DTrees
    dtree = tree.DecisionTreeRegressor().fit(X_fit, y_fit)
    accuracy = dtree.score(X_eval, y_eval)
    dt_file.write(f'Single Dtree: {accuracy}\n')

    for feature, imp in zip(features, dtree.feature_importances_):
        dt_file.write("\tFeature %s: %s\n" % (feature, imp))

    pickle.dump(dtree, open('dtree.p', 'wb'))

    #Random Forest Trees
    rf_dtree = RandomForestRegressor(n_estimators=8).fit(X_fit,y_fit)
    accuracy = rf_dtree.score(X_eval,y_eval)
    dt_file.write(f'Random Forest Dtrees: {accuracy}\n')

    #Extremely Randomized Trees
    extra_rf_dtree = ExtraTreesRegressor(n_estimators=8).fit(X_fit,y_fit)
    accuracy = extra_rf_dtree.score(X_eval,y_eval)
    dt_file.write(f'Extremely Randomized Dtrees: {accuracy}\n')


    #Gradient Boosting Trees
    gb_tree = GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_fit, y_fit)
    accuracy = gb_tree.score(X_eval, y_eval)
    dt_file.write(f'Gradient Boosting Dtrees {accuracy}')

# features = ['Date of Joining', 'Gender', 'Company Type', 'WFH Setup Available', 'Designation', 'Resource Allocation', 'Mental Fatigue Score']

def runtime():
    parser = argparse.ArgumentParser(
        description='Fit & Score Dtree Regressors')
    parser.add_argument('-f', '--features', nargs='+', default=['Gender', 'Company Type', 'WFH Setup Available', 'Designation', 'Resource Allocation', 'Mental Fatigue Score'],
                        help='the features to include')
    parser.add_argument('-cf', '--cat_features', nargs='+',
                        default=['Gender', 'Company Type', 'WFH Setup Available'],
                        help='the features to include')
    parser.add_argument('-ofn', '--outputfilename', default="output1")

    args = parser.parse_args()

    features = args.features
    cat_features = args.cat_features

    X_fit, y_fit = read_dataset('train.csv', features, cat_features)
    X_fit, X_eval, y_fit, y_eval = train_test_split(X_fit, y_fit, test_size=0.30, random_state=2)

    dt_file = open(args.outputfilename + '.txt', 'w')

    dtrees(X_fit, y_fit, X_eval, y_eval, features, dt_file)

runtime()