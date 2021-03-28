import argparse
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

def dtrees(X, y, features, dt_file):
    X_fit, X_eval, y_fit, y_eval = train_test_split(X, y, test_size=0.30, random_state=2)
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
    accuracy = rf_dtree.score(X_eval,y_eval)
    dt_file.write(f'Extremely Randomized Dtrees: {accuracy}\n')


    #Gradient Boosting Trees
    # gb_tree = GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_fit, y_fit)
    # accuracy = gb_tree.score(X_eval, y_eval)
    # dt_file.write(f'Gradient Boosting Dtrees {accuracy}')

def runtime():
    parser = argparse.ArgumentParser(
        description='Fit & Score Dtree Regressors')
    parser.add_argument('-dfn', '--data_filename', default="data_3.p",
                        help='which data pickled file to read in')
    parser.add_argument('-f', '--features', nargs='+', default=['Predicted Precipitation', 'Total Miles Plowed'],
                        help='Should this program run decision trees (dt) or random forrest baggging (rf)')
    parser.add_argument('-ofn', '--outputfilename', default="output3")

    args = parser.parse_args()

    dfn = args.data_filename

    X, y = pickle.load(open(dfn, 'rb'))
    features = args.features
    dt_file = open(args.outputfilename + '.txt', 'w')

    dtrees(X, y, features, dt_file)

runtime()