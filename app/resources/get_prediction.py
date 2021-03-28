import sys
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor


sen = sys.argv[1]
ra = sys.argv[2]
mfs = sys.argv[3]

def get_prediction():
    # url = 'https://cdn.glitch.com/97010f47-023e-493e-b71d-b283c5f1a1c7%2Fdtree.p?v=1607876471105'
    # r = requests.get(url, allow_redirects=True)

    # dtree = pickle.loads(r.content)
    dtree = pickle.load( open( "dtree.p", "rb" ))

    oup = dtree.predict([[sen, ra, mfs]])
    return oup

print(get_prediction())