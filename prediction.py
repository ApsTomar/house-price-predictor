import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from adaBoost import adaBoost


def load_house_data():
    data = pd.read_csv(path + '/house-price-predictor/dataset/house_data.csv')
    features_data = data[
        ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement',
         'waterfront', 'yr_built', 'lat', 'bedrooms', 'long']]
    X_train, X_test, y_train, y_test = train_test_split(features_data.values, data.price.values, test_size=0.2)
    return data, X_train, X_test, y_train, y_test


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocessing.set_path(path)

data, X_train, X_test, y_train, y_test = load_house_data()
# data visualization:
# preprocessing.preprocess_data(data)

# applying adaBoost:
adaBoost_score = adaBoost.ada_boost(X_train, X_test, y_train, y_test)
print('adaBoost: explained_variance_score: %f' % adaBoost_score)


