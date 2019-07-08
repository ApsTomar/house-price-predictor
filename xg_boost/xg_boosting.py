from xgboost import XGBRegressor
from sklearn.metrics import explained_variance_score


def xg_boost(x_train, x_test, y_train, y_test):
    reg = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.1, objective='reg:squarederror')
    reg.fit(x_train, y_train)
    prediction = reg.predict(x_test)
    print('xg_boost: r-squared score: %f' % reg.score(x_train, y_train))
    exp_variance_score = explained_variance_score(prediction, y_test)
    return exp_variance_score
