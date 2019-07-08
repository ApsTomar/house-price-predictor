from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import explained_variance_score


def ada_boost(x_train, x_test, y_train, y_test):
    reg = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, loss='exponential')
    reg.fit(x_train, y_train)
    reg_pred = reg.predict(x_test)
    print('adaBoost: r-squared score: %f' % reg.score(x_train, y_train))
    exp_variance_score = explained_variance_score(reg_pred, y_test)
    return exp_variance_score
