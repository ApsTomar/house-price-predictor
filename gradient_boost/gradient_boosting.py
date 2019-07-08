from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score


def gradient_boost(x_train, x_test, y_train, y_test):
    reg = GradientBoostingRegressor(n_estimators=100, loss='ls', max_depth=5, learning_rate=0.2, min_samples_split=2)
    reg.fit(x_train, y_train)
    prediction = reg.predict(x_test)
    print('gradient_boost: r-squared score: %f' % reg.score(x_train, y_train))
    exp_variance_score = explained_variance_score(prediction, y_test)
    return exp_variance_score
