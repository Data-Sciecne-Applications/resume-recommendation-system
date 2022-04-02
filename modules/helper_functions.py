from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def train_val_test_split(x, y, train_size, val_size, test_size, random_state=41):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size / (test_size + val_size), random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

def tunning(model, vectorizer, crit, cv, x, y):
    Grid = GridSearchCV(model, crit, cv=cv)
    Grid.fit(vectorizer.fit_transform(x), y)
    return Grid.best_estimator_