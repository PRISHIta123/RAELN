from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=20, random_state=0) 
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

cutoff = 0.99

y_pred_classes = np.zeros_like(y_pred) 
y_pred_classes[y_pred > cutoff] = 1

y_test_classes = np.zeros_like(y_pred) 
y_test_classes[y_test_classes > cutoff] = 1

print(classification_report(y_test_classes, y_pred_classes))

Accuracy=accuracy_score(y_test_classes, y_pred_classes)*100

print(Accuracy)
