from sklearn.tree import DecisionTreeClassifier # Result: Categorical value
from sklearn.tree import DecisionTreeRegressor  # Result: Numeric value
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np

# DecisionTreeClassifier
  
# 데이터 불러오기
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

# Define Model
  # *, criterion='gini', splitter='best', 
  # max_depth=None, min_samples_split=2, min_samples_leaf=1, 
  # min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
  # max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
  # class_weight=None, ccp_alpha=0.0
tree = DecisionTreeClassifier(random_state=0)

# Train
tree.fit(X_train, y_train)

# Result
print("훈련 세트 정확도: {:.2f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(tree.score(X_test, y_test)))
#####################################################################################
# DecisionTreeRegressor
# 랜덤 데이터 생성
rg = np.random.RandomState(1)
X = np.sort(5 * rg rang.rand(100,1), axis=0)
Y = np.sin(X)
Y[::5] += 5 * (0.5 - rg.rand(15))

# Define Model
  # *, criterion='mse', splitter='best', 
  # max_depth=None, min_samples_split=2, min_samples_leaf=1, 
  # min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
  # max_leaf_nodes=None, min_impurity_decrease=0.0, 
  # min_impurity_split=None, ccp_alpha=0.0
DecisionTreeRegressor_1 = DecisionTreeRegressor(max_depth = 2)
DecisionTreeRegressor_2 = DecisionTreeRegressor(max_depth = 5)

# Train
DecisionTreeRegressor_1.fit(X,Y)
DecisionTreeRegressor_2.fit(X,Y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
Predicted_1 = DecisionTreeRegressor_1.predict(X_test)
Predicted_2 = DecisionTreeRegressor_2.predict(X_test)

# Result
MSE_1 = mean_squared_error(Y, Predicted_1);
MSE_2 = mean_squared_error(Y, Predicted_2);
