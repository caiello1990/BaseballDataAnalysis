import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
df = pd.read_csv('df_bp1.csv')

df.season.value_counts().sort_index()

hv_mean = df.home_victory.mean()
df.run_diff.value_counts()

df.outs_total.value_counts()

df.home_victory[df.outs_total == 53].mean()
df.game_no_h.value_counts().sort_index()
df['day_night'] = df['day_night'].map({'D': 0, 'N': 1})

feature_column_names = ['dblheader_code',
      'runs_v',
      'runs_h',
      'day_night',
      # 'pitcher_start_id_h',
      # 'pitcher_start_name_h',
      # 'batter5_id_h',
      # 'batter8_id_h',
      'OBS_162_h',
      'OBS_162_v',
      'ERR_162_v',
      'BATAVG_30_h',
      'BATAVG_30_v',
      'OBP_30_h',
      'OBP_30_v',
      'SLG_30_h',
      'SLG_30_v',
      'OBS_30_h',
      'OBS_30_v',
      'SB_30_h',
      'SB_30_v',
      'CS_30_h',
      'CS_30_v',
      'ERR_30_h',
      'ERR_30_v'
      ]
encoded_cols = ['team_v',
      'team_h',
      ]

# filter useful columns out of df
df_filtered = df.loc[:, feature_column_names]

# one hot encode df
df_encoded_filtered = df.loc[:, encoded_cols]
df_encoded = pd.get_dummies(df_encoded_filtered, columns=encoded_cols)
#df_encoded.to_csv('df_encoded.csv', index=False)


df_final = pd.concat([df_filtered, df_encoded], axis=1)
#df_final.to_csv('df_final.csv', index=False)

# make numPy array of games
game_array = df.to_numpy()
target_var = np.array(game_array[:, 163])

df_final = df_final.fillna(0)
X = df_final
y = df['home_victory']


#df_encoded.to_csv('df_encoded_bp1.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def runModel(modelToRun, X_train, X_test, y_train, y_test):
      precision_metrics = {}
      # fit model to our training data
      modelToRun.fit(X_train, y_train)

      # create model predictions for training and test data
      train_model_predict = modelToRun.predict(X_train)
      test_model_predict = modelToRun.predict(X_test)

      # find precision metric values for model training data
      train_model_precision = precision_score(y_train, train_model_predict)
      train_model_recall = recall_score(y_train, train_model_predict)
      train_model_accuracy = accuracy_score(y_train, train_model_predict)
      train_model_f1 = f1_score(y_train, train_model_predict)
      train_model_confMat = confusion_matrix(y_train, train_model_predict)

      # find precision metric values for model test data
      test_model_precision = precision_score(y_test, test_model_predict)
      test_model_recall = recall_score(y_test, test_model_predict)
      test_model_accuracy = accuracy_score(y_test, test_model_predict)
      test_model_f1 = f1_score(y_test, test_model_predict)
      test_model_confMat = confusion_matrix(y_test, test_model_predict)

      # fill empty dictionary with train and test results
      precision_metrics['train'] = [train_model_precision, train_model_recall, train_model_accuracy, train_model_f1,
                                    train_model_confMat]

      precision_metrics['test'] = [test_model_precision, test_model_recall, test_model_accuracy, test_model_f1,
                                   test_model_confMat]

      return precision_metrics

#define function to print precision metrics
def precisionMetricPrinter(precision_metrics):
    print("--Precision:", precision_metrics[0])
    print("--Recall:", precision_metrics[1])
    print("--Accuracy:", precision_metrics[2])
    print("--F1 Score:", precision_metrics[3])
    print("--Confusion Matrix:")
    print(precision_metrics[4])

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print("Train Results - Ridge Regression:")
#precisionMetricPrinter(ridge['train'])

#Logistic Regression
logreg = LogisticRegression()
logreg_run = runModel(logreg, X_train, X_test, y_train, y_test)

print("Train Results - Logistic Regression:")
precisionMetricPrinter(logreg_run['train'])
print("Test Results - Logistic Regression:")
precisionMetricPrinter(logreg_run['test'])

#Support Vector Machine
svc = SVC()
svc_run = runModel(svc, X_train, X_test, y_train, y_test)

print("Train Results - Support Vector Classification:")
precisionMetricPrinter(svc_run['train'])
print("Test Results - Support Vector Classification:")
precisionMetricPrinter(svc_run['test'])

#Random Forest
rf = RandomForestClassifier()
rf_run = runModel(rf, X_train, X_test, y_train, y_test)

print("Train Results - Random Forest:")
precisionMetricPrinter(rf_run['train'])
print("Test Results - Random Forest:")
precisionMetricPrinter(rf_run['test'])

#Decision Tree
dtc = DecisionTreeClassifier()
dtc_run = runModel(dtc, X_train, X_test, y_train, y_test)

print("Train Results - Decision Tree:")
precisionMetricPrinter(dtc_run['train'])
print("Test Results - Decision Tree:")
precisionMetricPrinter(dtc_run['test'])

#Naive Bayes
gnb = GaussianNB()
gnb_run = runModel(gnb, X_train, X_test, y_train, y_test)

print("Train Results - Naive Bayes:")
precisionMetricPrinter(gnb_run['train'])
print("Test Results - Naive Bayes:")
precisionMetricPrinter(gnb_run['test'])

#Gradient Boosting Machine
gbc = GradientBoostingClassifier()
gbc_run = runModel(gbc, X_train, X_test, y_train, y_test)

print("Train Results - Gradient Boosting Machine:")
precisionMetricPrinter(gbc_run['train'])
print("Test Results - Gradient Boosting Machine:")
precisionMetricPrinter(gbc_run['test'])

#XGBoost
xgboost = XGBClassifier()
xgboost_run = runModel(xgboost, X_train, X_test, y_train, y_test)

print("Train Results - XGBoost:")
precisionMetricPrinter(xgboost_run['train'])
print("Test Results - XGBoost:")
precisionMetricPrinter(xgboost_run['test'])

