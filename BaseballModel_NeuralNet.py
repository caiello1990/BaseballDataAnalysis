import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
df = pd.read_csv('df_bp1.csv')


df.season.value_counts().sort_index()

hv_mean = df.home_victory.mean()
df.run_diff.value_counts()

df.outs_total.value_counts()

df.home_victory[df.outs_total == 53].mean()
df.game_no_h.value_counts().sort_index()

feature_column_names = ['dblheader_code',
      'team_v',
      'team_h',
      'runs_v',
      'runs_h',
      'day_night',
      'pitcher_start_id_h',
      'pitcher_start_name_h',
      'batter5_id_h',
      'batter8_id_h',
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

# filter useful columns out of df
df_filtered = df.loc[:, feature_column_names]


# one hot encode df
df_encoded = pd.get_dummies(df, columns=feature_column_names)

# make numPy array of games
game_array = df.to_numpy()
target_var = np.array(game_array[:, 163])
target_var_str = [str(x) for x in target_var]
feature_array = np.array(game_array)

feature_array_str = [str(x) for x in feature_array]

# # Define Model:
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
 ])


# # Compile the Model: Configure Training
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Adjust based on your task



# # Train the Model:
X_train, X_test, y_train, y_test = train_test_split(feature_array_str, target_var_str, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
#
# # # Evaluate the Model:
# model.evaluate(X_test, y_test)  # Replace with validation data if not using a separate test set
#
# prediction = model.predict()
# print('-----------------------------------------------------------------------------------------')
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
# print('-----------------------------------------------------------------------------------------')
# print(prediction)