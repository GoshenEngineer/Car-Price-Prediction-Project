#importing the dependencies
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#parameter
outfile = "nn_model.keras"
#Data Preparation
#loading the dataset to the Pandas
df = pd.read_csv(r'C:\Users\Admin\Downloads\Car Prediction Prediciton\cardekho_dataset.csv', index_col = 0)
pd.set_option('display.max_columns', None)
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numerical_columns = list(df.dtypes[df.dtypes == 'float64'].index) + list(df.dtypes[df.dtypes == 'int64'].index)
for c in categorical_columns[1:6]:
    print(str.upper(c))
    print(df[c].unique(), df[c].nunique())
df.columns
#Create the features and the Target Varaible
X = df.drop(['car_name','selling_price'], axis = 1)
Y = df['selling_price'] 
#Split Categorical and Numerical Features
numerical_cols = ['mileage', 'max_power', 'vehicle_age', 'km_driven', 'engine', 'seats']
categorical_cols = [
    'brand', 
    'seller_type', 
    'fuel_type', 
    'transmission_type'
    ]
model_col = ['model'] #high-cardinality
# Numerical pipeline
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])
#Categorical pipeline (using OrdinalEncoder for embedding later)
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
model_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
#Combine into ColumnTransformer
x_preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols),
    ('model', model_pipeline,model_col)
])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_poc = x_preprocessor.fit_transform(X_train)
X_test_proc = x_preprocessor.transform(X_test)

#scaling the target variable
from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1))

# Separate last column (model) from the rest
num_cat_train = X_train_poc[:, :-1]
model_train = X_train_poc[:, -1].astype('int32')
num_cat_test = X_test_proc[:, :-1]
model_test = X_test_proc[:, -1].astype('int32')
# get input shapes
print("num_cat_train shape:", num_cat_train.shape)
print("model_train shape:", model_train.shape)

#parameters for the embedding layers
model_vocab_size = len(X_train['model'].unique())
embedding_dim = 16  # good starting size

#creating the Artificial neural network
#Inputs
num_cat_input = layers.Input(shape=(num_cat_train.shape[1],))
model_input = layers.Input(shape=(1,))
#Embedding for model
embed = layers.Embedding(input_dim=model_vocab_size+1, output_dim=embedding_dim)(model_input)
embed = layers.Flatten()(embed)
#Combine
combined = layers.Concatenate()([num_cat_input, embed])
#Dense layers
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1)(x)  #regression output
#Build + compile
nn_model = models.Model(inputs=[num_cat_input, model_input], outputs=output)
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
#Train the model
nn_model.fit(
    [num_cat_train, model_train], y_train_scaled,
    validation_data=([num_cat_test, model_test], y_test_scaled),callbacks=[callback],
    epochs=100, batch_size=32
)
y_pred_scaled = nn_model.predict([num_cat_train, model_train])
# Convert predictions back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
#Save the Model
nn_model.save(outfile)
joblib.dump(x_preprocessor, ' preprocessor.joblib')
joblib.dump(scaler_y, 'scaler.joblib')