import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("1.Loading and preparing the data...")

# Load the cleaned data
df=pd.read_csv('data_cleaned.csv')

df['event_time']=pd.to_datetime(df['event_time'])
df=df.dropna(subset=['user_session'])

print("2. Engineering features for the model...")
# group by session to calculate features
session_features = df.groupby('user_session').agg(
    total_events=('event_type','count'),
    has_cart=('event_type', lambda x: (x=='cart').max()),
    has_purchase=('event_type', lambda x: (x=='purchase').max()),
    session_duration =('event_time', lambda x: (x.max()-x.min()).total_seconds()),
    hour_of_day = ('event_time', lambda x: x.dt.hour.iloc[0]),
    is_weekend= ('event_time', lambda x: 1 if x.dt.dayofweek.iloc[0]>=5 else 0)

).reset_index()
# filter for only sessions that actually added something to the cart
cart_sessions = session_features[session_features['has_cart']==1].copy()

# create the target variable: 1 if abandoned (no purchase), 0 if purchased
cart_sessions['abandoned_cart']=(cart_sessions['has_purchase']==0).astype(int)

print(f"total cart sessions to analyze:{len(cart_sessions)}")

print("3. Training the XGboost Model...")
# define our features (X) and Target (y)
features = ['total_events','session_duration','hour_of_day','is_weekend']
x= cart_sessions[features]
y= cart_sessions['abandoned_cart']

# split into 80% training data and 20% testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# initialize and train the model
model = xgb.XGBClassifier(use_label_encoder= False, eval_metric='logloss')
model.fit(x_train, y_train)

print("4. Evaluating the model...")
# predict on the test set
y_pred = model.predict(x_test)

# print the results
print("\n---model performance---")
print(classification_report(y_test,y_pred))

# 5. Extracting bussiness insights 
print("\n --- feature importance ---")
importance= model.feature_importances_
for i, col in enumerate(features):
    print(f"{col}: {importance[i]*100 : .2f}%")
