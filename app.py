import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

# Load CSV and model
@st.cache_data
def load_data():
    return pd.read_csv('Shoplytics_Nexus_Data.csv', encoding='unicode_escape')

@st.cache_resource
def load_model():
    return joblib.load('linear_sales_model.joblib')

# Preprocessing function
def preprocess(df):
    df = df.drop(columns=['User_ID','Cust_name','Product_ID','unnamed1','Status'], errors='ignore')
    df = df.dropna(subset=['Amount']).reset_index(drop=True)
    y = df['Amount']
    X = df.drop(columns=['Amount'])

    # Impute missing values
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])

    cat_cols = X.select_dtypes(include=['object']).columns
    X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

    # Encode categoricals
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    enc_array = encoder.fit_transform(X[cat_cols])
    enc_df = pd.DataFrame(enc_array,
                          columns=encoder.get_feature_names_out(cat_cols),
                          index=X.index)

    X_final = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)
    return X_final, y

# Load everything
df = load_data()
model = load_model()
X_all, y_all = preprocess(df)

# Sidebar filters
st.sidebar.header('🔍 Filter by Gender')
genders = st.sidebar.multiselect('Select Gender', df['Gender'].unique(),
                                 default=df['Gender'].unique())

df_filt = df[df['Gender'].isin(genders)]
X_filt, y_filt = preprocess(df_filt)
y_pred = model.predict(X_filt)
residuals = y_filt - y_pred

# Dashboard display
st.title('📊 Shoplytics Sales Dashboard')
st.metric(label='Average Predicted Amount', value=f"{y_pred.mean():.2f}")

# Residual plot
fig_res = px.scatter(x=y_pred, y=residuals,
                     labels={'x':'Predicted Amount','y':'Residuals'},
                     title='Residuals vs Predictions')
st.plotly_chart(fig_res)

# Feature importances
coeffs = pd.Series(model.coef_, index=X_all.columns)
top_features = coeffs.abs().sort_values(ascending=False).head(10)
fig_imp = px.bar(top_features, orientation='h', title='Top 10 Feature Importances')
st.plotly_chart(fig_imp)

# Distribution plot
fig_dist = px.histogram(df_filt, x='Amount', nbins=30, title='Sales Amount Distribution')
st.plotly_chart(fig_dist)

