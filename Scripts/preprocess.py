import pandas as pd
from sklearn.preprocessing import StandardScaler
def clean_data(df):
    yes_no_cols=["Partner","Dependents", "PhoneService", "PaperlessBilling","Churn"]
    nointernet_cols=["OnlineSecurity", "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df=df.drop(columns=["customerID"])
    df["PaymentMethod"]=df["PaymentMethod"].fillna("Unknown")
    df["gender"]=df["gender"].map({"Male":0, "Female":1})
    for column in yes_no_cols:
        df[column]=df[column].map({"No":0, "Yes":1})
    df["MultipleLines"]=df["MultipleLines"].map({"No":0,"Yes":1,"No phone service":2})
    for column in nointernet_cols:
        df[column]=df[column].map({"No":0, "Yes":1,"No internet service":2})
    df["PaymentMethod"]=df["PaymentMethod"].map({"Electronic check":0, "Mailed check":1,"Bank transfer (automatic)":2, "Credit card (automatic)":3,"Unknown":4})
    df["InternetService"]=df["InternetService"].map({"No":0,"Fiber optic":1, "DSL":2})
    df["Contract"]=df["Contract"].map({"Month-to-month":0, "One year":1,"Two year":2})
    df["TotalCharges"]=df["TotalCharges"].replace([" ",""], pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())
     
    return df
def scale_data(x_train, x_test):
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler
    