from Scripts.load_data import load_data
from Scripts.preprocess import clean_data, scale_data
from sklearn.model_selection import train_test_split
from Scripts.train_model import train, evaluate
from Scripts.model_io import save_model
df=load_data("data/customer-churn.csv")
print("Data Loaded successfully")
df_clean=clean_data(df)

#split features and target
x=df_clean.drop(columns=["Churn"])
y=df_clean["Churn"]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#Scale data
x_train_scaled, x_test_scaled, scaler= scale_data(x_train, x_test)

# training and evaluation

model=train(x_train_scaled, y_train)
accuracy= evaluate(model, x_test_scaled, y_test)
print(f"Accuracy of model is {accuracy}")

#save model and scaler
save_model(model, scaler)
