from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train(x,y):
    model=RandomForestClassifier(class_weight='balanced', max_iter=1000)
    model.fit(x,y)
    return model
def evaluate(model,x,y):
    y_predict = model.predict(x)
    accuracy= accuracy_score(y,y_predict)
    print(confusion_matrix(y,y_predict))
    print(classification_report(y,y_predict))
    return accuracy