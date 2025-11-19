# Implement-Simple-Linear-Regression-model
python program to implement  Simple Linear Regression model for given dataset

 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

data = {
    'Attendance':[85,70,90,60,95,55,80,50,75,92],
    'Assignment_Score':[80,65,88,58,91,45,75,40,70,89],
    'Exam_score':[78,60,85,55,90,50,72,45,68,87],
    'Result':['Pass','Fail','Pass','Fail','Pass','Fail','Pass','Fail','Pass','Pass']
    }

df = pd.DataFrame(data)
print("StudentDataset:")
print(df)

x = df[['Attendance','Assignment_Score','Exam_score']]
y = df['Result']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf_model = rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)

print("\n Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\n Classification Report")
print(classification_report(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred))

new_student = [[82,75,80]]
pred = rf_model.predict(new_student)
print("\n Prediction fofr new students [Attendance = 82,Assignment = 75,Exam = 80]",pred[0])
