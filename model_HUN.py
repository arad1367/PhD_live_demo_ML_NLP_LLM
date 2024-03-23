import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import warnings

# 0. App labels
# 1 FoodPanda, 2 Wolt, 3 Spar, 4 Tesco online, 5 myLidl

# 1. Get the data
data = pd.read_csv("./data/HUNGARYdata.csv")
print(data.head())

# 2. Features & target
X = data.drop("Apps", axis=1)
y = data["Apps"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Make model
clf_model = RandomForestClassifier()

# 5. Fit the model
clf_model.fit(X_train, y_train)
# print(clf_model.get_params())

# 6. Make prediction
y_pred = clf_model.predict(X_test)

# 7. Check Accuracy
acc = clf_model.score(X_test, y_pred)
print(f"Accuracy of RFC model is: {acc * 100:0.2f} %")

# 8. Save the model
# save the model
# filename = 'clfmodel.pkl'
# pickle.dump(clf_model, open(filename, 'wb'))

# 9. Load model
try:
    print("Model is loading ... ")
    time.sleep(2)
    load_model = pickle.load(open("./models/clfmodel.pkl",'rb'))
    print("clfmodel is loaded successfuly.")
except:
    print("Somethong is wrong")

# 10. Make prediction with loaded model

# Suppress the specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")

# Our features classes in label format
class_App_names = ['FoodPanda', 'Wolt', 'Spar', 'Tesco online', 'myLidl']
class_gender_names = ["Male", "Female"]
class_education_names = ["Under Diploma and Diploma", "Associate", "Bachelor", "Master", "PhD"]

# Features for prediction
Gender = 1
Education = 5
Age = 35
Exp_online = 10
Exp_app = 3

# make prediction live
predicted_label = int((load_model.predict([[Gender, Education, Age, Exp_online, Exp_app]])).item())

# See result
predicted_App_class = class_App_names[predicted_label-1]
predicted_Gender_class = class_gender_names[Gender - 1]
predicted_Education_class = class_education_names[Education - 1]
print(f"In Hungary, for a {predicted_Gender_class}, with education level of {predicted_Education_class}, age of {Age}, with online experience of {Exp_online} year/years, and shopping experience from online Apps {Exp_app} year/years, It seems that the prefetable Grocery App is : {predicted_App_class}.")
