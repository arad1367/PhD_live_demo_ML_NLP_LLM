# debug run:        flask --app filename run --debug

from flask import Flask, render_template, request
import csv
import pickle

app = Flask(__name__)

# load the models
load_model_HUN = pickle.load(open("./clfmodel.pkl",'rb'))
load_model_IRN = pickle.load(open("./clfmodel_IRN.pkl",'rb'))

# Our features classes in label format
class_App_names_HUN = ['FoodPanda', 'Wolt', 'Spar', 'Tesco online', 'myLidl']
class_App_names_IRN = ['Snappfood', 'Jimomarket', 'Digikala']
class_gender_names = ["Male", "Female"]
class_education_names = ["Under Diploma and Diploma", "Associate", "Bachelor", "Master", "PhD"]

@app.route("/")
@app.route("/home")
def home():  ## url_for work with name of def
    return render_template("index.html")

@app.route("/sentiment_analysis")
def sentiment_analysis():
    return render_template("work.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/slides")
def slides():
    return render_template("slide.html")

def write_to_csv_hun(data):
    """
    This function can add data to .csv database
    """
    with open('databaseHUN.csv', mode='a', newline="") as csv_database_hun:
        gender = data['gender']
        education = data['education']
        age = data['age']
        exp_online = data['exp_online']
        exp_app = data['exp_app']
        csv_writter = csv.writer(csv_database_hun, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        csv_writter.writerow([gender, education, age, exp_online, exp_app])

def write_to_csv_irn(data):
    """
    This function can add data to .csv database
    """
    with open('databaseIRN.csv', mode='a', newline="") as csv_database_irn:
        gender = data['gender']
        education = data['education']
        age = data['age']
        exp_online = data['exp_online']
        exp_app = data['exp_app']
        csv_writter = csv.writer(csv_database_irn, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        csv_writter.writerow([gender, education, age, exp_online, exp_app])

@app.route("/predict_hungary", methods=['POST','GET']) # This /predict_hungary will be active with action in form
def predict_HUN(): # After activation of route ---> def will be active
    if request.method == 'POST':
        try:
            # data is our forms input
            data = request.form.to_dict()
            # Add to .csv database
            write_to_csv_hun(data)
            # print(f"form new information in dict status: {data}")

            # manage form part and ML part
            gender = int(request.form['gender'])
            education = int(request.form['education'])
            age = int(request.form['age'])
            exp_online = int(request.form['exp_online'])
            exp_app = int(request.form['exp_app'])
            predicted_label = load_model_HUN.predict([[gender, education, age, exp_online, exp_app]])[0]
            predicted_App_class = class_App_names_HUN[predicted_label-1]
            predicted_Gender_class = class_gender_names[gender - 1]
            predicted_Education_class = class_education_names[education - 1]
            return render_template("/predicthun.html",
                                   gender = predicted_Gender_class,
                                   education = predicted_Education_class,
                                   age = age,
                                   exp_online = exp_online,
                                   exp_app = exp_app,
                                   predicted_app = predicted_App_class)
        except:
            return '<h1>Your data is out of bound! Try again!</h1>'
    return render_template("/predicthun.html")

@app.route("/predict_iran", methods=['POST','GET']) # This /predict_hungary will be active with action in form
def predict_IRN(): # After activation of route ---> def will be active
    if request.method == 'POST':
        try:
            # data is our forms input
            data = request.form.to_dict()
            # Add to .csv database
            write_to_csv_irn(data)
            # print(f"form new information in dict status: {data}")

            # manage form part and ML part
            gender = int(request.form['gender'])
            education = int(request.form['education'])
            age = int(request.form['age'])
            exp_online = int(request.form['exp_online'])
            exp_app = int(request.form['exp_app'])
            predicted_label = load_model_IRN.predict([[gender, education, age, exp_online, exp_app]])[0]
            predicted_App_class = class_App_names_IRN[predicted_label-1]
            predicted_Gender_class = class_gender_names[gender - 1]
            predicted_Education_class = class_education_names[education - 1]
            return render_template("/predictionirn.html",
                                   gender = predicted_Gender_class,
                                   education = predicted_Education_class,
                                   age = age,
                                   exp_online = exp_online,
                                   exp_app = exp_app,
                                   predicted_app = predicted_App_class)
        except:
            return '<h1>Your data is out of bound! Try again!</h1>'
    return render_template("/predictionirn.html")
