from flask import Flask, render_template,request
import pandas as pd
import pickle

model = pickle.load(open("car_prediction.pkl","rb"))
data = pd.read_csv(r"C:\Users\Dell\Desktop\ml\process_car_data.csv")
X = data.drop("selling_price", axis=1)
y = data["selling_price"]


categorical_column = X.select_dtypes(include="object").columns.to_list()
numerical_column = X.select_dtypes(include="number").columns.to_list()

item = {}

app = Flask(__name__)

@app.route("/")

def index():
    for column in categorical_column:
        item[column] = sorted(data[column].unique())
    year = sorted(data["year"].unique())
    seat = sorted(data["seats"].unique().astype(int))

    return render_template("index.html", items=item,years=year, seats=seat)

pre={}
@app.route("/predict",methods=["POST"])
def predict():
    for column in categorical_column:
       pre[column] = request.form.get(column)
    for column in numerical_column:
       pre[column] = int(request.form.get(column))
       
    
    prediction_value = model.predict(pd.DataFrame([pre]))
    
    return str(prediction_value[0])

if __name__ == "__main__":
    app.run(debug=True)
