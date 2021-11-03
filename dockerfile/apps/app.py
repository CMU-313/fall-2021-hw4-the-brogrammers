from flask import Flask, jsonify, request
from flasgger import Swagger
from flasgger.utils import swag_from
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def hello():
    return "try the predict route it is great!"

@swag_from("./api_docs/predict_doc.yml")
@app.route('/predict', methods=['GET'])
def predict():
	 #use entries from the query string here but could also use json
     age = request.args.get('age')
     absences = request.args.get('absences')
     health = request.args.get('health')
     # need to include bad returns if improper
     data = [[age],[health],[absences]]
     query_df = pd.DataFrame({ 'age' : pd.Series(age) ,'health' : pd.Series(health) ,'absences' : pd.Series(absences)})
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify(np.asscalar(prediction))

if __name__ == '__main__':
    clf = joblib.load('/apps/model2.pkl')
    app.run(host="0.0.0.0", debug=True)