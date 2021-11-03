from flask import Flask, jsonify, request
from flasgger import Swagger
from flasgger.utils import swag_from
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
swagger = Swagger(app)

@swag_from("./api_docs/predict_doc.yml")
@app.route('/predict', methods=['GET'])
def predict():
    include = ['school', 'age', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'higher', 'freetime', 'Dalc', 'Walc', 'health', 'absences']
	 #use entries from the query string here but could also use json
    school = request.args.get('school')
    age = request.args.get('age')
    traveltime = request.args.get('traveltime')
    studytime = request.args.get('studytime')
    failures = request.args.get('failures')
    schoolsup = request.args.get('schoolsup')
    famsup = request.args.get('famsup')
    higher = request.args.get('higher')
    freetime = request.args.get('freetime')
    Dalc = request.args.get('Dalc')
    Walc = request.args.get('Walc')
    health = request.args.get('health')
    absences = request.args.get('absences')

     # need to include bad returns if improper
    query_df = pd.DataFrame({
        'school' : pd.Series(school),
        'age' : pd.Series(age),
        'traveltime' : pd.Series(traveltime),
        'studytime' : pd.Series(studytime),
        'failures' : pd.Series(failures),
        'schoolsup' : pd.Series(schoolsup),
        'famsup' : pd.Series(famsup),
        'higher' : pd.Series(higher),
        'freetime' : pd.Series(freetime),
        'Dalc' : pd.Series(Dalc),
        'Walc' : pd.Series(Walc),
        'health' : pd.Series(health),
        'absences' : pd.Series(absences)
        })

    with open("/apps/encoder", "rb") as f:
        encoder = pickle.load(f)
    with open("/apps/categoricals", "rb") as f:
        categoricals = pickle.load(f)
    q_cat = pd.DataFrame(encoder.transform(query_df[categoricals]).toarray(), columns=encoder.get_feature_names(categoricals))
    q_val = query_df[query_df.columns.difference(categoricals)]
    query = q_val.join(q_cat)
    prediction = clf.predict(query)
    return jsonify(np.asscalar(prediction))

if __name__ == '__main__':
    # while True:
    #     pass
    clf = joblib.load('/apps/model_final.pkl')
    app.run(host="0.0.0.0", debug=True)