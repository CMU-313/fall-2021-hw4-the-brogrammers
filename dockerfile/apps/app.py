from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return "try the predict route it is great!"

@app.route('/predict')
def predict():
	 #use entries from the query string here but could also use json
    features = ['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
        'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    vals = []
    for f in features:
        vals.append(request.args.get(f))
    
    query_df = pd.DataFrame(vals, columns=features)

    with open("encoder", "rb") as f: 
        encoder = pickle.load(f)
    with open("categoricals", "rb") as f: 
        categoricals = pickle.load(f)

    query_cat = pd.DataFrame(encoder.transform(query_df[categoricals]).toarray(), columns=encoder.get_feature_names(categoricals))
    query_val = query_df[query_df.columns.difference(categoricals)]
    query = query_val.join(query_cat) # this order here is important

    prediction = clf.predict(query)
    return jsonify(np.asscalar(prediction))

if __name__ == '__main__':
    clf = joblib.load('/apps/model2.pkl')
    app.run(host="0.0.0.0", debug=True)