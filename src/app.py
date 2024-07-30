# app.py
from flask import Flask, request, render_template
from utils import load_model, preprocess_input

app = Flask(__name__)

# Load models
dt_model = load_model('models/decision_tree.pkl')
rf_model = load_model('models/random_forest.pkl')
svm_model = load_model('models/svm.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        data = [sepal_length, sepal_width, petal_length, petal_width]
        preprocessed_data = preprocess_input(data)

        dt_prediction = dt_model.predict(preprocessed_data)
        rf_prediction = rf_model.predict(preprocessed_data)
        svm_prediction = svm_model.predict(preprocessed_data)

        return render_template('index.html', dt_prediction=dt_prediction[0],
                               rf_prediction=rf_prediction[0],
                               svm_prediction=svm_prediction[0])
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
