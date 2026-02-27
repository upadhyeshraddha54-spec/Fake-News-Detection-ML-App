from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    # Transform input text
    transformed_text = vectorizer.transform([text])

    # Get prediction
    prediction = model.predict(transformed_text)[0]

    print("TEXT:", text)
    print("RAW PREDICTION:", prediction)

    # Correct label mapping
    if prediction == 1:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)