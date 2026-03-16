from flask import Flask, render_template, request, jsonify
from hybrid_recommender import recommend
from preprocess import load_dataset

app = Flask(__name__)

data = load_dataset()


@app.route("/", methods=["GET","POST"])
def index():

    results = []

    if request.method == "POST":
        course = request.form["course"]
        results = recommend(course,5)

    return render_template("index.html", results=results)


# autocomplete search
@app.route("/search")
def search():

    query = request.args.get("q")

    matches = data[data["title"].str.lower().str.contains(query.lower())]

    suggestions = matches["title"].head(10).tolist()

    return jsonify(suggestions)


if __name__ == "__main__":
    app.run(debug=True)