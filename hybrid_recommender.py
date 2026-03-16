from preprocess import load_dataset
from semantic_similarity import build_similarity

data = load_dataset()

similarity = build_similarity(data)


def recommend(course, top_n=5):

    course = course.lower()

    matches = data[data["title"].str.lower().str.contains(course)]

    if len(matches) == 0:
        return []

    idx = matches.index[0]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:top_n+1]

    indices = [i[0] for i in scores]

    results = data.iloc[indices][["title","url"]]

    return results.to_dict(orient="records")