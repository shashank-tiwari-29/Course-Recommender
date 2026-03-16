import pandas as pd
import re
import nltk
import urllib.parse
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def load_dataset():

    coursera = pd.read_csv("data/Coursera.csv")
    udemy = pd.read_csv("data/udemy_courses.csv")

    coursera = coursera[["course_title","course_organization","course_difficulty"]]
    coursera.columns = ["title","org","level"]

    udemy = udemy[["course_title","subject","level"]]
    udemy.columns = ["title","org","level"]

    data = pd.concat([coursera, udemy])

    data["content"] = (
        data["title"] + " " +
        data["org"] + " " +
        data["level"] + " " +
        data["title"]
    )

    data["content"] = data["content"].apply(clean_text)

    # create clickable course link
    data["url"] = data["title"].apply(
        lambda x: "https://www.coursera.org/search?query=" + urllib.parse.quote(x)
    )

    return data