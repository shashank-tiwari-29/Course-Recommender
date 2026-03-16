import numpy as np
from hybrid_recommender import recommend
from preprocess import load_dataset
from tabulate import tabulate


data = load_dataset()

test_courses = data.sample(50)


def evaluate(top_n):

    precision_list = []
    recall_list = []
    f1_list = []

    for _, row in test_courses.iterrows():

        course = row["title"]
        topic = row["org"]

        try:
            recs = recommend(course, top_n)

            relevant_count = 0

            for r in recs:

                rec_topic = data[data["title"] == r]["org"].values

                if len(rec_topic) > 0 and rec_topic[0] == topic:
                    relevant_count += 1

            precision = relevant_count / top_n
            recall = relevant_count / top_n

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        except:
            pass

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)


p3, r3, f3 = evaluate(3)
p5, r5, f5 = evaluate(5)
p8, r8, f8 = evaluate(8)


table = [[
"Hybrid MOOC Recommender",
round(p3,2), round(r3,2), round(f3,2),
round(p5,2), round(r5,2), round(f5,2),
round(p8,2), round(r8,2), round(f8,2)
]]


headers = [
"Model",
"Top3 Precision","Top3 Recall","Top3 F1",
"Top5 Precision","Top5 Recall","Top5 F1",
"Top8 Precision","Top8 Recall","Top8 F1"
]


print("\nEvaluation Results\n")

print(tabulate(table, headers=headers, tablefmt="grid"))