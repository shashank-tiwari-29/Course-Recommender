def classify_course(title):

    title = title.lower()

    if "introduction" in title or "basics" in title:
        return "Remember"

    elif "understanding" in title or "concepts" in title:
        return "Understand"

    elif "project" in title or "application" in title:
        return "Apply"

    elif "analysis" in title:
        return "Analyze"

    elif "evaluation" in title:
        return "Evaluate"

    else:
        return "Create"