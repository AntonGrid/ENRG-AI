def score(file, query):

    score = 0

    path = file["path"].lower()
    name = file["name"].lower()

    if path.endswith(".rs"):
        score += 100

    if "/program" in path:
        score += 200

    if query in name:
        score += 300

    if query in path:
        score += 150

    if query in file["content"]:
        score += 20

    return score
