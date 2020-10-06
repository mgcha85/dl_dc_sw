import json

contents = {'path': {'root': 'D:/Independent_research/KR'}}
with open("parameter.json", "w") as f:
    json.dump(contents, f)
