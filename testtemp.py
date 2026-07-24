import json
with open("new_dataset_final.json", "r", encoding="utf-8") as f:
    result = json.load(f)

print(len(result))  