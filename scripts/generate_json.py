import json

def write_json(fields, recognized_texts, output_path="output/scorecard.json"):
    """Generate a JSON file from detected fields and recognized text."""
    result = {}
    for field, text in zip(fields, recognized_texts):
        result[field["label"]] = text

    with open(output_path, "w") as json_file:
        json.dump(result, json_file, indent=4)

if __name__ == "__main__":
    fields = [{"label": "PlayerName", "bbox": [0, 0, 100, 50]}]
    recognized_texts = ["John Doe"]
    write_json(fields, recognized_texts)
