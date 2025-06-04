import os
import json
import glob


def check_empty_responses(folder_path):
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        return "No JSON files found in the specified folder."

    empty_responses = []

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check each image entry
            for image_entry in data:
                image_path = image_entry.get("image_path", "Unknown image")

                # Check each response for the image
                for i, response_obj in enumerate(image_entry.get("responses", [])):
                    question = response_obj.get("question", f"Question {i + 1}")
                    response = response_obj.get("response", "")

                    # Check if response is empty
                    if not response or response.strip() == "":
                        empty_responses.append({
                            "file": os.path.basename(json_file),
                            "image": image_path,
                            "question": question
                        })
        except Exception as e:
            empty_responses.append({
                "file": os.path.basename(json_file),
                "error": str(e)
            })

    return empty_responses


# Create a report
def generate_report(empty_responses, output_file="empty_responses_report.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        if not empty_responses:
            f.write("No empty responses found in any JSON file.\n")
            return "No empty responses found in any JSON file."

        f.write(f"Found {len(empty_responses)} empty responses:\n\n")

        for item in empty_responses:
            if "error" in item:
                f.write(f"Error processing file {item['file']}: {item['error']}\n")
            else:
                f.write(f"File: {item['file']}\n")
                f.write(f"Image: {item['image']}\n")
                f.write(f"Question: {item['question']}\n")
                f.write("-" * 50 + "\n")

        return f"Report generated with {len(empty_responses)} empty responses found."


# Example usage - replace with the actual folder path
folder_path = "./results/"
empty_responses = check_empty_responses(folder_path)

if isinstance(empty_responses, str):
    print(empty_responses)
else:
    result = generate_report(empty_responses)
    print(result)
    print(f"Detailed report saved to 'empty_responses_report.txt'")