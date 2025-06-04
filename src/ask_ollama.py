import requests
import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime
import os


class OllamaProcessor:
    def __init__(self):
        self.base_url = 'http://localhost:11434/api'
        self.headers = {'Content-Type': 'application/json'}

    def load_json_file(self, file_path):
        """Load and validate JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None

    def check_model_exists(self, model_name):
        """Check if a model is already available locally"""
        url = 'http://localhost:11434/api/tags'
        try:
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get('models', [])
            return any(model['name'] == model_name for model in models)
        except:
            return False

    def pull_model(self, model_name):
        """Pull a specific model"""
        url = f'{self.base_url}/pull'
        data = {'name': model_name}

        print(f"\n🚀 Pulling model: {model_name}")

        try:
            with tqdm(desc=f"Pulling {model_name}", unit="MB") as pbar:
                response = requests.post(url, headers=self.headers, json=data, stream=True)
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if 'error' in json_response:
                            print(f"\n❌ Error pulling {model_name}: {json_response['error']}")
                            return False
                        pbar.update(1)

                print(f"✅ Successfully pulled {model_name}")
                return True

        except Exception as e:
            print(f"\n❌ Error pulling {model_name}: {str(e)}")
            return False

    def create_output_directory(self):
        """Create timestamped output directory"""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_dir = f"results_{timestamp}"
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_image_questions(self, model_name, questions, image_path):
        """Process all questions for a single image, maintaining context within the image"""
        # Initialize an empty conversation history for this image
        conversation_history = []
        responses = []

        for question_id, question in enumerate(questions):
            url = f'{self.base_url}/chat'

            # Build messages array with conversation history
            messages = conversation_history + [{
                "role": "user",
                "content": question['text']
            }]

            data = {
                "model": model_name,
                "messages": messages,
                "stream": False
            }

            # Add image only to the first question
            if question_id == 0:  # If this is the first question
                base64_image = self.encode_image(image_path)
                if base64_image:
                    data["messages"][0]["images"] = [base64_image]
                else:
                    return f"Error: Could not process image {image_path}"

            try:
                response = requests.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                assistant_response = response.json()['message']['content'].strip()

                print("Question:\n\n", question, "\n\n")
                print("Response:\n\n", assistant_response, "\n\n")

                # Add the question and response to conversation history
                conversation_history.append({
                    "role": "user",
                    "content": question
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })

                # Store the response
                responses.append({
                    'question': question,
                    'response': assistant_response
                })

            except Exception as e:
                responses.append({
                    'question': question['text'],
                    'response': f"Error getting response: {str(e)}"
                })
                print(f"There was an error asking question:\n\n"
                      f"{question['text']}\n\n"
                      f"Error:\n\n"
                      f"{e}")

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

        return responses

    def process_all(self, models_file, questions_file, images_file):
        """Process all models, questions, and images"""
        # Load configuration files
        models = self.load_json_file(models_file)
        questions = self.load_json_file(questions_file)
        images = self.load_json_file(images_file)

        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        # Create output directory
        output_dir = self.create_output_directory()

        # Process each model
        for model in models:
            model_name = model['name']

            # Create model-specific results file
            results_file = os.path.join(output_dir, f"results_{model_name}.json")
            results = []

            if os.path.exists(results_file):
                print(f"{model_name} already evaluated, results in {results_file}. Skipping...")
                continue

            # Pull model if specified
            if not self.check_model_exists(model_name):
                status_pulled = self.pull_model(model_name)
                if not status_pulled:
                    print(f"Model {model_name} is not available")
                    continue

            print(f"\n📝 Processing model: {model_name}")

            # Process each image
            for image in images:
                image_path = image['path']
                if not Path(image_path).exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue
                print(f"Working with image: {image_path}")

                # Process all questions for this image while maintaining context
                image_responses = self.process_image_questions(
                    model_name,
                    questions,
                    image_path
                )

                # Store results for this image
                image_results = {
                    'image_path': image_path,
                    'responses': image_responses
                }
                results.append(image_results)

                # Save results after each image (in case of interruption)
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Completed processing for {model_name}")
            print(f"📁 Results saved to: {results_file}")


def main():
    processor = OllamaProcessor()

    # Configuration files
    models_file = "models.json"
    questions_file = "questions.json"
    images_file = "images.json"

    # Check if configuration files exist
    for file in [models_file, questions_file, images_file]:
        if not Path(file).exists():
            print(f"❌ Configuration file not found: {file}")
            return

    print("🚀 Starting processing...")
    processor.process_all(models_file, questions_file, images_file)
    print("\n✨ Processing completed!")


if __name__ == "__main__":
    main()
