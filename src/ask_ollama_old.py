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

    def get_llm_response_generate_endpoint(self, model_name, prompt, image_path=None):
        """Get response from model for a prompt and optional image"""
        url = f'{self.base_url}/generate'

        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        if image_path:
            base64_image = self.encode_image(image_path)
            if base64_image:
                data["images"] = [base64_image]
            else:
                return f"Error: Could not process image {image_path}"

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error getting response: {str(e)}"

    def get_llm_response_chat_endpoint(self, model_name, prompt, image_path=None):
        """Get response from model for a prompt and optional image with a fresh context"""
        url = f'{self.base_url}/chat'  # Using chat endpoint instead of generate

        # Start with an empty context for each new request
        messages = [{
            "role": "user",
            "content": prompt
        }]

        data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "context": None  # Explicitly set context to None to ensure a fresh start
        }

        if image_path:
            base64_image = self.encode_image(image_path)
            if base64_image:
                data["images"] = [base64_image]
            else:
                return f"Error: Could not process image {image_path}"

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            return f"Error getting response: {str(e)}"

    def create_output_directory(self):
        """Create timestamped output directory"""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_dir = f"results_{timestamp}"
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

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

                # Process each question for this image
                image_results = {
                    'image_path': image_path,
                    'responses': []
                }

                for question in tqdm(questions, desc="Asking questions"):
                    response = self.get_llm_response_chat_endpoint(
                        model_name,
                        question['text'],
                        image_path
                    )

                    image_results['responses'].append({
                        'model_name': model_name,
                        'question': question['text'],
                        'response': response
                    })

                    # Small delay to avoid overwhelming the API
                    time.sleep(0.5)

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