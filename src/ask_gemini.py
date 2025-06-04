import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime
import os
import google.generativeai as genai
from PIL import Image

class GeminiProcessor:
    def __init__(self, api_key):
        # Configure the Gemini API with your key
        genai.configure(api_key=api_key)
        self.model = None

    def load_json_file(self, file_path):
        """Load and validate JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def load_image(self, image_path):
        """Load image using PIL"""
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def check_model_exists(self, model_name):
        """Check if model is available in Gemini"""
        try:
            available_models = [m.name for m in genai.list_models()]
            return model_name in available_models
        except Exception as e:
            print(f"Error checking model availability: {str(e)}")
            return False

    def create_output_directory(self):
        """Create output directory"""
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_image_questions(self, model_name, questions, image_path):
        """Process all questions for a single image using Gemini's multimodal capabilities"""
        responses = []

        try:
            # Load the image
            image = self.load_image(image_path)
            if not image:
                return f"Error: Could not load image {image_path}"

            # Initialize the model
            self.model = genai.GenerativeModel(model_name)

            # Start a new chat for maintaining context
            chat = self.model.start_chat(history=[])

            for question in questions:
                try:
                    # For the first question, include the image
                    if question["id"] == 1:
                        response = chat.send_message([question['text'], image])
                    else:
                        # For subsequent questions, use only text but maintain context
                        response = chat.send_message(question['text'])

                    responses.append({
                        'question': question['text'],
                        'response': response.text
                    })

                except Exception as e:
                    responses.append({
                        'question': question['text'],
                        'response': f"Error getting response: {str(e)}"
                    })
                    print(f"Error processing question:\n{question['text']}\nError:\n{e}")

                # Small delay between requests
                time.sleep(15)

        except Exception as e:
            print(f"Error in process_image_questions: {str(e)}")
            return []

        return responses

    def process_all(self, models, questions_file, images_file):
        """Process all models, questions, and images"""
        # Load configuration files
        # models = self.load_json_file(models_file)
        questions = self.load_json_file(questions_file)
        images = self.load_json_file(images_file)

        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        # Create output directory
        output_dir = self.create_output_directory()

        # Process each model
        for model in models:
            model_name = model

            # Create model-specific results file
            results_file = os.path.join(output_dir, f"results_{model_name.replace(os.sep, '-')}.json")
            results = []

            if os.path.exists(results_file):
                print(f"{model_name} already evaluated, results in {results_file}. Skipping...")
                continue

            # Check if model is available
            if not self.check_model_exists(model_name):
                print(f"Model {model_name} is not available in Gemini")
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

                # Save results after each image
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Completed processing for {model_name}")
            print(f"📁 Results saved to: {results_file}")

def main():
    # Get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return

    processor = GeminiProcessor(api_key)

    # Configuration files
    models = [
        # "models/gemini-2.0-flash-lite",
        "models/gemini-2.5-flash-preview-04-17",
        # "gemini-2.5-pro-preview-05-06"  # not freely available
    ]
    questions_file = "questions.json"
    images_file = "images.json"

    # Check if configuration files exist
    for file in [questions_file, images_file]:
        if not Path(file).exists():
            print(f"❌ Configuration file not found: {file}")
            return

    print("🚀 Starting processing...")
    processor.process_all(models, questions_file, images_file)
    print("\n✨ Processing completed!")

if __name__ == "__main__":
    main()