import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime
import os

from anthropic import Anthropic
from PIL import Image

class ClaudeProcessor:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = "Eres un asistente útil que responde preguntas sobre imágenes y texto."

    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",  # Cambia si usas PNG, etc.
                        "data": base64_image
                    }
                }
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None

    def create_output_directory(self):
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_image_questions(self, model_name, questions, image_path):
        responses = []
        conversation_history = []

        try:
            image_content = self.encode_image(image_path)
            if not image_content:
                return f"Error: Could not load image {image_path}"

            for question in questions:
                try:
                    # Para la primera pregunta, incluye la imagen
                    print(f"Asking question {question['id']} ({len(questions)} questions)")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                image_content,
                                {"type": "text", "text": question['text']}
                            ]
                        }
                    ]

                    response = self.client.messages.create(
                        model=model_name,
                        max_tokens=2000,
                        messages=messages,
                        system=self.system_prompt
                    )

                    assistant_response = response.content[0].text

                    # Actualiza el historial de la conversación
                    conversation_history.extend([
                        {"role": "user", "content": question['text']},
                        {"role": "assistant", "content": assistant_response}
                    ])

                    responses.append({
                        'question': question['text'],
                        'response': assistant_response
                    })

                except Exception as e:
                    responses.append({
                        'question': question['text'],
                        'response': f"Error getting response: {str(e)}"
                    })
                    print(f"Error processing question:\n{question['text']}\nError:\n{e}")

                time.sleep(5)  # Espera para evitar rate limits

        except Exception as e:
            print(f"Error in process_image_questions: {str(e)}")
            return []

        return responses

    def process_all(self, models, questions_file, images_file):
        questions = self.load_json_file(questions_file)
        images = self.load_json_file(images_file)

        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        output_dir = self.create_output_directory()

        for model_name in models:
            results_file = os.path.join(output_dir, f"results_{model_name.replace(os.sep, '-')}.json")
            results = []

            if os.path.exists(results_file):
                print(f"{model_name} already evaluated, results in {results_file}. Skipping...")
                continue

            print(f"\n📝 Processing model: {model_name}")

            self.system_prompt = questions.pop(0)['text']

            for image in images:
                image_path = image['path']
                if not Path(image_path).exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue
                print(f"Working with image: {image_path}")

                image_responses = self.process_image_questions(
                    model_name,
                    questions,
                    image_path
                )

                image_results = {
                    'image_path': image_path,
                    'responses': image_responses
                }
                results.append(image_results)

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Completed processing for {model_name}")
            print(f"📁 Results saved to: {results_file}")

def main():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return

    processor = ClaudeProcessor(api_key)

    models = [
        #"claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-20250219",
        # Puedes agregar otros modelos aquí
    ]
    questions_file = "questions.json"
    images_file = "images.json"

    for file in [questions_file, images_file]:
        if not Path(file).exists():
            print(f"❌ Configuration file not found: {file}")
            return

    print("🚀 Starting processing...")
    processor.process_all(models, questions_file, images_file)
    print("\n✨ Processing completed!")

if __name__ == "__main__":
    main()