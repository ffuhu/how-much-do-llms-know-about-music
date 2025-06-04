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
                "content": question,
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
                    'question': question,
                    'response': f"Error getting response: {str(e)}"
                })
                print(f"There was an error asking question:\n\n"
                      f"{question}\n\n"
                      f"Error:\n\n"
                      f"{e}")

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

        return responses

    def process_all(self, models, questions, images):
        """Process all models, questions, and images"""

        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        # Create output directory
        output_dir = self.create_output_directory()

        # Process each model
        for model_name in models:

            # Create model-specific results file
            results_file = os.path.join(output_dir, f"results_{result_affix}{model_name}.json")
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
            for image_path in images:

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


result_affix = 'q9_'

def main():
    processor = OllamaProcessor()

    # Configuration files
    models = [
        # "gemma3:4b-it-qat",
        # "qwen2.5vl:7b",
        # "llava-llama3:8b",
        # "llava-phi3:3.8b",  # doesn't work as expected, doesn't "see" the image
        # "mistral-small3.1:24b",  # doesn't work
        # new models
        "minicpm-v:8b",
        "llava:7b",
        "bakllava:7b",
        "granite3.2-vision:2b",
        "llama3.2:3b",
    ]

    questions = [
        "Task Introduction: Hello! I need your expertise for a specific task. You are an expert in Optical Music Recognition (OMR) within the field of Computer Vision. The task involves analyzing an image which might or might not contain music notation within it. The music notation present might be monophonic, polyphonic or vocal. If present, it will be a music excerpt. Please answer any question I ask you based only on that excerpt. You can only provide information based on what is on the image, no matter how much you might know about it from other sources."
        ""
        "Task Details: In the monophonic case, the music excerpt consists of only the musical staff with a single melodic line, without any lyrics, performance directions, or other textual elements. In the polyphonic case, there will be more than one melodic line. In the vocal case, you might also find the lyrics of the song. Your task is to analyze the musical notation, when present, and use the melody's structure, rhythm, and characteristics to answer some questions about the piece. If you are uncertain about an answer, you should clearly indicate so and explain why."
        ""
        "Looking at the visual information on the image, can you find any kind of music representation in it?",
        "Please answer only with “Yes”, if there is any kind of music representation on the image, or “No”, if there is not.",
    ]

    images = [
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_0.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_55.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_109.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_118.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_139.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_181.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_247.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_262.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_282.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/a_378.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_0.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_15.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_38.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_89.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_183.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_225.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_239.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_266.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_326.jpg',
        '/home/felix/Scratch/01_music_llms/how-much-do-llms-know-about-music/src/images/q1/b_389.jpg',
    ]

    print("🚀 Starting processing...")
    processor.process_all(models, questions, images)
    print("\n✨ Processing completed!")


if __name__ == "__main__":
    main()

# Task Introduction: Hello! I need your expertise for a specific task. You are an expert in Optical Music Recognition (OMR) within the field of Computer V
# ision. The task involves analyzing an image which might or might not contain music notation within it. The music notation present might be monophonic, p
# olyphonic or vocal. If present, it will be a music excerpt. Please answer any question I ask you based only on that excerpt. You can only provide inform
# ation based on what is on the image, no matter how much you might know about it from other sources.
#
# Task Details: In the monophonic case, the music excerpt consists of only the musical staff with a single melodic line, without any lyrics, performance d
# irections, or other textual elements. In the polyphonic case, there will be more than one melodic line. In the vocal case, you might also find the lyric
# s of the song. Your task is to analyze the musical notation, when present, and use the melody's structure, rhythm, and characteristics to answer some qu
# estions about the piece. If you are uncertain about an answer, you should clearly indicate so and explain why.
#
# Looking at the visual information on the image, can you find any kind of music representation in it? Answer only with “Yes”, if there is any kind of music representation on the image, or “No”, if there is not.
# This is the image: /home/felix/Scratch/01_music_llms/how much do llm know about music/src/images/q1/a_0.jpg