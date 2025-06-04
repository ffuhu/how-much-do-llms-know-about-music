import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime
import os

from openai import OpenAI
from PIL import Image

class ChatGPTProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = "You are a helpful assistant that answers questions about images and text."

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
                return f"data:image/jpeg;base64,{base64_image}"
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

            for question_id, question in enumerate(questions):
                try:

                    print(f"Asking question {question_id} ({len(questions)} questions)")

                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_content}
                                },
                                {"type": "text", "text": question}
                            ]
                        }
                    ]

                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": m["role"],
                                "content": m["content"]
                            } for m in messages
                        ],
                        max_completion_tokens=2000
                    )

                    assistant_response = response.choices[0].message.content

                    print("Question:\n\n", question, "\n\n")
                    print("Response:\n\n", assistant_response, "\n\n")

                    conversation_history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": assistant_response}
                    ])

                    responses.append({
                        'question': question,
                        'response': assistant_response
                    })

                except Exception as e:
                    responses.append({
                        'question': question,
                        'response': f"Error getting response: {str(e)}"
                    })
                    print(f"Error processing question:\n{question}\nError:\n{e}")

                time.sleep(1)  # Rate limit prevention

        except Exception as e:
            print(f"Error in process_image_questions: {str(e)}")
            return []

        return responses

    def process_all(self, models, questions, images):


        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        output_dir = self.create_output_directory()

        for model_name in models:
            results_file = os.path.join(output_dir, f"results_{result_affix}{model_name.replace(os.sep, '-')}.json")
            results = []

            if os.path.exists(results_file):
                print(f"{model_name} already evaluated, results in {results_file}. Skipping...")
                continue

            print(f"\n📝 Processing model: {model_name}")

            self.system_prompt = questions.pop(0)

            for image_path in images:

                if not Path(image_path).exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue
                print(f"Working with image: {image_path}")

                image_responses = self.process_image_questions(
                    model_name,
                    questions.copy(),  # Make a copy to preserve original questions
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


result_affix = "q9_"

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    processor = ChatGPTProcessor(api_key)

    models = [
        "o4-mini-2025-04-16",
        #"o3-2025-04-16",  # For image analysis
        # You can add other GPT-4 or GPT-3.5 models here
    ]

    questions = [
        # system prompt
        "Task Introduction: Hello! I need your expertise for a specific task. You are an expert in Optical Music Recognition (OMR) within the field of Computer Vision. The task involves analyzing an image which might or might not contain music notation within it. The music notation present might be monophonic, polyphonic or vocal. If present, it will be a music excerpt. Please answer any question I ask you based only on that excerpt. You can only provide information based on what is on the image, no matter how much you might know about it from other sources."
        ""
        "Task Details: In the monophonic case, the music excerpt consists of only the musical staff with a single melodic line, without any lyrics, performance directions, or other textual elements. In the polyphonic case, there will be more than one melodic line. In the vocal case, you might also find the lyrics of the song. Your task is to analyze the musical notation, when present, and use the melody's structure, rhythm, and characteristics to answer some questions about the piece. If you are uncertain about an answer, you should clearly indicate so and explain why.",
        # questions
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