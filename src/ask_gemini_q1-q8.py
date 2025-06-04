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

            for question_id, question in enumerate(questions):
                try:
                    # For the first question, include the image
                    if question_id == 0:
                        response = chat.send_message([question, image])
                    else:
                        # For subsequent questions, use only text but maintain context
                        response = chat.send_message(question)

                    print("Question:\n\n", question, "\n\n")
                    print("Response:\n\n", response.text, "\n\n")

                    responses.append({
                        'question': question,
                        'response': response.text
                    })

                except Exception as e:
                    responses.append({
                        'question': question,
                        'response': f"Error getting response: {str(e)}"
                    })
                    print(f"Error processing question:\n{question}\nError:\n{e}")

                # Small delay between requests
                time.sleep(15)

        except Exception as e:
            print(f"Error in process_image_questions: {str(e)}")
            return []

        return responses

    def process_all(self, models, questions, images):
        """Process all models, questions, and images"""

        if not all([models, questions, images]):
            print("❌ Error loading configuration files")
            return

        # Create output directory
        output_dir = self.create_output_directory()

        # Process each model
        for model in models:
            model_name = model

            # Create model-specific results file
            results_file = os.path.join(output_dir, f"results_{result_affix}{model_name.replace(os.sep, '-')}.json")
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

                # Save results after each image
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Completed processing for {model_name}")
            print(f"📁 Results saved to: {results_file}")

result_affix = 'q1-q8_'
result_affix = 'q4_'

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

    questions = [
        "Task Introduction: Hello! I need your expertise for a specific task. You are an expert in "
        "Optical Music Recognition (OMR) within the field of Computer Vision. The task involves "
        "analyzing an image which might or might not contain music notation within it. The music "
        "notation present might be monophonic, polyphonic or vocal. If present, it will be a music "
        "excerpt. Please answer any question I ask you based only on that excerpt. You can only "
        "provide information based on what is on the image, no matter how much you might know about "
        "it from other sources."
        ""
        "Task Details: In the monophonic case, the music excerpt consists of only the musical staff "
        "with a single melodic line, without any lyrics, performance directions, or other textual "
        "elements. In the polyphonic case, there will be more than one melodic line. In the vocal "
        "case, you might also find the lyrics of the song. Your task is to analyze the musical "
        "notation, when present, and use the melody's structure, rhythm, and characteristics to "
        "answer some questions about the piece. If you are uncertain about an answer, you should "
        "clearly indicate so and explain why."
        ""
        # Q1
        "Based solely on the musical notation provided, can you determine which musical work this "
        "excerpt belongs to? Please use the melody’s rhythmic patterns, note sequences, and any "
        "other musical clues to make your deduction.",
        # Q2
        "What type of musical notation is used in this excerpt? Choose between: modern notation, "
        "tablature, mensural notation, Korean notation, “other notation” or “no music notation "
        "present”. Answer only with the name of the musical notation used in the excerpt.",
        # Q3a
        "Please transcribe the contents of the given music excerpt in the Kern formats. For your "
        "information, Kern forma is a text-based format used in the Humdrum toolkit for encoding "
        "musical scores. Ensure the excerpt's musical notation is accurately encoded in the Kern "
        "syntax, capturing the pitch, rhythm, and any relevant musical details.\n"
        "Make sure you are familiar with the Kern format when performing the transcriptions. "
        "Provide the transcript only for what is visible in the image, even if you have more "
        "knowledge."
        "If you do not know how to transcribe music, please say so.",
        # Q3b
        "Please transcribe the contents of the given music excerpt in the MusicXML format. For "
        "your information, MusicXML format is a structured, machine-readable format for Western "
        "music notation. Please transcribe the musical notation from the excerpt accurately in "
        "this format.\n"
        "Make sure you are familiar with the MusicXML format when performing the transcriptions. "
        "Provide the transcript only for what is visible in the image, even if you have more "
        "knowledge."
        "If you do not know how to transcribe music, please say so.",
        # Q3c
        "Please transcribe the contents of the given music excerpt in the ABC format. For your "
        "information, ABC format is a text-based format for Western music notation. Please "
        "transcribe the musical notation from the excerpt accurately in this format.\n"
        "Make sure you are familiar with the ABC format when performing the transcriptions. "
        "Provide the transcript only for what is visible in the image, even if you have more "
        "knowledge."
        "If you do not know how to transcribe music, please say so.",
        # Q4 --> REPEAT
        "Can you determine the tonality (key) of the given music excerpt? Please explain the "
        "steps and reasoning behind how you arrived at this conclusion.\n"
        "When determining the tonality, consider the following factors:\n"
        "- Key signature: Whether there are any sharps or flats.\n"
        "- Accidentals: Notes altered by sharps, flats, or naturals outside of the key signature.\n"
        "- Melodic structure: The shape and direction of the melody, including important pitches "
        "and phrase endings.\n"
        "- Harmonic implications: Some melodies imply certain harmonic progressions or tonic-dominant "
        "relationships.\n"
        "Provide a clear explanation of your process, detailing how each of these elements contributes "
        "to your conclusion about the tonality.",
        # Q5
        "Based on the musical notation present in the image, what instrument do you think it is "
        "intended for? Your task is to analyze the musical notation and use the melody's structure, "
        "rhythm, and characteristics to deduce the most likely instrument the musical work was "
        "written for.",
        # Q6
        "Based on the musical notation present in the image, what is the time signature?",
        # Q7
        "In the given music excerpt, are the following symbols present?\n"
        "- Quarter note\n"
        "- Whole note\n"
        "- Crescendo indicator\n"
        "- C major chord\n"
        "- Quarter rest\n",
        # Q8a
        "In the given music excerpt, is there information about the speed at which the piece should "
        "be played? (e.g., tempo markings like Allegro, Andante, or a BPM number).\n",
        # Q8b
        "In the given music excerpt, are there any articulation marks indicating how the piece, or "
        "a segment, should be played? (such as ties, legato slurs, dots for staccato, or other "
        "articulation marks)\n",
        # Q8c
        "In the given music excerpt, what clefs, if any, are present in this image of the staff?",
    ]

    questions = [questions[0], questions[5]]


    images = [
        "./images/1_1.png",
        "./images/1_2.png",
        "./images/1_3.png",
        "./images/1_4.png",
        "./images/1_5.png",
        "./images/2_1.png",
        "./images/2_1_ins.png",
        "./images/2_2.png",
        "./images/2_3.png",
        "./images/2_3_ins.png",
        "./images/2_4.png",
        "./images/2_5.png",
        "./images/2_5_ins.png",
    ]

    print("🚀 Starting processing...")
    processor.process_all(models, questions, images)
    print("\n✨ Processing completed!")

if __name__ == "__main__":
    main()