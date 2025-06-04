import streamlit as st
import json
import os

# Set page title
st.set_page_config(page_title="Question Evaluator", layout="wide")

# Initialize session state for storing evaluations
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = {}


# Function to load JSON data
def load_data(file_path=None):
    if file_path:
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        # Use the sample data provided in the question
        return st.session_state.get('sample_data', [])


# Function to save evaluations
def save_evaluations():
    with open('evaluations.json', 'w') as f:
        json.dump(st.session_state.evaluations, f, indent=2)
    st.success("Evaluations saved to evaluations.json")


# Main app
st.title("Question-Answer Evaluator")

# Sidebar for file upload and navigation
with st.sidebar:
    st.header("Controls")

    # File upload
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_data.json", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the data
        data = load_data("temp_data.json")
        st.session_state.sample_data = data
    else:
        # Use the sample data if available
        data = st.session_state.get('sample_data', [])

    # Navigation
    if data:
        st.subheader("Navigation")
        image_indices = list(range(len(data)))
        selected_image = st.selectbox("Select Image", image_indices, format_func=lambda x: f"Image {x + 1}")

        if selected_image is not None:
            image_data = data[selected_image]
            question_indices = list(range(len(image_data['responses'])))
            selected_question = st.selectbox("Select Question", question_indices,
                                             format_func=lambda x: f"Question {x + 1}")

            # Save button
            if st.button("Save All Evaluations"):
                save_evaluations()

# Main content area
if 'sample_data' in st.session_state and selected_image is not None and selected_question is not None:
    image_data = st.session_state.sample_data[selected_image]
    question_data = image_data['responses'][selected_question]

    # Display image path
    st.subheader("Image")
    st.info(image_data['image_path'])

    # Display question and answer
    st.subheader("Question")
    st.write(question_data['question'])

    st.subheader("Answer")
    st.write(question_data['response'])

    # Evaluation buttons
    st.subheader("Evaluation")

    # Create a unique key for this question
    eval_key = f"{image_data['image_path']}_{selected_question}"

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Correct", key=f"correct_{eval_key}"):
            st.session_state.evaluations[eval_key] = "correct"
            st.success("Marked as Correct")

    with col2:
        if st.button("So-So", key=f"soso_{eval_key}"):
            st.session_state.evaluations[eval_key] = "soso"
            st.info("Marked as So-So")

    with col3:
        if st.button("Wrong", key=f"wrong_{eval_key}"):
            st.session_state.evaluations[eval_key] = "wrong"
            st.error("Marked as Wrong")

    # Show current evaluation
    if eval_key in st.session_state.evaluations:
        st.write(f"Current evaluation: **{st.session_state.evaluations[eval_key].upper()}**")
else:
    st.info("Please upload a JSON file or use the sample data to begin evaluation.")

# Initialize with sample data if available
if 'sample_data' not in st.session_state and len(data) > 0:
    st.session_state.sample_data = data

# Display current evaluations count
if 'evaluations' in st.session_state and st.session_state.evaluations:
    st.sidebar.subheader("Evaluation Stats")
    evals = st.session_state.evaluations
    correct_count = sum(1 for v in evals.values() if v == "correct")
    soso_count = sum(1 for v in evals.values() if v == "soso")
    wrong_count = sum(1 for v in evals.values() if v == "wrong")

    st.sidebar.write(f"Correct: {correct_count}")
    st.sidebar.write(f"So-So: {soso_count}")
    st.sidebar.write(f"Wrong: {wrong_count}")
    st.sidebar.write(f"Total: {len(evals)}")

print("App created successfully!")