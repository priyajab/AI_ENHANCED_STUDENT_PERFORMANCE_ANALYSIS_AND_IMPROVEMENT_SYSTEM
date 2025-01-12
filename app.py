from flask import Flask, render_template, request, send_from_directory, session, jsonify
from PIL import Image
import pytesseract
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer  # Corrected import
import Levenshtein
import nltk
nltk.download('punkt')
print(nltk.data.path)
nltk.data.path.append("C:\\Users\\shafe\\AppData\\Local\\Programs\\Python\\Python311\\nltk_data")
from googletrans import Translator
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from rake_nltk import Rake
import requests
import rake_nltk
nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = b'\x86YIl%i\xd5\xc0\x8e\xe8T\x8d\xf75&\x98-k+1\xaf\x17\xbcB'  # Set a secret key for session management

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model to evaluation mode
model.eval()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()  # Corrected initialization

# Load data from CSV
students_data = pd.read_csv('student_data.csv')

feedback_columns = ['student_id', 'feedback_date', 'feedback', 'rating']
feedback_data = pd.DataFrame(columns=feedback_columns)

FEEDBACK_FILE = 'feedback.xlsx'


# Path to the Tesseract executable (update this with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the grading data from Excel sheet
grading_data = pd.read_csv('your_excel_sheet.csv')

# Initialize the translator
translator = Translator()

# Load educational resources
educational_resources = pd.read_csv('educational_resources.csv')

# Define a dictionary of rules for recommendation generation
recommendation_rules = {
    'data science': [
        "Consider taking additional data science courses or seeking tutoring to improve your understanding of the subject.",
        "Check out the following resources: {}".format(', '.join(educational_resources[educational_resources['subject'] == 'data science']['resource'].tolist()))
    ],
    'struggling with machine learning': [
        "Practice more machine learning exercises and projects to solidify your understanding.",
        "Refer to these helpful resources: {}".format(', '.join(educational_resources[educational_resources['subject'] == 'machine learning']['resource'].tolist()))
    ],
    # Add more rules based on common feedback patterns
}
def generate_recommendations(feedback_text):
    # Analyze sentiment of feedback text
    sentiment_score = sia.polarity_scores(feedback_text)['compound']

    recommendations = []

    # Check if the feedback text matches any predefined rules
    for pattern, pattern_recommendations in recommendation_rules.items():
        if pattern in feedback_text.lower():
            recommendations.extend(pattern_recommendations)

    # Extract resources based on keywords from educational resources CSV
    for index, row in educational_resources.iterrows():
        if row['subject'].lower() in feedback_text.lower():
            recommendations.append(row['resource'])
            recommendations.append(row['resource_link'])  # Assuming 'resource_link' is the column containing URLs
            break  # Exit loop after finding the first subject match

    # Provide additional recommendations based on sentiment
    if sentiment_score < 0:
        recommendations.append("It seems like you're facing difficulties. Don't hesitate to reach out for help.")

    return recommendations


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    email = request.form.get('email')
    password = request.form.get('password')

    user = students_data[students_data['mail'] == email]

    if not user.empty and str(user['student_id'].iloc[0]) == password:
        user_info = user.iloc[0]  # Assuming email is unique, taking the first match
        session['student_id'] = str(user_info['student_id'])  # Store student ID in session
        return render_template('dashboard.html', user=user_info)
    else:
        error = 'Invalid email or password'
        return render_template('login.html', error=error)

@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    if 'image' not in request.files:
        return render_template('upload_result.html', error='No image provided')

    image = request.files['image']

    if image.filename == '':
        return render_template('upload_result.html', error='No image selected')

    if image:
        # Save the image to the upload folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Perform analysis on the image (including grading)
        extracted_text, grade = perform_ocr_and_grade(image_path)

        return render_template('upload_result.html', image_path=image_path, extracted_text=extracted_text, grade=grade)

    return render_template('upload_result.html', error='Error processing the image')

def perform_ocr_and_grade(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)

    # Use Tesseract OCR to extract text
    extracted_text = pytesseract.image_to_string(img)

    # Get the grade based on extracted text
    grade = get_grade_from_extracted_text(extracted_text)

    return extracted_text, grade

def get_grade_from_extracted_text(extracted_text):
    similarity_scores = grading_data['answer'].apply(lambda ans: Levenshtein.ratio(extracted_text, ans))

    # Get the maximum similarity score
    max_similarity = similarity_scores.max()

    # Map the similarity score to the grade scale (adjust the mapping as needed)
    grade = max_similarity * 100

    return grade

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        recommendations = generate_recommendations(feedback_text)
        return jsonify({"recommendations": recommendations})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        # Get feedback data from the form
        feedback_text = request.form.get('feedback')

        # Get the student ID from the session
        student_id = session.get('student_id')

        # Check if student_id is not None
        if student_id is not None:
            # Validate if the student ID exists in your data
            user = students_data[students_data['student_id'] == int(student_id)]

            if not user.empty:
                # Convert int64 to native Python integer
                student_id = int(student_id)

                # Perform sentimental analysis
                sentiment_score = sia.polarity_scores(feedback_text)['compound']
                
                # Map sentiment score to the rating scale
                rating = get_sentiment_rating(sentiment_score)

                # Translate feedback to English
                translated_feedback = translate_feedback(feedback_text)

                # Append feedback and rating to the feedback DataFrame
                feedback_data.loc[len(feedback_data)] = [student_id, pd.Timestamp.now(), translated_feedback, rating]

                # Save the updated feedback DataFrame to Excel
                feedback_data.to_excel(FEEDBACK_FILE, index=False)

                return render_template('feedback_confirmation.html', student_id=student_id, rating=rating)
            else:
                error = 'Invalid Student ID'
                return render_template('error_page.html', error=error)
        else:
            error = 'Invalid Student ID'
            return render_template('error_page.html', error=error)

    return render_template('error_page.html', error='Error submitting feedback')

def get_sentiment_rating(compound_score):
    if compound_score >= 0.5:
        return 5.0  # Very positive
    elif 0.2 <= compound_score < 0.5:
        return 3.5  # Positive
    elif -0.2 <= compound_score < 0.2:
        return 2.5  # Neutral
    elif -0.5 <= compound_score < -0.2:
        return 2.0  # Negative
    else:
        return 1.0  # Very negative

def translate_feedback(text):
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

if __name__ == '__main__':
    app.run(debug=True)







# from flask import Flask, render_template, request, send_from_directory, session
# from PIL import Image
# import pytesseract
# import os
# import pandas as pd
# from nltk.sentiment import SentimentIntensityAnalyzer
# import Levenshtein
# import nltk
# from googletrans import Translator
# nltk.download('vader_lexicon')

# app = Flask(__name__)
# app.secret_key = b'\x86YIl%i\xd5\xc0\x8e\xe8T\x8d\xf75&\x98-k+1\xaf\x17\xbcB'  # Set a secret key for session management

# # Read data from CSV
# students_data = pd.read_csv('student_data.csv')

# feedback_columns = ['student_id', 'feedback_date', 'feedback', 'rating']
# feedback_data = pd.DataFrame(columns=feedback_columns)

# FEEDBACK_FILE = 'feedback.xlsx'

# # Set the upload folder
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Path to the Tesseract executable (update this with your actual path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Initialize the SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()

# # Load the grading data from Excel sheet
# grading_data = pd.read_csv('your_excel_sheet.csv')

# # Initialize the translator
# translator = Translator()

# @app.route('/')
# def login():
#     return render_template('login.html')

# @app.route('/dashboard', methods=['POST'])
# def dashboard():
#     email = request.form.get('email')
#     password = request.form.get('password')

#     user = students_data[students_data['mail'] == email]

#     if not user.empty and str(user['student_id'].iloc[0]) == password:
#         user_info = user.iloc[0]  # Assuming email is unique, taking the first match
#         session['student_id'] = str(user_info['student_id'])  # Store student ID in session
#         return render_template('dashboard.html', user=user_info)
#     else:
#         error = 'Invalid email or password'
#         return render_template('login.html', error=error)

# @app.route('/upload', methods=['POST'])
# def upload_and_analyze():
#     if 'image' not in request.files:
#         return render_template('upload_result.html', error='No image provided')

#     image = request.files['image']

#     if image.filename == '':
#         return render_template('upload_result.html', error='No image selected')

#     if image:
#         # Save the image to the upload folder
#         image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
#         image.save(image_path)

#         # Perform analysis on the image (including grading)
#         extracted_text, grade = perform_ocr_and_grade(image_path)

#         return render_template('upload_result.html', image_path=image_path, extracted_text=extracted_text, grade=grade)

#     return render_template('upload_result.html', error='Error processing the image')

# def perform_ocr_and_grade(image_path):
#     # Open the image using Pillow
#     img = Image.open(image_path)

#     # Use Tesseract OCR to extract text
#     extracted_text = pytesseract.image_to_string(img)

#     # Get the grade based on extracted text
#     grade = get_grade_from_extracted_text(extracted_text)

#     return extracted_text, grade

# def get_grade_from_extracted_text(extracted_text):
#     similarity_scores = grading_data['answer'].apply(lambda ans: Levenshtein.ratio(extracted_text, ans))

#     # Get the maximum similarity score
#     max_similarity = similarity_scores.max()

#     # Map the similarity score to the grade scale (adjust the mapping as needed)
#     grade = max_similarity * 100

#     return grade
    
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/submit_feedback', methods=['POST'])
# def submit_feedback():
#     if request.method == 'POST':
#         # Get feedback data from the form
#         feedback_text = request.form.get('feedback')

#         # Get the student ID from the session
#         student_id = session.get('student_id')

#         # Check if student_id is not None
#         if student_id is not None:
#             # Validate if the student ID exists in your data
#             user = students_data[students_data['student_id'] == int(student_id)]

#             if not user.empty:
#                 # Convert int64 to native Python integer
#                 student_id = int(student_id)

#                 # Perform sentimental analysis
#                 sentiment_score = sia.polarity_scores(feedback_text)['compound']
                
#                 # Map sentiment score to the rating scale
#                 rating = get_sentiment_rating(sentiment_score)

#                 # Translate feedback to English
#                 translated_feedback = translate_feedback(feedback_text)

#                 # Append feedback and rating to the feedback DataFrame
#                 feedback_data.loc[len(feedback_data)] = [student_id, pd.Timestamp.now(), translated_feedback, rating]

#                 # Save the updated feedback DataFrame to Excel
#                 feedback_data.to_excel(FEEDBACK_FILE, index=False)

#                 return render_template('feedback_confirmation.html', student_id=student_id, rating=rating)
#             else:
#                 error = 'Invalid Student ID'
#                 return render_template('error_page.html', error=error)
#         else:
#             error = 'Invalid Student ID'
#             return render_template('error_page.html', error=error)

#     return render_template('error_page.html', error='Error submitting feedback')

# def get_sentiment_rating(compound_score):
#     if compound_score >= 0.5:
#         return 5.0  # Very positive
#     elif 0.2 <= compound_score < 0.5:
#         return 3.5  # Positive
#     elif -0.2 <= compound_score < 0.2:
#         return 2.5  # Neutral
#     elif -0.5 <= compound_score < -0.2:
#         return 2.0  # Negative
#     else:
#         return 1.0  # Very negative

# def translate_feedback(text):
#     try:
#         translation = translator.translate(text, dest='en')
#         return translation.text
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return text

# if __name__ == '__main__':
#     app.run(debug=True)




