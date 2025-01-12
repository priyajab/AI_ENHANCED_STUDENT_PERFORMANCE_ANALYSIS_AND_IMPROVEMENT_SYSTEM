from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to handle image upload and analysis
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

        # Perform analysis on the image (replace this with your actual analysis code)

        # For now, just return the uploaded image path
        return render_template('upload_result.html', image_path=image_path)

    return render_template('upload_result.html', error='Error processing the image')

# Function to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
