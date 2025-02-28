from flask import Flask, request, jsonify, render_template
import os
from api import genhive

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"Saving file to: {file_path}")
        
        result = {
            "name": file.filename,
            "genre": genhive(file_path)
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5555)
