from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    file_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # In a real application, you'd want to save the file
            # For now, we'll just return the filename
            file_path = file.filename
    return render_template('index.html', file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)