# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils import load_trained_model, predict_model, preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_trained_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            predictions = predict_model(image, model)

            # Example logic to interpret predictions
            skin_type = "Oily" if predictions[0][0] > 0.5 else "Dry"
            skin_issues = "Wrinkles, Spots"

            recommendations = [
                {"name": "Vitamin C Serum", "image": url_for('static', filename='products/vitc.jpg'), "link": "#"},
                {"name": "Moisturizer", "image": url_for('static', filename='products/moist.jpg'), "link": "#"},
                {"name": "Sunscreen", "image": url_for('static', filename='products/sunscreen.jpg'), "link": "#"}
            ]

            return render_template("results.html", 
                                   image_url=url_for('static', filename=f'uploads/{filename}'),
                                   skin_type=skin_type,
                                   skin_issues=skin_issues,
                                   recommendations=recommendations)
    return render_template('analyze.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
