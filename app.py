from flask import Flask, render_template
from model import get_top_students  # Import your ML code to get top students

app = Flask(__name__)

@app.route('/')
def home():
    top_students = get_top_students()  # Get the top 3 students from the model
    return render_template('result.html', students=top_students)  # Pass them to the HTML template

if __name__ == '__main__':
    app.run(debug=True)
