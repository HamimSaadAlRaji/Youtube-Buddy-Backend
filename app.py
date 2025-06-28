from flask import Flask, request, jsonify
from flask_cors import CORS
from Agent import answer_question

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    url = data['url']
    question = data['question']
    answer = answer_question(url, question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5000)