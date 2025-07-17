from flask import Flask, request, jsonify
from flask_cors import CORS
from Agent import answer_question

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/ask', methods=['POST'])
def ask():
    context = request.json
    answer = answer_question(context)
    print("Received context:", context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5000)