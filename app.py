from flask import Flask, render_template, request, jsonify

from bot import chat

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.form.get("msg", "")
    response = chat(user_text)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
