from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def health():
    return "GestureSpeak server is running!"

if __name__ == "__main__":
    # Use Railway's PORT environment variable, fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
