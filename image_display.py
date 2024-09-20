from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
