"""User interface app for Surveys team."""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    """Single page application.

    Returns
    -------
        html: Page
    """
    return render_template("index.html")


# if __name__ == "__main__":
#     app.run(debug=True)
