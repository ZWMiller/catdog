import flask
from flask import request, render_template
import numpy as np
import pandas as pd
from copy import deepcopy
from werkzeug import secure_filename
from PIL import Image
import catdog

# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open("frontend.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/uploader", methods=["GET","POST"])
def get_image():
    if request.method == 'POST':
        f = request.files['file']
        sfname = 'static/'+str(secure_filename(f.filename))
        f.save(sfname)

        clf = catdog.classifier()
       # clf.save_image(f.filename)
        
        return render_template('result.html', pred = clf.predict(sfname), imgpath = sfname)


#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=5000, debug=True)
