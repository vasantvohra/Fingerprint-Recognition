import requests
import os
import json

from flask import Flask, request, jsonify, make_response,render_template , redirect, url_for ,send_from_directory

from Fingerprint_Recognition import *
from downloadImage import *
app=Flask(__name__)
UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods = ['POST','GET']) 
def application():
     Original = "" ; Candidate ="" ; flag=""
     if request.method == "POST":
          try:
               uid = request.form["id"]
               finger = request.form["finger"]
          except:
               uid = None
               finger = None
          Candidate = request.files["Candidate"]
          print(uid,finger); print(Candidate.filename)

          if finger:
               filename = download(uid,finger)
               Original = "./%s"%filename
               f2 = os.path.join(app.config["UPLOAD_FOLDER"], Candidate.filename); Candidate.save(f2)
               result = verify(Original,Candidate.filename)
          else:
               Original = request.files["Original"]
               print(Original.filename) 
               if Original and Candidate:
                    f1 = os.path.join(app.config["UPLOAD_FOLDER"], Original.filename)
                    f2 = os.path.join(app.config["UPLOAD_FOLDER"], Candidate.filename)
                    Original.save(f1); Candidate.save(f2)
                    result = verify(Original.filename,Candidate.filename)

          image_names = os.listdir(r'./static/img')
          #print(image_names)
          return render_template('index.html', result = result,image_names=image_names)        
     else:
        return render_template('index.html')
                          
@app.route('/<path:filename>')
def send_image(filename):
    return send_from_directory(".", filename)

               
if __name__=="__main__":
     app.run(debug = True)
