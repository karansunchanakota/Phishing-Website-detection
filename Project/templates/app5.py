import os
from flask import Flask, render_template, request

__author__ = 'Karan Sunchanakota'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
       url = request.form['Site']

if __name__ =="__main__":
    app.run(port=4555, debug=True)