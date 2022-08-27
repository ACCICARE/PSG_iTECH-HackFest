from flask import Flask, render_template, url_for, request, session, redirect, flash
from flask_pymongo import PyMongo
import bcrypt
import pymongo
import numpy as np
app = Flask(__name__)


url="mongodb+srv://Accicare:Accicare@cluster0.fl69kut.mongodb.net/?retryWrites=true&w=majority"
client=pymongo.MongoClient(url)
db=client.Accicare
gps = db.GPS
for doc in gps.find():
    # print (doc)
    for key,value in doc.items():
       gmaps= "https://maps.google.com/maps?q={}".format(value)

hospitalusers = db.nearhospital
for doc in hospitalusers.find({},{ "_id": 0, "nemail": 1 }):
    #print (doc)
    for key,value in doc.items():
      print(value)


near=value
arraynear=np.array(near)

x1=arraynear[1]
x2=arraynear[0]
x3=arraynear[2] 
x4 = arraynear[3]
x5 = arraynear[4]

"""
url="mongodb+srv://Accicare:Accicare@cluster0.fl69kut.mongodb.net/Accicare?retryWrites=true&w=majority"
client=pymongo.MongoClient(url)
db=client.Accicare
nearhosp=db.nearhospital
for auth in nearhosp.find({},{"_id": 0,"nemail": 1}):
  print(auth)
"""

app.config['MONGO_DBNAME'] = 'Accicare'
app.config['MONGO_URI'] = 'mongodb+srv://Accicare:Accicare@cluster0.fl69kut.mongodb.net/Accicare?retryWrites=true&w=majority'

mongo = PyMongo(app)


@app.route("/")
@app.route("/main")
def main():
    return render_template('index.html')


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'email': request.form['email']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt(14))
            users.insert_one({'username': request.form['username'], 'password': hashpass, 'location': request.form['location'] ,'email': request.form['email']})
            return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/dashbord')
def index():
    if 'email' in session:
        return render_template('dashbord.html', username=gmaps ,email =session['email'])

    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    
    if request.method == 'POST':
        users = mongo.db.users
        signin_user = users.find_one({'email': request.form['email']})
        

        if signin_user:
            if bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt(14)):
                session['email'] = request.form['email']
                for x in near:
                    print(x)
                    if  x1== session['email'] or x2== session['email'] or x3==session['email'] or x4==session['email'] or x5==session['email'] :
                        return redirect(url_for('index'))
                    else:
                        return render_template('nodata.html',email=session['email'])
               

                     
                

                  

        flash('Username and password combination is wrong')
        return render_template('signin.html')

    return render_template('signin.html')
    

@app.route('/logout')
def logout():
    session.pop('username', None)
    return render_template('/signin')


if __name__ == "__main__":
    app.secret_key = 'mysecret'
    app.run(debug=True)
    app.run()
    

