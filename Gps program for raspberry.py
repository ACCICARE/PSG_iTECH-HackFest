'''

------------ACCICARE ---------------
Python program to get location using neo-6m gps module in raspberry pi controller
**This program was designed for PYTHON 2.7 or later version**

'''


import serial
import time
import string
import pynmea2
url="mongodb+srv://Accicare:Accicare@cluster0.fl69kut.mongodb.net/Accicare?retryWrites=true&w=majority"
client=pymongo.MongoClient(url)
db=client.Accicare
gps=db.GPS

while True:
	port="/dev/ttyAMA0"
	ser=serial.Serial(port, baudrate=9600, timeout=0.5)
	dataout = pynmea2.NMEAStreamReader()
	newdata=ser.readline()

	if newdata[0:6] == "$GPRMC":
		newmsg=pynmea2.parse(newdata)
		lat=newmsg.latitude
		lng=newmsg.longitude
		loc = str(lat) +"," + str(lng)
        newloc={'location':loc}
        updatemongo=gps.update_one(newloc,{{"$set"}:{newloc}})
		print(updatemongo)