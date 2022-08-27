# -*- coding: utf-8 -*-
"""

ACCICARE

"""

import cv2 
import dlib 
from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
from time import sleep
import time
from threading import Thread
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import os 
import pyrebase
#from bson import json_util
#import json
import pyttsx3
import speech_recognition as sr 
import datetime
import pymongo



t=30
t2=34

class faceblood(Thread):
    def run (self):
        def detect_and_predict_blood(frame, faceNet, bloodNet):
            
           
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            
           
            faceNet.setInput(blob)
            detections = faceNet.forward()
            
          
            faces = []
            locs = []
            preds = []
        
            for i in range(0, detections.shape[2]):
                
               
                confidence = detections[0, 0, i, 2]
                
            
                if confidence > 0.5:
                    
                    
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                   
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    
                    
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
       
            if len(faces) > 0:
                
               
                faces = np.array(faces, dtype="float32")
                preds = bloodNet.predict(faces, batch_size=32)
                
            
            return (locs, preds)
       
        prototxtPath = os.path.sep.join([r"deploy.prototxt"])
        weightsPath = os.path.sep.join([r"res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        bloodNet = load_model('blood_noblood_classifier.model')
       
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        while True:
            
            
           
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            
            
            (locs, preds) = detect_and_predict_blood(frame, faceNet, bloodNet)
       
            for (box, pred) in zip(locs, preds):
                
               
                (startX, startY, endX, endY) = box
                (blood, noblood) = pred
                
                global label
                label = "Blood" if blood > noblood else "No Blood" 
                color = (0, 0, 255) if label == "Blood" else (0, 255, 0)
                
             
                #label = "{}: {:.2f}%".format(label, max(blood, noblood) * 100)
                
                print(label)
               
                
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
            cv2.imshow("Frame", frame)
            cv2.waitKey(1) 
            
            
            global t2
            if t2==1:
                global outblood
                outblood= label
                print('------------------Decision:',outblood,'-------------------' )
                cv2.destroyAllWindows()
                vs.stop()
                break
                

        
        
class timer2(Thread):
     def run(self):
         global t2
         while t2:
             time.sleep(1)
             t2 -= 1     
             #print(t2)
             #sleep(1)


#-----------------------------------------------------------------------------------------------------
SCORE1 = 0
class blink(Thread):
    def run(self):
        cap = cv2.VideoCapture(0)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        def eye_aspect_ratio(eye):
        	A = distance.euclidean(eye[1], eye[5])
        	B = distance.euclidean(eye[2], eye[4])

        	C = distance.euclidean(eye[0], eye[3])
        	eye = (A + B) / (2.0 * C)

        	return eye

        total = 0
        
        count = 0
         
        while True: 
            success,img = cap.read()
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)

            for face in faces:
                landmarks = predictor(imgGray,face)

                landmarks = face_utils.shape_to_np(landmarks)
                leftEye = landmarks[42:48]
                rightEye = landmarks[36:42]

                leftEye = eye_aspect_ratio(leftEye)
                rightEye = eye_aspect_ratio(rightEye)

                eye = (leftEye + rightEye) / 2.0
                if eye<0.3:
                    count+=1
                else:
                    if count>=3:
                        total+=1

                    count=0
                    
            global res                 
            
            def result(total):
                
                if(total<=2): 
                    SCORE1 = 1
                    return SCORE1
                elif(total>=2 and total<=5): 
                    SCORE1 = 2
                    return SCORE1
                elif(total>5): 
                    SCORE1 = 3
                    return SCORE1
                
            
            res=result(total)  
            #ct = json.dumps(datetime.now(), default=json_util.default)



            #ct=datetime.now()
            cv2.putText(img, "Blink Count:{}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(res)
            cv2.imshow('ASIAI Live',img)
            cv2.waitKey(1)
            
             
            global outblink 
            global t
            if t == 1:
                outblink=res
                print('\n\n\n\n------------------ Decision:',outblink, '-------------------')
                return outblink
                cap.stop() 
                cv2.destroyAllWindows() 
                break  
                     
            #sleep(1)
    
class timer(Thread):
     def run(self):
         global t
         while t:
             time.sleep(1)
             t -= 1     
             #print(t)
             #sleep(1)


             
def main_conscious():
    
    obj1=blink()
    obj2=timer()


    obj1.start()
    sleep(0.2)
    obj2.start()
    sleep(0.2)


    obj1.join()
    obj2.join()
    
    

def main_faceblood():
    
    obj3=faceblood()
    obj4=timer2()  
    
    obj3.start()
    sleep(0.2)
    obj4.start()
    sleep(0.2)


    obj3.join()
    obj4.join()
    
SCORE2 = 0
def nlp():
    def SpeakText(command):
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def msg_to_victim():
        engine = pyttsx3.init()
        engine.say("Kindly hold on!! All the necessary measures are being taken and you will receive the medical help within next 10 minutes from a nearby hospital!")
        engine.runAndWait()

    def msg_to_victim_neg():
        engine = pyttsx3.init()
        engine.say("Alright. glad that you are safe")
        engine.runAndWait()

    text_speech = pyttsx3.init()
    text_speech.say("Hello! This is ACCICARE. Do you need any medical assistance?")
    text_speech.runAndWait()
    r = sr.Recognizer()
    cond = 1
    while(cond):   
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.6)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                #assist = ['ah','aahh','aaahhh','ahhh','ahh','aaah','aah','please','yes','help','need','yup','yeah','haan','haaa']
                assist = ['no','nope','not','nah','no need','not necessary','negative','never','fine','good']
                print("Victim: "+MyText)
                result = [ele for ele in assist if(ele in MyText)]
                T_OR_F = False
                VAL = ">>>>>VICTIM REQUIRES MEDICAL ASSISTANCE<<<<<"
                global nlpout 
                def nlpoutput():
                    if(VAL==">>>>>VICTIM REQUIRES MEDICAL ASSISTANCE<<<<<") :
                        T_OR_F = True

                        SCORE2 = 1
                        return SCORE2
                    else:
                        T_OR_F = False

                        SCORE2 = 3
                        return SCORE2
                    print(T_OR_F)
                nlpout = nlpoutput()
                if(str(bool(result))!='True') :
                    print(VAL)
                    msg_to_victim()
                    cond = 0
                else:
                  msg_to_victim_neg()  
                  cond = 0  
                  print(T_OR_F)  
                #print(T_OR_F)   
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))     
        except sr.UnknownValueError:
            print("Listening . . .")
        
        
    
def findhospital():
    

    url="mongodb+srv://Accicare:Accicare@cluster0.fl69kut.mongodb.net/Accicare?retryWrites=true&w=majority"
    client=pymongo.MongoClient(url)
    db=client.Accicare
    hospitalusers=db.users
    gps=db.GPS
    nearhosp=db.nearhospital
    for doc in gps.find({},{ "_id": 0, "location": 1 }):
        for key,value in doc.items():
          loc=value
          acc_loc=('({} )'.format(value))
          
    acc_loc=(11.065422480254,77.09276856754401)   
    from numpy import array
    import pandas as pd 
    from math import radians, cos, sin, asin, sqrt
    userdf = pd.DataFrame(list(hospitalusers.find()))
    #userdf

    def dist(a, b):
      lon1 = radians(a[1])
      lon2 = radians(b[1])
      lat1 = radians(a[0])
      lat2 = radians(b[0])
          
      # Haversine formula
      dlon = lon2 - lon1
      dlat = lat2 - lat1
      a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
     
      c = 2 * asin(sqrt(a))
        

      r = 6371
          
      # calculate the result
      return(c * r)
    locarr = userdf['location'].values
    #locarr
    loc_split_arr = []
    for loc in locarr:
      #print(loc)
      lat, lng = loc.split()
      loc_split_arr.append((float(lat), float(lng)))

    distances = []

    for loc in loc_split_arr:
       loct = tuple(loc)
       print(loct)
       distances.append(dist(loct,acc_loc))
       
    #distances
    idx = []
    auth=[]
    for i, d in enumerate(distances):
      if d < 9: idx.append(i) 
    nearuserdf = userdf.iloc[idx] 
    nearuserloc = nearuserdf['location'].values
    #nearuserloc
    for i, loc in enumerate(nearuserloc):
      lat, lng = loc.split()
      nearuserdf['location'].iloc[i]= "https://maps.google.com/maps?q={},{}".format(lat, lng)
    near_hosp_json = nearuserdf['email']
   
    j=[]
    for i in  near_hosp_json:

      j.append(i)
      new_loc={'nemail':j}
    
    
            
def conclude():
   
   finalscore = outblink + nlpout
   if finalscore == 6:
       check = 'Consious'
   else:
       check= 'Unconsious'
       
   if outblood == 'Blood' and  check == 'Unconsious':
       print(outblood)
       print(check)
       print("Decision: Need Medical help")
       return 'Need medical help'
   elif outblood == 'No Blood' and check == 'Consious':
       print(outblood)
       print(check)
       print('Decision: No Need Medical help' )
       return 'No Need medical help'
   elif outblood == 'No Blood' and check == 'Unconsious':   
       print(outblood)
       print(check)
       print('Decision: Need Medical help')
       return 'Need medical help'
   elif outblood == 'Blood' and check == 'Consious':   
       print(outblood)
       print(check)
       print('Decision: Need Medical help')
       return 'Need medical help'



    
if __name__ == '__main__':
    
    main_faceblood()
    main_conscious()
    nlp()   
    conclude()
    
    con = conclude()
    if con =='Need medical help':
        findhospital()
    else:
        print('Exiting....')
        

    
    

    
    



   