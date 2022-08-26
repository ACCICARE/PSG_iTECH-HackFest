# > > > > > > > > > - < ACCICARE > - < < < < < < < < <

![PSX_20210129_215539](https://user-images.githubusercontent.com/64918181/186962876-42c2bd30-34a3-442a-af38-377639874e2f.png "abc")

### OBJECTIVE :-

Injuries and deaths resulting from road accidents are a growing public health problem in India. Road crash deaths have increased by 31% from 2007 to 2017. Surveys show that a 
10 min reduction of the medical response time can be statistically associated with an average 
decrease in the probability of death by one-third, both on motorways and conventional roads. We 
propose an accident mitigation system that can be used in post-accident vehicles to reduce the 
death rate due to road accidents. The proposed framework is a hybrid decision algorithm that 
enables the system to take accurate and precise decisions that save lives that matter. We use 2 
conditions to decide if the accident victim needs medical help or is safe. This valuable accident 
information is sent to the hospitals along with the GPS location to enable hospitals in a 5km 
radius around the accident zone to see the information and respond to the accidents immediately. 
This system will reduce the response time by at least 10 min thereby reducing the death rate by 
one-third of the present death rate.
<pre>For detailed project description, kindly <a href="https://docs.google.com/document/d/1bpxX9G5cddCNAQdMuopLMRYfakJPcGxBzKy6H3qe60M/edit">click here</a>.</pre>


### WORKING SCREENSHOTS :-

#### NLP module
![](https://user-images.githubusercontent.com/64918181/186961190-5c31c06e-9cc9-4e72-b38d-f3bb26f09871.png)
<br>
<br>
#### Backend
![](https://user-images.githubusercontent.com/64918181/186962455-0cd22377-2764-475e-bf57-5101b74282a5.png)
<br>
<br>
#### Frontend
![](https://user-images.githubusercontent.com/64918181/186962864-18ded5a5-5619-4487-a67f-f3081dc71cb1.png)
<br>
<br>
#### Blood detection module
![](https://user-images.githubusercontent.com/64918181/186962470-5333adc6-b6a1-4f4c-b262-8c50fa3e75e9.png)
<br>
<br>
### METHODOLOGY :-

To check if the accident occurred or not, the Raspberry Pi is connected to the OBD (On-Board Diagnostics) port of the vehicle. The vehicle velocity and airbag information is taken from the OBD port to check for abrupt drop in velocity indicating high deceleration. This condition is checked and used to trigger the system as any accident will lead to high deceleration of the vehicle. The medical help request decision made by Accicare checks for 2 conditions to finalize the decision. 
- Presence of blood 
- Consciousness 

Once these conditions are checked using the input from the sensors, Accicare will decide if the passenger/driver needs medical help or not. Once Accicare decides if the victim needs medical help or is safe, the decision is sent to the hospitals in the 5km range to take quick emergency actions which include sending an ambulance to the accident zone and informing the police about the accident. To achieve this, the control system shown is used. 
<br>
GPS signals are collected using the NEO-6M module by the Raspberry Pi controller from the accident zone. Then, the controller generates a google maps link using the coordinates received from the GPS module.  This information along with all the accident information is sent to the cloud database.
Cloud database stores the accident information referenced with a unique ID. The database also has the details of the hospitals registered in the portal designed for emergency response. The cloud database is linked with the portal to enable real-time updating of accidents with reference to their ID as shown below.


![](https://user-images.githubusercontent.com/64918181/186972019-2f99fd77-32c3-4db6-aa2c-cc0dd47ccdba.png)

### TECHNOLOGIES USED :-
<pre>
               <b>Machine Learning                       IOT                         Computer Vision 
                     NLP                            MongoDB                       Embedded System 
                    Python                            HTML                              CSS 
                  JavaScript                          Flask                           Markdown <b>
</pre>
### VIDEO LINK :-


[![IMAGE ALT TEXT HERE](https://user-images.githubusercontent.com/64918181/186975848-c4cd1171-dee5-40f0-a0f7-f07f0de206e7.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)


### TEAM :-

| Bennett Joseph | Vishnuvasan | Venkat | Srirangan | Siddeshwar
| --- | --- | --- | --- | --- |
| [Github](https://github.com/issacharben "Bennett profile") | [Github](https://github.com/Cipher-unhsiV "Vishnu profile") | [Github](https://github.com/venkat-p-r "Venkat profile") | [Github](https://github.com/SriranganK "Srirangan profile") | [Github](https://github.com/Marcus270503 "Siddeshwar profile") |
| [LinkedIn](https://www.linkedin.com/in/bennett-joseph-718a18191/ "Bennett")| [LinkedIn](https://www.linkedin.com/in/vishnuvasan-srinivasan-0b2012194/ "Vishnu") | [LinkedIn](https://www.linkedin.com/in/venkat-p-r/ "Venkat") | [LinkedIn](https://www.linkedin.com/in/srirangan-kannan-7017a8245/ "Srirangan") | [LinkedIn](https://www.linkedin.com/in/siddeshwar-v-a-765521238/ "Siddeshwar") |
|![@Bennett](https://avatars.githubusercontent.com/issacharben?s=150&v=1)| ![@Vishnuvasan](https://avatars.githubusercontent.com/Cipher-unhsiV?s=150&v=1) | ![@Venkat](https://avatars.githubusercontent.com/venkat-p-r?s=150&v=1) | ![@Srirangan](https://avatars.githubusercontent.com/SriranganK?s=150&v=1) | ![@Siddeshwar](https://avatars.githubusercontent.com/Marcus270503?s=150&v=1)


