# import the opencv library

import cv2
import pickle
import face_recognition


# define a video capture object
cap = cv2.VideoCapture(0)
cap.set(3,1280)#Setting width with 3
cap.set(4,720)#Setting height with 4

#load the file
file=open("Encodedfiles.p",'rb')
encodedlistwithID=pickle.load(file)#loading the data in the file Encodedfiles.p
file.close()
encodedimage, id=encodedlistwithID

while True:
	
	# Capture the video frame
	# by frame
	success, img = cap.read()
	imgS=cv2.resize(img,(0,0),None,.25,.25)#rescaling the image to reduce the computational time
	imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
	face=face_recognition.face_locations(imgS)#identifying the face location
	facecurrent=face_recognition.face_encodings(imgS,face)# Encoding images
    # Display the resulting frame
	s="None"
	for encodeface, faceloc in zip(facecurrent,face):
		matches=face_recognition.compare_faces(encodedimage,encodeface)
		for i in range(len(matches)):
			if matches[i]==True:
				s=id[i]
	if s!="None":
		img=cv2.putText(img,s,(50,50),cv2.HOGDESCRIPTOR_L2HYS,1,(0,255,0),2)
	cv2.imshow("Frame",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Destroy all the windows
cv2.destroyAllWindows()
		
    



