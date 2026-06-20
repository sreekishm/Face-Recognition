import cv2
import face_recognition
import pickle#to dump the list generated to a file 
import os# to import the name of the files from a directory
faceslist=os.listdir('faces')#get the name of the files in the folder faces
images=[]
ID=[]
for path in faceslist:
    images.append(cv2.imread(os.path.join('faces',path)))#get the images from the files
    ID.append(os.path.splitext(path)[0])#adding the image name to the ID list

def encoding(images):
    encodeList=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)#converting from BGR to RGB as opencv uses BGR and Face Recognition uses RGB
        encode=face_recognition.face_encodings(i)[0]# Encoding images
        encodeList.append(encode)
    return encodeList    

print("Encode starting..")
encodeList=encoding(images)
encodeListwithID=[encodeList,ID]
print("Encoding completed")
file=open("Encodedfiles.p",'wb')#opening a file to store the encodings and ID
pickle.dump(encodeListwithID,file)#dumping the encodings with ID to file
file.close()
print("file saved")
