#In this python file we are not recognizing the faces but Training to recognize the face
# We are using default recogniser that is provided by the openCV Library
# We are also using os module to find the directory of the images
# We are going to implement recognizer by using function called cv2.face.LBPHFaceRecognizer_create()
# Then we will use Python pillow library to load images from the directory
# We will also use Pickle library to store the Lables 
#________________________Lets start the project_______________________#

# Importing Modules necessary for the Training
import cv2
import os
import numpy
from PIL import Image
import pickle

#Determining the Base or root directory where our python file is present 
BaseDirectory = os.path.dirname( os.path.abspath(__file__))
print(BaseDirectory)

#We have fount the working directory with the help of os library
#Now the path of the Images Folder is going to be initialised
ImageDirectory = os.path.join(BaseDirectory,"Training Data")
print(ImageDirectory)

#As the path of the image directory is been found out
# We are going to create an recognizer 
Recogniser =cv2.face.LBPHFaceRecognizer_create()

#Recogniser is created
#Now we have to load the Face Casecade or haarcascade file to predict the faces from the images
FaceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#Now the Haarcascade.xml file is loaded

# creating a variable CurrentID to initialise an id for the pictures we have to keep track of
CurrentId = 0 

# Creating an dictionary to store label name with the respective label ID
LabelID = {}

# creating a list to store Label id data
YLabel=[]
# creating a list to store the imagein the form of array 
XTrain=[]

#Now running loop across the Image Directory that has been found 
for root , dir, files in os.walk(ImageDirectory):
    
    # running another loop across files in files that has been found
    for file in files:
        
        #Implementing an If statement for capturing only .jpg file
        if file.endswith('jpg'):
            
            # for that particular file there is a folder named the persons Name
            #Folder Name is initialised to the label name 
            label = os.path.basename(root)
            
            #root directory and file name is added to create the complete path of file 
            path = os.path.join(root ,file)
            print(path) 
            print(label)

            # checking if label is already initialised with an ID if not Id is initialised
            if not label in LabelID:
                
                #Initialising to the label
                LabelID[label]= CurrentId
                
                # Incrementing the ID
                CurrentId += 1

            ID = LabelID[label]

            # Now Pillow library is used
            # Image Object from Pillow Module is used to open the image through the given path
            # Parallell the Image is been converted into the Grayscale

            OriginalImage = Image.open(path).convert("L")
            # image is Being Resized for better understanding

            OriginalImage = OriginalImage.resize((550 , 550) , Image.ANTIALIAS )  
            
            # Converting Image into an numpy array 
            ImageArray = numpy.array(OriginalImage,"uint8")
            print(ImageArray)      
            print(LabelID)

            #Using FaceCascade to detect the multiscale Faces from an Image
            Faces = FaceCascade.detectMultiScale(ImageArray , 1.3 , 5)
            
            for x,y,w,h in Faces:

                    # Finding the region of intrest     
                    ROI = ImageArray[y:y+h , x:x+w]
                    
                    # Appending the region of intrest in to Xtrain List
                    XTrain.append(ROI)

                    #Appending ID in to label list
                    YLabel.append(ID)
print(YLabel)
#print(XTrain)                    

# Now label are being writed in to an pickel file that can be furthur used while recognition
with open('Label.pickle' , 'wb') as file:
    pickle.dump(LabelID , file)

# training the computer for the Given setup of images with the label id
Recogniser.train( XTrain ,numpy.array(YLabel))

# Saving the trained data into a .yml file for recognising
Recogniser.save('Training.yml')