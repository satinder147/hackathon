from direction import path
import cv2
import keras
from keras.models import load_model
from unet import Models
from keras.preprocessing.image import img_to_array
import numpy as np
from scipy import stats
from hardware.controll_arduino import Arduino
'''

obj=path()
img=cv2.imread("Segmentation2/mask/4.png",0)
img=cv2.resize(img,(300,300))
edges=cv2.Canny(img,100,150)
edges=cv2.dilate(edges,None,iterations=3)

cv2.imshow("ds",edges)
cv2.waitKey(0)
'''
ard=Arduino()
model=Models(256,256,3)
model=model.arch3()
model.load_weights("segmentation.MODEL")
cap=cv2.VideoCapture("/home/satinder/Desktop/hackathon/hackathon/data/videos/3.mp4")
def lane(lines):

    xs = []
    ys = []
    
    for x1, y1, x2, y2 in lines:
        xs.append(x1)
        xs.append(x2)
        ys.append(y1)
        ys.append(y2)
    if(len(xs)):
        slope, intercept,_,_,_ = stats.linregress(xs, ys)
        return (slope, intercept)
    return 1e-3,1e-3

def side(x,y,x1,y1,x2,y2):
    return (y2-y1)*(x-x1)-(y-y1)*(x2-x1)

def area(d1,d2,d3):
    label="none"
    if(d1>0):
        label="offroad->left"
    elif(d1<0 and d2>0):
        label="left region"
    elif(d3<0):
        label="offroad->right"
    elif(d2<0 and d3>0):
        label="right region"

    return label
  
n=0
while 1:
    n+=1
    try:
        if(n>181):
            n=0
        frame=cap.read()[1]
        frame=cv2.resize(frame,(256,256))

        frame2=cv2.blur(frame,(3,3))
        frame=img_to_array(frame2)
        frame=frame.astype("float")/255.0
        frame=np.expand_dims(frame,axis=0)
        prediction=model.predict(frame)[0]
        pre=prediction*255
        pre=cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY)
        cv2.imshow("pre",pre)
        pre=pre.astype("uint8")
        t,pre=cv2.threshold(pre,200,255,cv2.THRESH_BINARY)
        pre=cv2.dilate(pre,None,iterations=2)
        pre=cv2.erode(pre,None,iterations=2)
        pre=cv2.dilate(pre,None,iterations=4)
        lines=cv2.HoughLinesP(pre,1,np.pi/180,50,maxLineGap=100,minLineLength=5)
        left=[]
        right=[]
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    if(x1!=x2):
                        slope=(y2-y1)/(x2-x1)
                        if(slope>0):
                            left.append([x1,y1,x2,y2])
                        else:
                            right.append([x1,y1,x2,y2])

                        #cv2.line(frame2,(x1,y1),(x2,y2),(255,0,0),2)
        
        s1,i1=lane(left)
        s2,i2=lane(right)
        y1=256
        x1=(y1-i1)/s1
        x2=(y1-i2)/s2

        y2=40
        x3=(y2-i1)/s1
        x4=(y2-i2)/s2
        #cv2.line(frame2,(int(x1),y1),(int(x3),y2),(255,0,0),2)
        #cv2.line(frame2,(int(x2),y1),(int(x4),y2),(255,0,0),2)
        person=(128,256)
        cv2.circle(frame2,person,6,(0,255,0),-6)
        lower=(int((x1+x2)/2),256)
        lower_x=(i2-i1)/(s1-s2)
        lower_y=lower_x*s1+i1
        upper=(int(lower_x),int(lower_y))
        
        #d=((upper[1]-lower_y)/(upper[0]-lower_x))*person[0]-lower_x+lower_y-person[1]
        cv2.circle(frame2,lower,5,(0,0,255),-5)
        cv2.circle(frame2,upper,5,(0,0,255),-5)
        d1=side(person[0],person[1],lower[0],lower[1],upper[0],upper[1])#<0-> right else left
        d2=side(person[0],person[1],x1,y1,upper[0],upper[1]) #right       <0-> right else left
        d3=side(person[0],person[1],x2,y2,upper[0],upper[1]) #left
        lanee=area(d3,d1,d2)
        cv2.putText(frame2, lanee, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.line(frame2,lower,upper,(0,255,0),2)
        cv2.line(frame2,(int(x1),y1),upper,(255,0,255),2) #right
        cv2.line(frame2,(int(x2),y1),upper,(255,0,0),2) #left
        cv2.imshow("segmentation",pre)
        cv2.imshow("frame",frame2)
        
        if cv2.waitKey(1) and 0xFF==ord('q'):
            break
    except:
        pass
    
    #if(lanee=="")
    

    
