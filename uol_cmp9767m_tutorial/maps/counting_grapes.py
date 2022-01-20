
import cv2
import numpy as np
from numpy import mean, array
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
from cv2 import namedWindow, cvtColor, imshow, inRange
from cv2 import destroyAllWindows, startWindowThread
from cv2 import COLOR_BGR2GRAY, waitKey, COLOR_BGR2HSV
from cv2 import blur, Canny, resize, INTER_CUBIC

class count_fruits:

    def __init__(self): 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/thorvald_001/kinect2_right_camera/hd/image_color_rect",Image, self.process_image)

    def process_image(self, camera):


        #cam= cv2.VideoCapture(0)
        self.kernelOpen=np.ones((1,0))
        self.kernelClose=np.ones((15,15))
        img = self.bridge.imgmsg_to_cv2(camera, "bgr8")
        img = resize(img, None, fx=0.6, fy=0.6, interpolation = INTER_CUBIC)
        #img=cv2.resize(img,(340,220))

        # lowerBound=np.array([65,15,10])
        # upperBound=np.array([255,256,256])

        lowerBound=np.array([100,18,46])
        upperBound=np.array([107,256,256])
        font = cv2.FONT_HERSHEY_SIMPLEX

        #convert BGR to HSV
        imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV',imgHSV)
        # create the Mask
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        #morphology
        self.maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernelOpen)

        # self.maskClose=cv2.morphologyEx(self.maskOpen,cv2.MORPH_CLOSE,self.kernelClose)
        self.maskClose=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernelClose)

        maskFinal=self.maskClose
        _,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(img,conts,-1,(255,256,256),1)
        for i in range(len(conts)):
            x,y,w,h=cv2.boundingRect(conts[i])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
            cv2.putText(img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
        
        cv2.imshow("maskOpen",self.maskOpen)
        cv2.imshow("maskClose", self.maskClose)
        cv2.imshow("mask",mask)
        cv2.imshow("cam",img)
        print(len(conts))
        if cv2.waitKey(10) &0xFF ==ord('q'):
                    cv2.cap.release()
                    cv2.destroyAllWindows()
        
                
if __name__ =='__main__':
    rospy.init_node('count_fruits')
    cf = count_fruits()
    rospy.spin()


