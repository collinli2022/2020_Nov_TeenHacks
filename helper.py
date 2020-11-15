import cv2
import numpy as np
import time
from process import Process
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
import io
import qimage2ndarray


class HeartRate:

    def __init__(self):
        self.frame = np.zeros((10,10,3),np.uint8)
        self.process = Process()
        self.bpm = 0

    def main_loop(self, frame):

            self.process.frame_in = frame
            status = self.process.run()
            
            self.frame = self.process.frame_out #get the frame to show in GUI
            self.f_fr = self.process.frame_ROI #get the face to show in GUI
            #print(self.f_fr.shape)
            self.bpm = self.process.bpm #get the bpm change over the time
            
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            cv2.putText(self.frame, "FPS "+str(float("{:.2f}".format(self.process.fps))),
                           (20,460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),2)
            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            #self.lblROI.setGeometry(660,10,self.f_fr.shape[1],self.f_fr.shape[0])
            self.f_fr = np.transpose(self.f_fr,(0,1,2)).copy()

            consistantBPM = -1

            if self.process.bpms.__len__() > 50:
                if(max(self.process.bpms-np.mean(self.process.bpms))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
                    consistantBPM = ("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")

            return self.f_fr, consistantBPM, status