import kivy
from kivy.app import App
#kivy.require('1.9.0') #<-- not really needed if its updated... so... lets pray :)
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.pagelayout import PageLayout 
from kivy.properties import ObjectProperty
from kivy.utils import get_color_from_hex 
from kivy.config import Config
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
import kivy.core.image
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np
import cv2
import requests
from deepface import DeepFace
from PIL import Image as PILImage

# personal imports
from helper import HeartRate
Config.set('graphics', 'resizable', True)


# ----------HeartRate---------
class KivyCamera(Image):
    def __init__(self, parent, capture, labelBpm, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture  # data to read
        self.parent = parent    # this object's parent (= box layout)
        self.paused = False     # pause state
        self.started = False    # start state <= unused !
        self.displayBpm = labelBpm

    # starts the "Camera", capturing at 30 fps by default
    def start(self, fps=24):
        print('start')
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.heartRate = HeartRate()

    def start1(self, fps=24):
        print('start')
        Clock.schedule_interval(self.update1, 1.0 / fps)
        self.heartRate = HeartRate()

    # for just face recognition
    def update(self, dt):
        # should be in start, not update
        rectangle_color = (0, 165, 255)

        ret, self.frame = self.capture.read()

        if ret and not self.paused:
            # Initialize a face cascade using the frontal face haar cascade provided
            # with the OpenCV2 library
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # For the face detection, we need to make use of a gray colored
            # image so we will convert the baseImage to a gray-based image
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Now use the haar cascade detector to find all faces in the image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # For now, we are only interested in the 'largest' face, and we
            # determine this based on the largest area of the found
            # rectangle. First initialize the required variables to 0
            max_area = 0
            x = 0
            y = 0
            w = 0
            h = 0

            # Loop over all faces and check if the area for this face is
            # the largest so far
            for (_x, _y, _w, _h) in faces:
                if _w * _h > max_area:
                    x = _x
                    y = _y
                    w = _w
                    h = _h
                    max_area = w * h

            # If one or more faces are found, draw a rectangle around the
            # largest face present in the picture
            if max_area > 0:
                cv2.rectangle(self.frame, (x - 10, y - 20), (x + w + 10, y + h + 20),rectangle_color, 2)
            
            # convert it to texture
            image_texture = self.get_texture_from_frame(self.frame, 0)

            # display image from the texture
            self.texture = image_texture
            self.parent.ids['imageCamera'].texture = self.texture

            print(type(self.frame))
            print(type(self.texture))

    # for heart rate
    def update1(self, dt):
        ret, self.frame = self.capture.read()
        adjFrame, bpmArr, status = self.heartRate.main_loop(self.frame)
        if status:
            image_texture = self.get_texture_from_frame(adjFrame, 0)

            if bpmArr != -1:
                self.displayBpm.text = bpmArr

            # display image from the texture
            self.texture = image_texture
            self.parent.ids['imageCamera'].texture = self.texture

    def stop(self):
        Clock.unschedule(self.update)

    def stop1(self):
        Clock.unschedule(self.update1)

    def get_texture_from_frame(self, frame, flipped):
        buffer_frame = cv2.flip(frame, flipped)
        buffer_frame_str = buffer_frame.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #print('('+str(frame.shape[1])+';'+str(frame.shape[0])+')')
        image_texture.blit_buffer(buffer_frame_str, colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

class HeartRateLayout(BoxLayout):
    started = False
    # initialize the "Camera"
    # kivyCamera = Image(source='mini.jpg')
    kivyCamera = None

    def startCamera(self, imageCamera, buttonStart, buttonStop, labelBpm):

        if not self.started:
            self.capture = cv2.VideoCapture(0)

            self.kivyCamera = KivyCamera(self, self.capture, labelBpm)
            imageCamera = self.kivyCamera

            bpm = self.kivyCamera.start1()

            # Set as started, so next action will be 'Pause'
            self.started = True
            buttonStart.text = 'Pause'
            buttonStop.text = 'Stop' # useless ?
            # when started, set stop to enabled
            # &&
            # when started, let start enabled
            buttonStop.disabled = False  # enabled
        else:
            # Set as paused
            self.kivyCamera.paused = not self.kivyCamera.paused
            imageCamera.source = 'mini.jpg'

            if not self.kivyCamera.paused:
                buttonStart.text = 'Pause'
            else:
                buttonStart.text = 'Start'

    # stops the kivy camera (doesn't care either camera is started or paused)
    def stopCamera(self, imageCamera, buttonStart, buttonStop, labelBpm):

        if self.started: # Was running at click
            buttonStop.disabled = True
            self.started = False  # stop what was "started"
            # Reset kivy camera to 'mini.jpg' (home image)
            self.kivyCamera.stop1()
            self.kivyCamera = Image(source='mini.jpg')
            imageCamera.source = self.kivyCamera.source
            imageCamera.reload()

            # Reset Start Button to 'Start value' (e.g. instead of eventually 'Pause value')
            buttonStart.text = 'Start'

            # Release the capture
            self.capture.release()

# ----------HeartRate---------

# ----------Happiness---------
class FacialAttributeLayout(BoxLayout):
    kivyCamera = None

    def capture(self, buttonCapture, resultLabel):
        camera = self.ids['camera']
        camera.export_to_png("ffaaccee.png")
        im1 = PILImage.open('./ffaaccee.png')
        im1 = im1.convert('RGB')
        im1.save('ffaaccee.jpg')
        print("Captured")
        self.analyze('ffaaccee.jpg')


    def analyze(self, imgFileLocation):
        obj = DeepFace.analyze(imgFileLocation, actions = ['age', 'gender', 'race', 'emotion'])
        output = "Years old: " + str(obj["age"]) + "\nRace: " + str(obj["dominant_race"]) + "\nEmotion: " + str(obj["dominant_emotion"]) + "\nGender: " + str(obj["gender"])
        print(output)
        self.ids['results'].text = output
        return obj
# ----------Happiness---------

# ----------Complements---------

# ----------Complements---------

# -----------COVID-----------
class CovidLayout(BoxLayout):
    def getDataGlobal(self):
        url = "https://api.covid19api.com/summary"
        payload = {}
        headers= {}
        response = requests.request("GET", url, headers=headers, data = payload)

        stuff = response.json()
        return stuff['Global']

    def refresh(self, NewConfirmed, TotalConfirmed, NewDeaths, NewRecovered, TotalRecovered):
        data = self.getDataGlobal()
        NewConfirmed.text = "New Cases: " + str(data['NewConfirmed'])
        TotalConfirmed.text = "Total: " + str(data['TotalConfirmed'])
        NewDeaths.text = "New Deaths: " + str(data['NewDeaths'])
        NewRecovered.text = "New Recovered: " + str(data['NewRecovered'])
        TotalRecovered.text = "Total Recovered: " + str(data['TotalRecovered'])

# -----------COVID-----------

# create App class 
class MyApp(App):
    # override the build method and return the root widget of this App 
  
    def build(self): 
        # Define a grid layout for this App 
        self.layout = GridLayout(cols = 2, padding = 10) 
  
        self.button1 = Button(text = "Heart Rate") 
        self.layout.add_widget(self.button1)

        self.button2 = Button(text = "Happiness") 
        self.layout.add_widget(self.button2)

        self.button3 = Button(text = "Wink")
        self.layout.add_widget(self.button3)

        self.button4 = Button(text = "Covid")
        self.layout.add_widget(self.button4)
  
        # Attach a callback for the button press event 
        self.button1.bind(on_press = self.onButtonPress1)
        self.button2.bind(on_press = self.onButtonPress2)
        self.button3.bind(on_press = self.onButtonPress3)
        self.button4.bind(on_press = self.onButtonPress4)
          
        return self.layout 
  
    # On button press - Create a popup dialog with a label and a close button 
    def onButtonPress1(self, button):
        layout = HeartRateLayout()
        self.popup1 = Popup(title ='Heart Rate', 
                      content = layout, 
                      size_hint =(0.85, 0.85))
        self.popup1.open()

    def stopCam(self, button):
        self.popup1.dismiss()

    def onButtonPress2(self, button):
        layout = FacialAttributeLayout()
        self.popup2 = Popup(title ='Facial Attribute', 
                      content = layout, 
                      size_hint =(0.85, 0.85))
        self.popup2.open()

    def onButtonPress3(self, button):
        pass

    def onButtonPress4(self, button):
        layout = CovidLayout()
        self.popup4 = Popup(title ='Covid', 
                      content = layout, 
                      size_hint =(0.85, 0.85))
        self.popup4.open()

# run the App 
if __name__ == '__main__': 
    MyApp().run()