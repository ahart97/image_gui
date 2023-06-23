# YOLO object detection
import cv2 as cv
import numpy as np
import time
import os

class YoloModel():
    def __init__(self, confidence = 0.5):
        # Load names of classes and get random colors

        self.model_dir = os.path.join(os.getcwd(), 'App', 'Models', 'YoloModelData')

        self.classes = open(os.path.join(self.model_dir, 'coco.names')).read().strip().split('\n')
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

        # Give the configuration and weight files for the model and load the network.
        self.net = cv.dnn.readNetFromDarknet(os.path.join(self.model_dir, 'yolov3.cfg'), os.path.join(self.model_dir, 'yolov3.weights'))
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # determine the output layer
        ln = self.net.getLayerNames()
        self.ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.conf = confidence
        self.observed_classes = ['']

    def classify_objects(self, img):
        """
        img: n_width x n_height x n_channels
        """
        
        self.observed_classes = ['']

        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        #Set the input image
        self.net.setInput(blob)
        #Propagate through the model
        outputs = self.net.forward(self.ln)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        self.post_process(img, outputs)
        #cv.imshow('window',  img)
        #cv.waitKey(0)

    def post_process(self, img, outputs):
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > self.conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf, self.conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                #cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #self.observed_classes.append("{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i]))
                #cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                self.observed_classes.append(self.classes[classIDs[i]])

        #Only take the unqie class instances
        self.observed_classes = np.unique(self.observed_classes)


if __name__ == '__main__':
    Model = YoloModel(confidence = 0.6)

    cv.namedWindow('window')
    Model.load_image(cv.imread('images/horse.jpg'))

    cv.destroyAllWindows()