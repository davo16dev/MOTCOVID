

from __future__ import division, print_function, absolute_import


import argparse
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
import math

#deep sort
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')



# argparse
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--config", required=True,
	help="path to confif file")

ap.add_argument("-m", "--modelt", required=True,
	help="path to model file")

ap.add_argument("-c", "--confidence", type=float, default=0.25,
	help="minimum probability to filter weak detections")

ap.add_argument("-s", "--covid", type=float, default=50,
	help="distance covid")

args = vars(ap.parse_args())



#main
def main():

    contt = 0

    #auxiliar var
    D = []
    dism = []
    centroids = []

    #cargar caffe model
    net = cv2.dnn.readNetFromCaffe(args["config"], args["modelt"])

   #deep sort arguments
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    #create tracker
    tracker = Tracker(metric)

    #capture the video
    video_capture = cv2.VideoCapture(0)


    #fps count
    fps = 0.0
    while True:

        ret, frame = video_capture.read()
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame[...,::-1]) #bgr to rgb

        #   flag, image = video.read()

        #   det = image[:, :, ::-1]

        (h, w) = frame.shape[:2]

        #capture the box
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (800, 800)), 0.007843, (300, 300), 127.5)

        #variables
        det = []

        net.setInput(blob)

        ssddet = net.forward()

        #distance sensor
        violate = []

        confidence = 0

        for i in np.arange(0, ssddet.shape[2]):
            # extraer la confianza para las predicciones
            confidence = ssddet[0, 0, i, 2]
            # filtrar la confianza
            if confidence > args["confidence"]:
                # extraer el indice de las detecciones y los puntos
                idx = int(ssddet[0, 0, i, 1])
                box = ssddet[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x = int(x1 - (x2 / 2))
                y = int(y1 - (y2 / 2))
                #guardar los centroides
                centroids.append([ int((x1+x2)/2)  ,int((y1+y2)/2) ])
                #coordenadas de las detecciones
                det.append([x1, y1, x2, y2])

                ppp = confidence*100
                cv2.putText(frame, "conf"+str(round(ppp))+"%" , (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)


        #covid
            for cen in centroids:
                   for cen2 in centroids:
                        print(math.sqrt(math.pow((cen[0]-cen2[0]),2)+math.pow((cen2[1]-cen2[1]),2)))
                        d = math.sqrt(math.pow((cen[0]-cen2[0]),2)+math.pow((cen2[1]-cen2[1]),2))
                        if (d) < args["covid"]:
                            if not (d==0):
                                violate.append(cen)

        boxs = det
        features = encoder(frame,boxs)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        #non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        #call tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.putText(frame, "ID:"+ str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        cont = 0

        for cen in centroids:
             if cen in violate:
                cv2.circle(frame,(cen[0],cen[1]),40,(0,0,250),10)
                cont+=1

        contt+=1
        cv2.putText(frame,"Riesgo de Covid: "+ str(cont),(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow('salida', frame)
        

            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        cv2.putText(frame , str(fps) ,  (300,300) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0) , 2  )
        centroids = []


        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
