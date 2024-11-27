import cv2
import math
import cvzone 
from ultralytics import YOLO                                      
from sort import *                                 # to track the number of cars download the sort.py file by abewley from github      [ https://github.com/abewley/sort/blob/master/sort.py ]


# loading the video file
cap = cv2.VideoCapture('CC_Assets/cars.mp4')                          

# YOLO Model
model = YOLO('YOLO_Weights/yolov8n.pt') 


# list of class names
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# mask
mask = cv2.imread('CC_Assets/cars_mask.png')                 # since there's lot of regions in the area that are not to be detected and where desired object can not be detected,  so create a mask image that will hide all the undesired regions and will only keep the optimal required region and pass it for the detection


# Tracker

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# crossing line coordinates
lim = [80, 550, 1250, 550]


# total number of cars count
total_count = []



while True :
    success, img = cap.read()
    
    if not success:
        print("Video finished or cannot load frame.")
        break                                              # exit if there is no frame to read, i.e, the video ends


    img = cv2.resize(img, (1280, 720))                     # resizing the video file 
    mask = cv2.resize(mask, (1280, 720))                   # resizing the mask image 

    img_region = cv2.bitwise_and(img, mask)
    
    # YOLO object detection
    results = model(img_region, stream=True)             # stream=True use generators which is more efficient 


    detections = np.empty((0,5))


    for result in results :
        boxes = result.boxes
        for box in boxes :
            # BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            
            # finding confidence
            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            original_class = classNames[cls]

            # displaying confidence and class name
            #if original_class == 'car' or original_class == 'bus' or original_class == 'truck' or original_class == 'motorcycle' and conf>0.3 :                # only display class or category label if the object detected is a car/truck/bus/bike and have confidence greater than 30%
            if original_class in {'car', 'bus', 'truck', 'motorcycle'} and conf > 0.4:
                #cvzone.putTextRect(img, f'{original_class} {conf}', (max(0, x1), max(35, y1)), scale=0.75, thickness=2, offset =4)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=6, rt=4)
                currentArr = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArr))


    # update tracker
    track_results = tracker.update(detections)

    # drawing the crossing line
    cv2.line(img, (lim[0], lim[1]), (lim[2], lim[3]), (0, 0, 255), 5)


    for result in track_results :
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        # drawing bbox and id
        cvzone.cornerRect(img, (x1, y1, w, h), l=6, rt=4, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset =4)


        # to count the number, check whether the bbox center of cars intersect the red line, if they do then it will be count++
        
        # drawing centers of bbox
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # determining count
        if lim[0]<cx<lim[2] and lim[1]-15<cy<lim[3]+15 :
            if total_count.count(Id) == 0 :                                   # upon just using a count var it coinsiders a region and if a point comes between that region it increases count value, now there can be a single car's center point in that region multiple times and the count value will increase each time that point is detected
                total_count.append(Id)                                      # to resolve this the count list is created and it is checked whether a car with some id has crossed or not and stores it, the length of the list will give the number of cars 
                
                # to signal a count
                cv2.line(img, (lim[0], lim[1]), (lim[2], lim[3]), (0, 255, 0), 5)



    # display count
    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50), colorR=(0, 0, 0))
    
    cv2.imshow("Image", img)

    #cv2.imshow("Image Region", img_region)               
    
    #cv2.waitKey(0)                                         # the video will go forward only if a key is pressed

    # break the loop if 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x') :
        break


print(f"Total number of cars that crossed the line: {len(total_count)}")
