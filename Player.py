import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

def drawBox(img, bbox):
  x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
  cv2.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)
  cv2.putText(img, "Tracking", (120,75), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)

def overlap_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1]) 
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0
def merge_rects(a, b):
    return min(a[0], b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]) 


def draw_connected_components(img):
    # Threshold the image to create a binary image
    # _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


    # Find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    # Create a copy of the original image for drawing rectangles
    img_with_rectangles = np.copy(img)

    bboxes = []

    # Draw rectangles around connected components
    for i in range(1, len(stats)):
        x, y, w, h, _ = stats[i]
        bboxes.append((x, y, w, h))

    bboxes = sorted(bboxes, key = lambda x:-x[2]*x[3])
    while True:
        if len(bboxes) <= 2:
            break
        x, y, w, h = bboxes[1]
        x2,y2,w2,h2 = bboxes[0]
        dist = (((x+w) / 2 - (x2+w2) / 2)**2 + ((y+h) / 2 - (y2+h2) / 2)**2) ** 0.5
        if dist < 150:
            del bboxes[1]
        else:
            break
    return bboxes[:2]


def initialize_trackers(frame, bboxes):
    trackers = cv2.legacy.MultiTracker_create()
    for bbox in bboxes:
          trackers.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)
    return trackers

def is_near(bboxes_1, bboxes_2):
    for bbox_1 in bboxes_1:
        is_near = False
        for bbox_2 in bboxes_2:
            x, y, w, h = bbox_1
            x2,y2,w2,h2 = bbox_2
            dist = (((x+w) / 2 - (x2+w2) / 2)**2 + ((y+h) / 2 - (y2+h2) / 2)**2) ** 0.5
            if dist < 300:
                is_near = True
                break
        if not is_near:
            return False
    return True
            

def get_player_boxes(frames):
    MAX_FRAMES = 10
    LEARNING_RATE = -1   
    fgbg = cv2.createBackgroundSubtractorMOG2()


    trackers = None
    frame_number = 0
    boxes = []
    for frame in frames:
        player_boxes = []
        
        timer = cv2.getTickCount()
        #Apply MOG 
        motion_mask = fgbg.apply(frame, LEARNING_RATE)
        #Get background
        background = fgbg.getBackgroundImage()
        bboxes = draw_connected_components(motion_mask)
        if frame_number == 0:
            frame_number += 1
            continue
        # print(len(bboxes))
        # print(bboxes)
        if trackers is not None:
            success, tracker_bboxes = trackers.update(frame)
            # print(success)
            if success and is_near(tracker_bboxes, bboxes):
                for bbox in tracker_bboxes:
                    drawBox(frame,bbox)
                    player_boxes.append(box)
            else:
                trackers = initialize_trackers(frame, bboxes)
                for box in bboxes:
                    drawBox(frame,box)
                    player_boxes.append(box)
                    
                
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            cv2.putText(frame, str(int(fps)), (120,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        elif len(bboxes) == 2 and trackers is None:
            trackers = initialize_trackers(frame, bboxes)
            for box in bboxes:
                drawBox(frame,box)
                player_boxes.append(box)
        
        frame_number += 1

        boxes.append(player_boxes)

    cv2.destroyAllWindows()
    return boxes


