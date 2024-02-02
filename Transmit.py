import argparse
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from threading import Thread

class VideoStream:
    def __init__(self, partial=True):
        self.output_width = None
        self.output_height = None
        self.fixedBack = None
        self.partial = partial
        self.lenFrames = 0


    def read_video(self,path_video, skip_frames=1):
        """ Read video file    
        :params
            path_video: path to video file
            skip_frames: number of frames to skip (to speed up processing)
        :return
            frames: list of video frames
            fps: frames per second
        """
        cap = cv2.VideoCapture(path_video)
        self.output_width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_count % skip_frames == 0:
                    frames.append(frame)
            else:
                break
            frame_count +=1
        cap.release()
        self.lenFrames = len(frames)
        return frames
    
    def image_frame(self,image_path):
        """ Read image file
        :params
            image_path: path to image file
        :return
            fixedBack: stiff background
        """
        self.fixedBack = cv2.imread(image_path)
        self.fixedBack = cv2.resize(self.fixedBack, ( self.output_width, self.output_height ))
    
    def retireve_stiff_background(self):
        """ Retrieve stiff background
        :return
            fixedBack: stiff background
        """
        return np.array([self.fixedBack for _ in range(self.lenFrames)])

    def draw_ball(self, frames, ballpoints):
        """ Draw ball on the frames
        :params
            frames: list of video frames
            ballpoints: list of ball points
        :return
            outframes: list of frames with ball drawn on them
        """
        outframes = [frames[0],frames[1]]
        for i in range(2,len(frames)):
            PIL_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)

            
            if ballpoints[i][0] is not None:
                draw_x, draw_y = ballpoints[i][0],ballpoints[i][1]
                bbox = (draw_x - 6, draw_y - 6, draw_x + 6, draw_y + 6)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline='red', fill='red')
                del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            outframes.append(opencvImage)

        return outframes

    def draw_people(self, frames,balladded, peoplepoints):
        """ 
        """
        newFrames = []
        for i in range(len(frames)):
            img = balladded[i]
            if i == 0:
                newFrames.append(img)
                continue
            for box in peoplepoints[i-1]:
                # print(box)
                img[int(box[0]):int(box[0]+box[2]),int(box[1]):int(box[1]+box[3])] = frames[i][int(box[0]):int(box[0]+box[2]),int(box[1]):int(box[1]+box[3])]
            newFrames.append(img)
        return newFrames
    
    def output_video(self,final_frames,name='output.mp4'):
        """ Preset output video
        :params
            f1: first frame of the video
            f2: second frame of the video
        :return
            output_video: video object to write frames to the output video file. 
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(name,fourcc, 30.0, (self.output_width,self.output_height))

        for f in final_frames:
            output_video.write(f)
        output_video.release()
    
    def transmit(self,path_video,image_path,balls,people,name='output.mp4'):
        self.image_frame(image_path)
        frames = self.retireve_stiff_background()
        frames_with_ball = self.draw_ball(frames, balls)
        frames_with_people = self.draw_people(frames, frames_with_ball, people)
        self.output_video(frames_with_people,name)
        # del self.fixedBack
        del frames
        del frames_with_ball
        del frames_with_people