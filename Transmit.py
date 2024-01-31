import argparse
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from threading import Thread

class VideoStream:
    def __init__(self, src=0, width=640, height=480, partial=False):
        self.output_width = width
        self.output_height = height
        self.frames = None
        self.fixedBack = None
        self.out_video = None


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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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
        return frames, fps
    
    
    def image_frame(self,frames,image_path):
        self.fixedBack = cv2.imread(image_path)
    
    
    def retireve_stiff_background(self):
        return np.array([self.fixedBack for _ in range(len(self.frames))])
    

    def preset_output_video(self,f1,f2,name='output.mp4'):
        """ Preset output video
        :params
            f1: first frame of the video
            f2: second frame of the video
        :return
            output_video: video object to write frames to the output video file. 
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(name,fourcc, 30.0, (self.output_width,self.output_height))
      
        output_video.write(f1)
        output_video.write(f2)
        return output_video


    
    def draw_ball(self, frames, ballpoints, name='output.mp4'):
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

    def draw_people(self, frames,balladded, peoplepoints, name='output.mp4'):
        outframes = [frames[0],frames[1]]
        # peoplepoints each
        newFrames = []
        for i in range(len(frames)):
            img = balladded[i]
            box = peoplepoints[i]
            img[box[0][0]:box[0][1],box[1][0]:box[1][1]] = frames[i][box[0][0]:box[0][1],box[1][0]:box[1][1]]
            newFrames.append(img)
        return newFrames