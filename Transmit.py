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
        self.frames = []


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
        self.frames = frames
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

    def draw_ball(self, frames, ballpoints, realFrames):
        """ Draw ball on the frames
        :params
            frames: list of video frames
            ballpoints: list of ball points
        :return
            outframes: list of frames with ball drawn on them
        """
        outframes = [frames[0],frames[1]]
        for i in range(2,len(frames)):
            newframe = frames[i]
            
            if ballpoints[i][0] is not None:
                l,u,r,d = self.get_big_box(ballpoints[i][0],ballpoints[i][1],5,5,8)
                newframe[u:d,l:r] = realFrames[i][u:d,l:r]

            
            outframes.append(newframe)

        return outframes
    
    def get_big_box(self,x,y,h,w,pad=30):
        x = int(x)
        y = int(y)
        h = int(h)
        w = int(w)
        left = x - pad
        if left < 0:
            left = 0
        right = x + h + pad
        if right > self.output_width:
            right = self.output_width
        up = y - pad
        if up < 0:
            up = 0
        down = y + w + pad
        if down > self.output_height:
            down = self.output_height
        return left,up,right,down

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
                l,u,r,d = self.get_big_box(box[0],box[1],box[2],box[3])
                img[u:d,l:r] = frames[i][u:d,l:r]
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
    
    def transmit(self,image_path,balls,people,name='output.mp4'):
        self.image_frame(image_path)
        frames = self.retireve_stiff_background()
        frames_with_ball = self.draw_ball(frames, balls,self.frames)
        frames_with_people = self.draw_people(self.frames, frames_with_ball, people)
        self.output_video(frames_with_people,name)
        # del self.fixedBack
        del frames
        del frames_with_ball
        del frames_with_people

    
    def cute_detail_ball(self,ballpoints):
        frames = self.frames
        points_queue = queue.deque(maxlen=8)
        outframes = [frames[0],frames[1]]
        for i in range(2,len(frames)):
            PIL_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)
            draw = ImageDraw.Draw(PIL_image)

            # Update the queue with the current point
            points_queue.appendleft(ballpoints[i])

            # Define a list of colors for the 8 points
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black']

            # Iterate over the points in the queue
            for j, point in enumerate(points_queue):
                if point[0] is not None:
                    draw_x, draw_y = point
                    # Adjust the size of the point based on its position in the queue
                    size_factor = 6 - j  # Size decreases for older points in the queue
                    size_factor = max(1, size_factor)  # Ensure size is at least 1
                    bbox = (draw_x - size_factor, draw_y - size_factor, draw_x + size_factor, draw_y + size_factor)
                    # Use the j-th color from the colors list
                    draw.ellipse(bbox, outline=colors[j], fill=colors[j])

            del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            outframes.append(opencvImage)
        return outframes


    def cute_detail_player(self, players, ballFrames):
        frames = ballFrames
        outframes = []
        for i in range(2, len(frames)):
            PIL_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)
            draw = ImageDraw.Draw(PIL_image)

            for player in players[i-2]:
                draw_x, draw_y, draw_w, draw_h = player
                draw.rectangle((draw_x, draw_y, draw_x + draw_w, draw_y + draw_h), outline='red', fill='red')

            del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            outframes.append(opencvImage)
        return outframes
    
    def cute_detail_court(self,coutPoints, playerBallFrames):
        #TODO
        pass
    
    def show_details_on_frame(self,balls,players,court=None,name='outputComplete.mp4'):
        bf = self.cute_detail_ball(balls)
        pbf = self.cute_detail_player(players, bf)
        # TODO: Add court here
        self.output_video(pbf, name)

    def delete_redundancy(self):
        del self.frames