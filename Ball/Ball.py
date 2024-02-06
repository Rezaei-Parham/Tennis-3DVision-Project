import argparse
import Ball.Models as Models
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw

class BallDetector:
    def __init__(self, path_weights=None, n_classes =  256, device='mps'):
        self.n_classes = n_classes
        self.width , self.height = 640, 360
        # model definition
        modelFN = Models.TrackNet.TrackNet
        m = modelFN(n_classes,input_height=self.height,input_width=self.width)
        m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
        m.load_weights(path_weights)
        self.output_width = 0
        self.output_height = 0
        self.model = m
        self.device = device

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
        """ Read image file
        :params
            frames: list of video frames
            image_path: path to image file
        :return
            f: list of image frames with the same size as the input frames. 
        """
        backImage = cv2.imread(image_path)
        f = np.array([backImage for _ in range(len(frames))])
        return f
    
    def resize_frames(self,frames, width, height):
        """ Resize frames to the input size of the model
        :params
            frames: list of video frames
            width: width of the resized frames
            height: height of the resized frames
        :return
            f: list of resized frames
        """
        f = []
        for frame in frames:
            new_frame = cv2.resize(frame, ( width , height ))
            new_frame = new_frame.astype(np.float32)
            f.append(new_frame)
        return f

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
    
    def find_ball_points(self, frames):
        """ Find the ball points in the video
        :params
            frames: list of video frames
        :return
            ballpoints: list of ball points in the video (x,y) coordinates. 
        """
        frames = self.resize_frames(frames, self.width, self.height)

        ballpoints = [[None,None],[None,None]]
        for i in range(2, len(frames)):
            X = np.concatenate((frames[i], frames[i - 1], frames[i - 2]), axis=2)
            X = np.rollaxis(X, 2, 0)
            pr = self.model.predict(np.array([X]))[0]
            pr = pr.reshape((self.height, self.width, self.n_classes)).argmax(axis=2)
            pr = pr.astype(np.uint8)
            heatmap = cv2.resize(pr, (self.output_width, self.output_height))
            ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)


            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

            x, y = None, None
            if circles is not None and len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

            ballpoints.append([x, y])

        return ballpoints
    
    def draw_ball(self, frames, ballpoints, name='output.mp4'):
        output_video = self.preset_output_video(frames[0], frames[1],name)
        for i in range(2,len(frames)):
            PIL_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)

            
            if ballpoints[i][0] is not None:
                draw_x, draw_y = ballpoints[i][0],ballpoints[i][1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline='red', fill='red')
                del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            output_video.write(opencvImage)

        output_video.release()

    def draw_trajectory_tracknetStyle(self, frames, ballpoints, name='output.mp4'):
        q = queue.deque()
        for i in range(0,8):
            q.appendleft(None)
        output_video = self.preset_output_video(frames[0], frames[1],name)
        for i in range(2,len(frames)):
            PIL_image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)
            q.pop()
            q.appendleft([ballpoints[i][0],ballpoints[i][1]])
            for i in range(0,8):
                if q[i] is not None:
                    draw_x = q[i][0]
                    draw_y = q[i][1]
                    bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                    draw = ImageDraw.Draw(PIL_image)
                    draw.ellipse(bbox, outline ='red')
                    del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            output_video.write(opencvImage)

        output_video.release()

    def draw_trajectory_pride(self, frames, ballpoints, name='output.mp4'):
        output_video = self.preset_output_video(frames[0], frames[1], name)
        # Define a queue to keep track of the last 8 points
        points_queue = queue.deque(maxlen=8)

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
                    size_factor = max(1, size_factor/2)  # Ensure size is at least 1
                    bbox = (draw_x - size_factor, draw_y - size_factor, draw_x + size_factor, draw_y + size_factor)
                    # Use the j-th color from the colors list
                    draw.ellipse(bbox, outline=colors[j], fill=colors[j])

            del draw

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            output_video.write(opencvImage)

        output_video.release()

    def draw_trajectory(self, frames, ballpoints, name='output.mp4', style='simple'):
        if style == 'simple':
            return self.draw_ball(frames, ballpoints, name)
        elif style == 'tracknet':
            return self.draw_trajectory_tracknetStyle(frames, ballpoints, name)
        elif style == 'pride':
            return self.draw_trajectory_pride(frames, ballpoints, name)
        else:
            print('Invalid style')
            return None

    def interpolate_missing_points(self,ballpoints):
        for i in range(len(ballpoints)):
            if ballpoints[i] == [None, None]:
                prev_index = next((j for j in range(i-1, -1, -1) if ballpoints[j] != [None, None]), None)
                next_index = next((j for j in range(i+1, len(ballpoints)) if ballpoints[j] != [None, None]), None)

                if prev_index is not None and next_index is not None:
                    # Interpolate using both the previous and next valid points
                    ballpoints[i] = [(ballpoints[prev_index][0] + ballpoints[next_index][0]) / 2,
                                    (ballpoints[prev_index][1] + ballpoints[next_index][1]) / 2]
                elif prev_index is not None:
                    # Use the previous point if no next valid point is available
                    ballpoints[i] = ballpoints[prev_index]
                elif next_index is not None:
                    # Use the next point if no previous valid point is available
                    ballpoints[i] = ballpoints[next_index]
        return ballpoints
    
    def interpolate_far_points(self,ballpoints, max_dist=50):
        for i in range(1, len(ballpoints)):
            if ballpoints[i] is not None and ballpoints[i - 1] is not None:
                dist = np.linalg.norm(np.array(ballpoints[i]) - np.array(ballpoints[i - 1]))
                if dist > max_dist:
                    # Interpolate
                    if i < len(ballpoints) - 1 and ballpoints[i + 1] is not None:
                        # Interpolate using the next point if it's not None
                        ballpoints[i] = [(ballpoints[i - 1][0] + ballpoints[i + 1][0]) / 2,
                                        (ballpoints[i - 1][1] + ballpoints[i + 1][1]) / 2]
                    else:
                        # Interpolate using just the previous point if the next point is None
                        ballpoints[i] = ballpoints[i - 1]
        return ballpoints
    
    def postprocess_points(self,ballpoints,interpolate_none=True,interpolate_far=True,max_dist=50):
        balls = ballpoints
        if interpolate_none:
            balls = self.interpolate_missing_points(balls)
        if interpolate_far:
            balls = self.interpolate_far_points(balls,max_dist)
        return balls
    


# bd = BallDetector(path_weights='weights/model.3')
# frames, fps = bd.read_video('test.mp4')
# ballpoints = bd.find_ball_points(frames)
# ballpoints = bd.postprocess_points(ballpoints)
# bd.draw_trajectory_pride(frames, ballpoints, 'test1output.mp4')
