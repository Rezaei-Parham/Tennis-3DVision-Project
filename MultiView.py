import argparse
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from threading import Thread


class MultiView:
    def __init__(self):
        self.view1points = []
        self.view2points = []
        self.F= None


    def set_initial_correspondences(self,view1,view2):
        """ Set initial correspondences
        :return
            correspondences: list of correspondences
        """
        self.view1points = view1
        self.view2points = view2

    def update_correspondences(self, view1, view2):
        """ Update correspondences
        :return
            correspondences: list of correspondences
        """
        if self.view1points[0].shape != view1[0].shape:
            raise ValueError("The shape of the points must be the same")
        else:
            self.view1points.append(view1)
            self.view2points.append(view2)

    def print_correspondences(self):
        """ Get correspondences
        :return
            correspondences: list of correspondences
        """
        for i in range(len(self.view1points)):
            print(f"view1: {self.view1points[i]} <-> view2: {self.view2points[i]}")

    def visualize_points(self,img1,img2):
        """ write points on the frames with label number
        :params
            frames: list of video frames
            ballpoints: list of ball points
        :return
            frames: list of video frames
        """
        for i in range(len(self.view1points)):
            cv2.circle(img1, (self.view1points[i][0], self.view1points[i][1]), 5, (0, 0, 255), -1)
            cv2.putText(img1, str(i), (self.view1points[i][0], self.view1points[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img2, (self.view2points[i][0], self.view2points[i][1]), 5, (0, 0, 255), -1)
            cv2.putText(img2, str(i), (self.view2points[i][0], self.view2points[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img1, img2
        
    
    def calculate_fundamental_matrix(self):
        """ Calculate fundamental matrix
        :return
            F: fundamental matrix
        """
        F, mask = cv2.findFundamentalMat(np.array(self.view1points), np.array(self.view2points), cv2.FM_RANSAC, 0.1)
        if F.shape != (3, 3):
            raise ValueError("Fundamental matrix is not 3x3, use more points")
        self.F = F
        
    def draw_epipolar_line(self,point1,img2):
        """draw epipolar line of point1 in img2
        :params
            point1: point in img1
            img2: image to draw epipolar line
        """
        lines = cv2.computeCorrespondEpilines(point1.reshape(-1, 1, 2), 1, self.F)
        lines = lines.reshape(-1, 3)
        r, c, _ = img2.shape
        for r in lines:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        return img2