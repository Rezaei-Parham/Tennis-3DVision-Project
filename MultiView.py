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
        self.F = None


    def set_initial_correspondences(self,view1,view2):
        """ Set initial correspondences
        :return
            correspondences: list of correspondences
        """
        self.view1points = view1
        self.view2points = view2
        self.calculate_fundamental_matrix()

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
        self.calculate_fundamental_matrix()

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
    
    def get_epipolar_line(self,point1):
        """get the epipolar line
        :params
            point1: point in image1
        :return
            line: the epipolar line"""
        return cv2.computeCorrespondEpilines(point1.reshape(-1, 1, 2),1,self.F).reshape(-1,3)
    

    def line_from_points(self,p1, p2):
        """ Calculate line coefficients A, B, C from two points (x1, y1) and (x2, y2) """
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0]*p1[1] - p1[0]*p2[1]
        return A, B, -C

    def intersection(self,line1, line2):
        """ Find intersection of two lines given by coefficients A, B, C """
        A1, B1, C1 = line1
        A2, B2, C2 = line2

        determinant = A1*B2 - A2*B1
        if determinant == 0:
            return None  # Lines are parallel

        x = (C1*B2 - C2*B1) / determinant
        y = (A1*C2 - A2*C1) / determinant
        return x, y
    
    def estimate_ball_position(self, lastBall, preLastBall, view2Ball):
        """ Estimate ball position in new frame
        :params
            lastBall: last ball position
            preLastBall: pre-last ball position
            view2Ball: ball position in view2
        :return
            intersection: intersection of epipolar line and line from lastBall and preLastBall
        """
        epipolarLine = self.get_epipolar_line(view2Ball)
        points_line = self.line_from_points(preLastBall, lastBall)
        intersection = self.intersection(points_line, epipolarLine.ravel())
        return intersection
    
    def estimate_from_two_epipolars(self, view1line, view2line):
        """ Estimate ball position in new frame
        :params
            view1line: epipolar line in view1
            view2line: epipolar line in view2
        :return
            intersection: intersection of epipolar line and line from lastBall and preLastBall
        """
        intersection = self.intersection(view1line.ravel(), view2line.ravel())
        return intersection

        