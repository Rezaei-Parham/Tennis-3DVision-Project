import argparse
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from threading import Thread


class MultiView:
    def __init__(self):
        self.viewPoints = [[],[],[]]
        self.F1 = None
        self.F2 = None


    def set_initial_correspondences(self,view1,view2,view3=None):
        """ Set initial correspondences
        :return
            correspondences: list of correspondences
        """
        self.viewPoints[0] = view1
        self.viewPoints[1] = view2
        self.viewPoints[2] = view3
        self.calculate_fundamental_matrix()

    def update_correspondences(self, view1, view2, view3=None):
        """ Update correspondences
        :return
            correspondences: list of correspondences
        """
        if self.view1points[0][0].shape != view1[0].shape:
            raise ValueError("The shape of the points must be the same")
        else:
            self.viewPoints[0].append(view1)
            self.viewPoints[1].append(view2)
            if view3 is not None:
                self.viewPoints[2].append(view3)
        
        self.calculate_fundamental_matrix()

    def print_correspondences(self):
        """ Get correspondences
        :return
            correspondences: list of correspondences
        """
        for i in range(len(self.viewPoints[0])):
            print(f"view1: {self.view1points[0][i]} <-> view2: {self.view2points[1][i]}")

    def visualize_points(self,img1,img2,img3=None):
        """ write points on the frames with label number
        :params
            frames: list of video frames
            ballpoints: list of ball points
        :return
            frames: list of video frames
        """
        for i in range(len(self.view1points)):
            cv2.circle(img1, (self.viewPoints[0][i][0], self.viewPoints[0][i][1]), 5, (0, 0, 255), -1)
            cv2.putText(img1, str(i), (self.viewPoints[0][i][0], self.viewPoints[0][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img2, (self.viewPoints[1][i][0], self.viewPoints[1][i][1]), 5, (0, 0, 255), -1)
            cv2.putText(img2, str(i), (self.viewPoints[1][i][0], self.viewPoints[1][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.viewPoints[2]) > 0 and img3 is not None:
                cv2.circle(img3, (self.viewPoints[2][i][0], self.viewPoints[2][i][1]), 5, (0, 0, 255), -1)
                cv2.putText(img3, str(i), (self.viewPoints[2][i][0], self.viewPoints[2][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img1, img2, img3
        
    
    def calculate_fundamental_matrix(self):
        """ Calculate fundamental matrix
        :return
            F: fundamental matrix
        """
        F1, mask = cv2.findFundamentalMat(np.array(self.viewPoints[0]), np.array(self.viewPoints[1]), cv2.FM_RANSAC, 0.1)
        if F1.shape != (3, 3):
            raise ValueError("Fundamental matrix 2 to 1 is not 3x3, use more points")
        if len(self.viewPoints[2]) > 0:
            F2, mask = cv2.findFundamentalMat(np.array(self.viewPoints[0]), np.array(self.viewPoints[2]), cv2.FM_RANSAC, 0.1)
            if F2.shape != (3, 3):
                raise ValueError("Fundamental matrix 3 to 1 is not 3x3, use more points")
            self.F2 = F2
        self.F1 = F1
        
    def draw_epipolar_line(self,point1,img2):
        """draw epipolar line of point1 in img2
        :params
            point1: point in img1
            img2: image to draw epipolar line
        """
        lines = cv2.computeCorrespondEpilines(point1.reshape(-1, 1, 2), 1, self.F1)
        lines = lines.reshape(-1, 3)
        r, c, _ = img2.shape
        for r in lines:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        return img2
    
    def get_epipolar_line(self,point1, view):
        """get the epipolar line
        :params
            point1: point in image1
        :return
            line: the epipolar line"""
        if view==1:
            return cv2.computeCorrespondEpilines(point1.reshape(-1, 1, 2),1,self.F1).reshape(-1,3)
        else:
            return cv2.computeCorrespondEpilines(point1.reshape(-1, 1, 2), 1, self.F2).reshape(-1, 3)
    

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
    
    def estimate_from_two_views(self, ballView1, ballView2):
        """ Estimate ball position in new frame
        :params
            ballView1: ball position in view1
            ballView2: ball position in view2
        :return
            intersection: intersection of epipolar line and line from lastBall and preLastBall
        """
        view1line = self.get_epipolar_line(ballView1)
        view2line = self.get_epipolar_line(ballView2)
        intersection = self.intersection(view1line.ravel(), view2line.ravel())
        return intersection
    
    def draw_estimation_from_other_views(self, img, ballView1, ballView2):
        point = self.estimate_from_two_views(ballView1, ballView2)
        # draw point on img1
        PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(PIL_image)

        if point[0] is None or point[1] is None:
            raise ValueError("Point is None")
        else:
            draw_x, draw_y = point[0], point[1]
            bbox = (draw_x - 4, draw_y - 4, draw_x + 4, draw_y + 4)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='blue', fill='blue')
            del draw

        opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", opencvImage)
        