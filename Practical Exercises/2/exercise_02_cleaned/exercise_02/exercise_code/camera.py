import numpy as np
from math import tan, atan
from abc import ABC, abstractmethod

def compute_relative_pose(pose_1,pose_2):
    '''
    Inputs:
    - pose_i transform from cam_i to world coordinates, matrix of shape (3,4)
    Outputs:
    - pose transform from cam_1 to cam_2 coordinates, matrix of shape (3,4)
    '''

    ########################################################################
    # TODO:                                                                #
    # Compute the relative pose, which transform from cam_1 to cam_2       #
    # coordinates.                                                         #
    ########################################################################


    R1, t1 = pose_1[:, :3], pose_1[:, 3]
    R2, t2 = pose_2[:, :3], pose_2[:, 3]

    R_rel = R2.T @ R1
    t_rel = R2.T @ (t1 - t2)

    pose = np.zeros((3, 4))
    pose[:, :3] = R_rel
    pose[:, 3] = t_rel

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return pose



class Camera(ABC):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @abstractmethod
    def project(self, pt):
        """Project the point pt onto a pixel on the screen"""
        
    @abstractmethod
    def unproject(self, pix, d):
        """Unproject the pixel pix into the 3D camera space for the given distance d"""


class Pinhole(Camera):

    def __init__(self, w, h, fx, fy, cx, cy):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the pinhole model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the pinhole model.                 #
        ########################################################################

        pt = np.asarray(pt).reshape(3,)
        pt_cam = pt / pt[2]  # Normalize to get [x/z, y/z, 1]
        pix_hom = self.K @ pt_cam
        pix = pix_hom[:2]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the pinhole model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the pinhole#
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        x = (pix[0] - self.K[0, 2]) / self.K[0, 0]
        y = (pix[1] - self.K[1, 2]) / self.K[1, 1]
        vec = np.array([x, y, 1.0])
        vec = vec / np.linalg.norm(vec)  # Unit vector direction
        final_pt = d * vec
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return final_pt


class Fov(Camera):

    def __init__(self, w, h, fx, fy, cx, cy, W):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.W = W
        

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the Fov model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the Fov model.                     #
        ########################################################################
        
        pt = np.asarray(pt).reshape(3,)
        x = pt[0] / pt[2]
        y = pt[1] / pt[2]
        r = np.sqrt(x**2 + y**2)

        if r > 1e-8:
            theta = np.arctan(2 * r * np.tan(self.W / 2)) / self.W
            x_d = (theta / r) * x
            y_d = (theta / r) * y
        else:
            # r very small, avoid division by zero
            x_d = x
            y_d = y

        u = self.K[0, 0] * x_d + self.K[0, 2]
        v = self.K[1, 1] * y_d + self.K[1, 2]

        pix = np.array([u, v])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the Fov model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the FOV    #
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################
        u = pix[0]
        v = pix[1]
        x_d = (u - self.K[0, 2]) / self.K[0, 0]
        y_d = (v - self.K[1, 2]) / self.K[1, 1]
        r_d = np.sqrt(x_d**2 + y_d**2)

        if r_d > 1e-8:
            # invert theta = arctan(2 r tan(W/2)) / W to get r
            theta = r_d * self.W
            r = np.tan(theta) / (2 * np.tan(self.W / 2))
            scale = r / r_d
            x = x_d * scale
            y = y_d * scale
        else:
            x = x_d
            y = y_d

        vec = np.array([x, y, 1.0])
        vec = vec / np.linalg.norm(vec)
        final_pt = d * vec


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return final_pt
