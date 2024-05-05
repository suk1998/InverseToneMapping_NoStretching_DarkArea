"""
A code to completely darken  dark areas without applying inverse tone mapping.

In a scene where fireworks explode at night, applying tone mapping to the fireworks while no applying tone mapping to the night sky is inteded for.

2023/04/04
"""
import sys
import cv2

import numpy as np

# Multiprocessing Library
import pypardiso
import scipy.sparse as sparse

import os

# WLS filter Library
from scipy.sparse import spdiags, linalg
from scipy.sparse.linalg import spsolve, lsqr

import time
start = time.time()

# WLS Filter Function(this part already exists.)

def wlsFilter(IN, Lambda=1.0, Alpha=1.2):
    L = np.log(IN+1e-22)
    smallNum = 1e-6
    height, width = IN.shape
    k = height * width

    dy = np.diff(L, n=1, axis=0)
    print("dy.shape",dy.shape)
    dy = -Lambda/(np.abs(dy)**Alpha + smallNum)
    print("dy.shape",dy.shape)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')
    print("dy.shape",dy.shape)
    dy = dy.flatten(order='F')
    print("dy.shape",dy.shape)

    dx = np.diff(L, n=1, axis=1)

    dx = -Lambda/(np.abs(dx)**Alpha + smallNum)
    dx = np.pad(dx, ((0,0),(0,1)), 'constant')
    dx = dx.flatten(order='F')
    print("dx.shape",dx.shape)

    B = np.concatenate([[dx],[dy]], axis=0)
    d = np.array([-height, -1])
    print("B.shape",B.shape)
    print("d.shape",d.shape)

    A = spdiags(B, d, k, k)

    e = dx
    w = np.pad(dx, (height, 0), 'constant'); w = w[0:-height]
    s = dy
    n = np.pad(dy, (1,0), 'constant'); n = n[0:-1]

    D = 1.0 -(e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, k, k)
    print("A.shape",A.shape)
    A = A.tocsc()
    
    b = IN.flatten(order='F')
    
    # This part is the most computationally intensive section
    OUT = pypardiso.spsolve(A, b)
    
    print("OUT.shape",OUT.shape)
    return np.reshape(OUT, (height, width), order='F')

# a function added to make a smooth gradients(4/04/2024)

def sigmoid(x, a=1):
        return 1/(1+np.exp(-a*-x))


# Virtual illumination generation(VIG)
def scale_fun(v_, mean_i_, max_i_):
    r = 1.0 - mean_i_/max_i_
    fv = lambda v : r*(1/(1+np.exp(-1.0*(v - mean_i_))) - 0.5)

    fv_k_ = [fv(vk) for vk in v_]
    return fv_k_

def VIG(illuminance, inv_illuminance):
    inv_illuminance /= np.max(inv_illuminance)
    
    mi = np.mean(illuminance)
    maxi = np.max(illuminance)
    # v1 represents the value of the darkest area, v5 represents the value of the brightest area, v2 and v4 are set as intermediate values, and v3 corresponds to the average brightness value of the original image.
    v1 = 0.8
    v3 = mi
    v2= 0.5 * (v1+v3)
    v5 = 0.8
    v4 = 0.5*(v3+v5)
    v = [v1, v2, v3, v4, v5]
    fvk_list = scale_fun(v, mi, maxi)

    I_k = [(1+fvk) * (illuminance + fvk * inv_illuminance) for fvk in fvk_list]

    soft_threshold = 0.15

    I_K = [sigmoid(10*(Ik-soft_threshold))*Ik for Ik in I_k]

    return I_k

# Tonemapping
def tonereproduct(bgr_image, L, R_, Ik_list, FLAG):
    
    Lk_list = [ np.exp(R_) * Ik for Ik in Ik_list ] 
    L = L + 1e-22 

    rt = 1.0
    threshold = 0.1
    not_dark = L>threshold
    b, g, r = cv2.split(bgr_image)
    #b, g, r = bgr_image[:,:,0], bgr_image[:,:,1], bgr_image[:,:,2]
    # Reconstruct the color image by combining the channels.
    if FLAG == False:
        Sk_list = [cv2.merge((Lk*(b/L)**rt, Lk*(g/L)**rt, Lk*(r/L)**rt)) for Lk in Lk_list]
        return Sk_list[2]
    else:  # Weight maps

        Wk_list = []
        for index, Ik in enumerate(Ik_list):
            if index < 3:
                wk = Ik / np.max(Ik)
            else:
                temp = 0.5*(1 - Ik)
                wk = temp / np.max(temp)
            Wk_list.append(wk)

        A = np.zeros_like(Wk_list[0])
        B = np.zeros_like(Wk_list[0])
        for lk, wk in zip(Lk_list, Wk_list):
            A = A + lk * wk 
            B = B + wk

        L_ = (A/B)
        ratio = np.clip(L_/L, 0, 3) # Clipping unreasonable values
        brightness_threshold = 0.4
        alpha = 10 # Adjust the steepness of the sigmoid function

        # Calculate the sigmoid function for the smooth transition between dark and bright weights
        transition = sigmoid(L - brightness_threshold, alpha)
        
        dark_weight = 1-transition
        bright_weight = transition
      
        b_ = np.where(not_dark, (dark_weight * ratio + bright_weight) * b, b)
        g_ = np.where(not_dark, (dark_weight * ratio + bright_weight) * g, g)
        r_ = np.where(not_dark, (dark_weight * ratio + bright_weight) * r, r)

        
        out = cv2.merge( ( b_, g_, r_ ) )
        return np.clip(out, 0.0, 1.0)

# Stretching the pixel whose illuminance is brighter than mean value

def SRS(reflectance, illuminance):
    # When the value of r_R is low, the stretch decreases, and when it's high, the stretch increases.
    r_R = 0.5
    threshold = 0.1
    mean_I = np.mean(illuminance)
    def compare_func(r, i):
        if i <= mean_I:
            return r
        else:
            return r * (i/mean_I)**r_R

    srs_fun = np.vectorize(compare_func)
    result = srs_fun(reflectance, illuminance)
    return result

class HDRiTMO():
    def __init__(self, flag):
        self.weighted_fusion = flag
        self.wls = wlsFilter
        self.srs = SRS
        self.vig = VIG
        self.tonemap = tonereproduct
    
    def process(self, image):
        
        if image.shape[2] == 4:
            image = image[:,:, 0:3]
        
        S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        #S = np.dot(image[...,:3],[0.299, 0.587,0.114])SR
        image = 1.0*image/255
        L = 1.0*S

        I = self.wls(S)
        R = np.log(L+1e-22)-np.log(I+1e-22)
        R_ = self.srs(R, L)
        I_K = self.vig(L, 1.0 - L)

        result_ = self.tonemap(image, L, R_, I_K, self.weighted_fusion)
        return result_

################## Display and save the resulting image. ####################

if __name__ == '__main__':
    input_folder = './images'
    output_folder = './hdr_result_12'
    image_files = os.listdir(input_folder)

    HDR_filter = HDRiTMO(True)

    for image_file in image_files:
        # Check if the file is an image
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            print(f"Processing {image_path}")
            
            image = cv2.imread(image_path)
            print("image.shape", image.shape)
            
            output_image = HDR_filter.process(image)
            end = time.time()
            print(f"{end-start:.5f}sec")
            print("output_image.shape", output_image.shape)
            print(output_image)

            # Save the result
            output_path = os.path.join(output_folder, f'HDR_{image_file}')
            cv2.imwrite(output_path, 255 * output_image)
   
    