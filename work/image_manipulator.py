import numpy as np       
import cv2
import skimage.segmentation
from skimage import img_as_ubyte, exposure
import os
from imgaug import augmenters
import work.custom_logger as cl
import random

logger = cl.get_logger()

def view_image(_im):
    cv2.imshow("view", _im)
    cv2.waitKey(0)

def view_images(_im1, _im2):
    cv2.imshow("before", _im1)
    cv2.imshow("after", _im2)
    cv2.waitKey(0)

def felzenszwalb(_image, _k):
    
    res = skimage.segmentation.felzenszwalb(_image, scale = _k)
    return cv2.cvtColor(img_as_ubyte(exposure.rescale_intensity(res)), cv2.COLOR_GRAY2BGR)
    

def remove_background(_img):
    
    # img_file = 'C:/datasets/COI/v2/baza/modeling_v2/all_images_heat/heat_resized_out_shape_114_2065_22.png'
    # _img = cv2.imread(img_file)

    bg_color = _img[1, 1] 
    mask = cv2.inRange(_img, bg_color, bg_color)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv_3ch = cv2.merge([mask_inv] * 3)
    image_no_bg = cv2.bitwise_and(_img, mask_inv_3ch)
    image_rgb = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2RGB)
    
    # plt.imshow( image_rgb   )
    
    return image_rgb

def drawMask(_image, _cnts, fill=True):
    image = np.array(_image)
    markers = np.zeros((image.shape[0], image.shape[1]))
    heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    t = 2
    if fill:
      t = -1
    cv2.drawContours(markers, _cnts, -1, (255, 0, 0), t)
    mask = markers>0
    image[mask,:] = heatmap_img[mask,:]
    return image


def heatmap(_image):

    return cv2.applyColorMap(_image, cv2.COLORMAP_RAINBOW)
    
def edges(_image, _lower, _upper):
    img = cv2.Canny(_image, _lower, _upper, L2gradient = True )
    return img
    
def sobel(_im, _dx=1, _dy=1, _ksize=3):
    
    gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    res = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return res# cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)


def view_images_matrix(_im_list):
    res = cv2.vconcat([cv2.hconcat(list_h) for list_h in _im_list])
    # image resizing
    res = cv2.resize(res, dsize = (0,0), fx = 0.3, fy = 0.3)

    # show the output image
    cv2.imshow('images', res)
    cv2.waitKey(0)
        

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def get_images_matrix(_im_list):

    return vconcat_resize_min(_im_list)
        
    
def dilate(_im):
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (70,1))
    res = cv2.dilate(gray, horizontal_kernel, iterations=1)

    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR) 

def bw_mask(_im):
    # convert to gray
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    # mask = np.zeros(_im.shape, dtype=np.uint8)  
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # res = cv2.bitwise_and(_im, _im, mask = mask)
    # #res[mask==0] = (255,255,255)
    (thresh, res) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #thresh = 127
    #res = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return res 

def threshold(_im):
    
    # convert to gray
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    
    # create mask
    #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    res = _im.copy()
    res[mask == 255] = (255,255,255)
    
    return res 

def blur(_im):
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
        
    # create mask
    thresh = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY)[1]
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    blur = cv2.blur(_im,(5,5),0)
    res = _im.copy()
    res[mask>0] = blur[mask>0]

    #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    return res
    
def blur_manual(_im, _blur_range=1, _border=5):
    
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im

    
    # create mask
    thresh = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY)[1]
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    res = _im.copy()
    res[mask == 255] = (255,255,255)
    
    w_min = _border
    w_max = _im.shape[1] - _border
    
    h_min = _border
    h_max = _im.shape[0] - _border 
    
    if len(_im.shape) == 3:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    
    for row in range(h_min, h_max):
        for col in range(w_min, w_max):
            if res[row][col] == 255:
                tmp_arr = res[row -_blur_range:row + _blur_range, col - _blur_range:col + _blur_range][res[row - _blur_range:row + _blur_range, col - _blur_range:col + _blur_range] < 255]
              
                if tmp_arr.size>0:
                    tmp = np.mean(tmp_arr)
                else:
                    tmp = np.mean(res)
                    
                res[row][col] = round(np.mean(tmp))
      
    return res 


def draw_angle():
    if random.random() < 0.5:
        return random.uniform(-180, -10)  # Range 1: -180 to -10
    else:
        return random.uniform(10, 180)    # Range 2: 10 to 180

def augment(_img):
    angle = draw_angle()

    # Get image size
    (h, w) = _img.shape[:2]

    # Get background color from pixel (1,1)
    bg_color = _img[1, 1].tolist()

    # Compute the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation with background fill
    rotated_img = cv2.warpAffine(_img, rotation_matrix, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    
    return rotated_img

def extract_image(_image, _image_path):
    IMG_BORDER = 10
    raw_image = cv2.imread(_image_path)
    
    height, width = _image.shape[:2]
    hsv = _image#cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([120, 120, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    cv2.polylines(_image, [contour], isClosed=True, color=(0, 0, 255), thickness=2)
    
    hsv = _image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    bw_mask = np.zeros_like(raw_image)
    cv2.fillPoly(bw_mask, pts=[contour], color=(255, 255, 255))
    raw_image = cv2.bitwise_and(raw_image, bw_mask)

    x, y, w, h = cv2.boundingRect(contour)
    y1 = y - IMG_BORDER if y - IMG_BORDER > 0 else 0
    y2 = y + h + IMG_BORDER if y + IMG_BORDER + h < height else height
    x1 = x - IMG_BORDER if x - IMG_BORDER > 0 else 0
    x2 = x + w + IMG_BORDER if x + IMG_BORDER + w < width else width
    
    return raw_image[y1:y2, x1:x2]


def resize_with_aspect_ratio(_image, width=None, height=None):
    (h, w) = _image.shape[:2]

    if width is None and height is None:
        return _image

    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(_image, dim, interpolation=cv2.INTER_AREA)

    # Check if we need to add padding
    delta_w = width - resized.shape[1]
    delta_h = height - resized.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    top = 0 if top < 0 else top
    bottom = 0 if bottom < 0 else bottom
    left = 0 if left < 0 else left
    right = 0 if right < 0 else right
    
    color = [0, 0, 0]  # Black padding
    resized_with_padding = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    resized_with_padding = cv2.resize(resized_with_padding, (width, height), interpolation=cv2.INTER_AREA)
    return resized_with_padding
    
def extract_and_resize_images(_input_path, _output_path, _raw_path, _width, _height):
    res = 0
    for f in os.listdir(_input_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            raw_image = filename.replace("out_shape_", "")            
            logger.info(f"processing {f} to {raw_image}")

            image = cv2.imread(_input_path + f)
            image = extract_image(image, _raw_path + raw_image)
            image = blur_manual(image, 2, 1)
            resized = resize_with_aspect_ratio(image, _width, _height)
            gray = resized #cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(_output_path+'resized_' + f, gray)

            res = res + 1
    return res

def resize_images(_input_path, _output_path, _width, _height):
    
    res = 0
    for f in os.listdir(_input_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            image = cv2.imread(_input_path+f)
            image = blur_manual(image, 2, 1)
            resized = cv2.resize(image, (_width, _height), interpolation = cv2.INTER_AREA)
            gray = resized #cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(_output_path+'resized_' + f, gray)
            res = res + 1
    return res
