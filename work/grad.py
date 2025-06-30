import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

import pandas as pd
import numpy as np
import work
import work.models as m
import work.data as d
import work.globals as g
import work.custom_logger as cl
import work.reports as r
import work.image_manipulator as im
import tensorflow as tf
from datetime import datetime
import keras
import random
from datetime import timedelta
import importlib
import timeit
import absl.logging
import cv2
import glob
import os
import re

logger = cl.get_logger()
              

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def grad_model_iterator():
    INPUT_PATH = g.IMG_PATH_BASE
      
    file_list = glob.glob("models/*")
    pos = 0
    
    files_err, files_ok = r.find_impact_images()
    
    for mdl in file_list:
        pos = pos + 1
        logger.info(f"processing: {pos}/{len(file_list)} {mdl}") 
        m1 = load_model(mdl)
        last_conv_layer_name = [layer.name for layer in m1.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
        model_filename = os.path.basename(mdl)
        parts = model_filename.split('_', 2)  
        part = parts[2]
        
        i = 0
        for fl in files_err:
            i = i + 1
            img_base = d.load_img_to_predict(INPUT_PATH + fl)
            curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"figures/{curr_date}_err_{part}_{i}.png"  
            img = np.expand_dims(img_base, axis=0)
            heatmap = make_gradcam_heatmap(img, m1, last_conv_layer_name)
            plt.imshow(img_base)
            # plt.imshow(heatmap, alpha=0.4)
            plt.imshow(tf.image.resize(heatmap[..., tf.newaxis], (140, 140)), cmap='jet', alpha=0.3)
            plt.axis('off') 
            # plt.show()
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()  
            logger.info(f"image {i+1}/10: {filename}") 

        i = 0
        for fl in files_ok:
            i = i + 1
            img_base = d.load_img_to_predict(INPUT_PATH + fl)
            curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"figures/{curr_date}_ok_{part}_{i}.png"  
            img = np.expand_dims(img_base, axis=0)
            heatmap = make_gradcam_heatmap(img, m1, last_conv_layer_name)
            plt.imshow(img_base)
            # plt.imshow(heatmap, alpha=0.4)
            plt.imshow(tf.image.resize(heatmap[..., tf.newaxis], (140, 140)), cmap='jet', alpha=0.3)
            plt.axis('off') 
            # plt.show()
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()  
            logger.info(f"image {i+1}/10: {filename}") 

# grad_model_iterator()



def grad_for_image(_source_filename, _dest_folder, _model, _prefix):
    INPUT_PATH = g.IMG_PATH_BASE
    file_list = glob.glob("models/*")
    
    cnn2_files = [f for f in file_list if _model in f]

    file_scores = []
    for f in cnn2_files:
        match = re.search(r'h5_(\d+\.\d+)', f)
        if match:
            score = float(match.group(1))
            file_scores.append((f, score))
    
    if file_scores:
        best_file, best_score = max(file_scores, key=lambda x: x[1])
        print(f"Best file: {best_file} with score {best_score}")
    else:
        print("No matching cnn2 files found.")
    
    m1 = load_model(best_file)
    last_conv_layer_name = [layer.name for layer in m1.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    model_filename = os.path.basename(best_file)
    parts = model_filename.split('_', 2)  
    part = parts[2]
    

    img_base = d.load_img_to_predict(INPUT_PATH + _source_filename)
    # img_base = im.remove_background(img_base)
    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # _source_filename = 'base_resized_out_shape_101a_1707_38.png'
    # sgh_id = re.search(r'(\d+_\d+)\.png$', _source_filename).group(1)
    
    dest_filename = f"figures/{_dest_folder}/{_prefix}_{os.path.splitext(_source_filename)[0]}_{part}.png"  
    img = np.expand_dims(img_base, axis=0)
    heatmap = make_gradcam_heatmap(img, m1, last_conv_layer_name)
    # heatmap = im.remove_background(heatmap)
    plt.imshow(img_base)
    # plt.imshow(heatmap, alpha=0.4)
    plt.imshow(tf.image.resize(heatmap[..., tf.newaxis], (140, 140)), cmap='jet', alpha=0.3)
    plt.axis('off') 
    # plt.show()
    plt.savefig(dest_filename, bbox_inches='tight', pad_inches=0)
    plt.close()  
    logger.info(f"image {_source_filename} saved to {dest_filename}") 
    
    return dest_filename
       