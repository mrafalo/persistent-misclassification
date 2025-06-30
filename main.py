import os

if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")

import pandas as pd
import numpy as np
import work
import work.models as m
import work.data as d
import work.globals as g
import work.grad as gr
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
import matplotlib.pyplot as plt

absl.logging.set_verbosity(absl.logging.ERROR)

logger = cl.get_logger()


def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))

def model_cv_iterator(_c, _epochs, _iters, _result_file, _result_details_file):
    
    logger.info(f"starting model training for {_c['model_name']}....")

    if _c['filter'] == "none": INPUT_PATH = g.IMG_PATH_BASE
    if _c['filter'] == "canny": INPUT_PATH = g.IMG_PATH_CANNY
    if _c['filter'] == "heat": INPUT_PATH = g.IMG_PATH_HEAT
    if _c['filter'] == "sobel": INPUT_PATH = g.IMG_PATH_SOBEL
    if _c['filter'] == "bw": INPUT_PATH = g.IMG_PATH_BW
    if _c['filter'] == "felzen": INPUT_PATH = g.IMG_PATH_FELZEN

    iter_num = 0
    
    for i in range(_iters):
        start = timeit.default_timer()
        
        iter_num = iter_num + 1
        logger.info(f"starting iteration ({_c['model_name']}) {iter_num}/{_iters}")
            
                     
        m1 = m.get_model_by_name(_c)
          
        keras.backend.clear_session()
        
        if "ext" in _c['model_name']:
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4cancer_ext(INPUT_PATH, _c, 0.15, 0.1)
            
        else:
            X_train, y_train, X_val, y_val, X_test, y_test, _, _, id_coi_list_test, images_list = d.split_data_4cancer(INPUT_PATH, _c, 0.15, 0.1)
            
        ev, res_details = m.model_fitter(m1, X_train, y_train, X_val, y_val, X_test, y_test, id_coi_list_test, images_list, _epochs, 
                            _c['learning_rate'], _c['batch_size'], _c['optimizer'], _c['loss_function'], _c['model_name']);
                          

        stop = timeit.default_timer()
        elapsed = timedelta(minutes=stop-start)
        curr_date = datetime.now().strftime("%Y%m%d_%H%M")
        
        res = pd.DataFrame([ev])
        res['model_name'] = _c['model_name']
        res['learning_rate'] = _c['learning_rate']
        res['batch_size'] = _c['batch_size']
        res['optimizer'] = _c['optimizer']
        res['loss_function'] = _c['loss_function']        
        res['features'] = _c['features']        
        res['img_size'] = _c['img_size']    
        res['augment'] = _c['augment']    
        res['filter'] = _c['filter']    
            
        res['train_dataset_size'] = len(y_train)
        res['train_dataset_size'] = len(y_train)
        
        res['elapsed_mins'] = elapsed.seconds//1800
        res['train_dataset_size'] = len(y_train)
        res['test_dataset_size'] = len(y_test)
        
        res['train_target_ratio'] = round(sum(y_train[:,1])/len(y_train),2)
        res['test_target_ratio'] = round(sum(y_test[:,1])/len(y_test),2),
        
        res['train_target_cases'] = sum(y_train[:,1])
        res['test_target_cases'] = sum(y_test[:,1])
        
        res['run_date'] = curr_date
        res['epochs'] = _epochs
        res['iteration'] = i+1
        
        existing_df = pd.read_csv(_result_file, sep=';')
        res = res[existing_df.columns]
        res.to_csv(_result_file, mode='a', header=False, index=False, sep=';')

        existing_df = pd.read_csv(_result_details_file, sep=';')
        res_details['iteration'] = [i+1 for x in range(len(y_test))]
        res_details['model_name'] = [_c['model_name'] for x in range(len(y_test))]
        res_details = res_details[existing_df.columns]
        res_details.to_csv(_result_details_file, mode='a', header=False, index=False, sep=';')
        
        
def train_cv(_config_file, _result_file, _result_details_file):
    
    # _config_file = "model_config.csv"
    cfg = pd.read_csv(_config_file, sep=";")
    
    if not os.path.exists(_result_file):
        df = pd.DataFrame(columns=g.RESULT_COLUMNS)
        df.to_csv(_result_file, index=False, sep=';')

    if not os.path.exists(_result_details_file):
        df = pd.DataFrame(columns=g.RESULT_DETAILS_COLUMNS)
        df.to_csv(_result_details_file, index=False, sep=';')
        
    df = pd.read_csv(_result_file, sep=";")
    
    for _, c in cfg.iterrows():
        if not(m.result_found(df, c)):
            logger.info(f"processing config ... model: {c['model_name']}")     
            model_cv_iterator(c, g.EPOCHS, g.CV_ITERATIONS, _result_file, _result_details_file)

        else: 
            logger.info("config found! skipping...")
            logger.info(c.to_dict())
            
def main_loop(_config_file):
    random.seed(g.SEED)
    np.random.seed(g.SEED)
    tf.keras.utils.set_random_seed(g.SEED)
    
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = f'results/{curr_date}_results.csv'
    result_details_file = f'results/{curr_date}_details.csv'
    
    logger.info(f"training loop starting... epochs: {str(g.EPOCHS)} cv iterations: {str(g.CV_ITERATIONS)}")
    train_cv(_config_file, result_file, result_details_file)
    logger.info("training finished!")



# importlib.reload(work.models)
# importlib.reload(work.reports)
# importlib.reload(work.data)
# importlib.reload(work.globals)
# importlib.reload(work.grad)
# importlib.reload(work.image_manipulator)

# featre importance stabilityi in cross validaton
# g.generte_model_config('model_config.csv')

main_loop("model_config.csv")

# res = m.log_reg_cv(10)

# r.raport_image_details()

