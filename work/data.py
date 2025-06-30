import pandas as pd
import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import random
import shutil
import work.image_manipulator as im
import work.globals as g
import fnmatch
from sklearn.model_selection import train_test_split
import work.custom_logger as cl
import ast

logger = cl.get_logger()


def label_cancer (row):
   if row['hp_ptc'] == 1 :
      return 'PTC'
   if row['hp_ftc'] == 1 :
      return 'FTC'
   if row['hp_hurthlea'] == 1:
      return 'HURTHLEA'
   if row['hp_mtc']  == 1:
      return 'MTC'
   if row['hp_dobrze_zroznicowane'] == 1:
      return 'DOBRZE_ZROZNICOWANY'
   if row['hp_ana'] == 1:
      return 'ANAPLASTYCZNY'
   if row['hp_plasko'] == 1:
      return 'PLASKONABLONKOWY'
   else:
    return 'BENIGN'

def load_base_data_file():
    df = pd.read_csv(g.BASE_FILE_PATH, sep=';')
    # df['label_cancer'] = df.apply (lambda row: label_cancer(row), axis=1)
    df.columns = df.columns.str.lower()
    df = df.fillna(-1)
    df[g.BASE_VARIABLES] = df[g.BASE_VARIABLES].astype(int)

    # df.loc[df.BACC_2==1, 'BACC_Bethesda']='kat2'
    # df.loc[df.BACC_3==1, 'BACC_Bethesda']='kat3'
    # df.loc[df.BACC_4==1, 'BACC_Bethesda']='kat4'
    # df.loc[df.BACC_5==1, 'BACC_Bethesda']='kat5'
    # df.loc[df.BACC_6==1, 'BACC_Bethesda']='kat6'

    # df.loc[df.tirads_2==1, 'tirads']='2'
    # df.loc[df.tirads_3==1, 'tirads']='3'
    # df.loc[df.tirads_4==1, 'tirads']='4'
    # df.loc[df.tirads_5==1, 'tirads']='5'
    
    df['size_max'] = df[['szerokosc', 'grubosc', 'dlugosc']].max(axis=1)

    
    
    return df

def split_files(_source_path, _train_path, _val_path, _test_path, _val_ratio, _test_ratio):

    
    paths = [_train_path, _test_path, _val_path]
    for p in paths:
        isExist = os.path.exists(p)
        if not isExist:
            os.makedirs(p)
    
    for f in os.listdir(_train_path):
       file_path = os.path.join(_train_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)

    for f in os.listdir(_test_path):
       file_path = os.path.join(_test_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)
         
    for f in os.listdir(_val_path):
       file_path = os.path.join(_val_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)
         
    number_of_test_files = int(np.round(_test_ratio * len(fnmatch.filter(os.listdir(_source_path), '*.png'))))
    number_of_val_files = int(np.round(_val_ratio * len(fnmatch.filter(os.listdir(_source_path), '*.png'))))
   
    test_list = []
    for i in range(number_of_test_files):
        random_file = random.choice(os.listdir(_source_path))
        filename = os.fsdecode(random_file)
        if filename.endswith(".png"): 
            shutil.copy(_source_path + random_file, _test_path + random_file)
            test_list.append(filename)
                   
    val_list = []
    for i in range(number_of_val_files):
        random_file = random.choice(os.listdir(_source_path))
        filename = os.fsdecode(random_file)
        if filename.endswith(".png"): 
            shutil.copy(_source_path + random_file, _val_path + random_file)
            val_list.append(filename)
       
    for f in os.listdir(_source_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            if (not filename in val_list) and (not filename in test_list):
                shutil.copy(_source_path + f, _train_path + f)
        

def split_data_4cancer_ext(_base_path, _cfg, _val_ratio, _test_ratio):
    
    df = load_base_data_file()
    # id_coi = '2'
    X = []
    X_ext = []
    y = []

    for f in os.listdir(_base_path):
        f_slit = f.split('_')
    
        id_coi = f_slit[4]
     
        if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
            rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
            feature_list = ast.literal_eval(_cfg['features'])
            ext_data =  df[df.id_coi==id_coi][feature_list].values[0]
            
            y.append(rak)
            
            if _cfg['channels'] == 1:
                img = cv2.imread(_base_path + f, cv2.IMREAD_GRAYSCALE)
            else:
               img = cv2.imread(_base_path + f) 
               img = im.remove_background(img)
               
            resized = im.resize_with_aspect_ratio(img, _cfg['img_size'], _cfg['img_size'])
            
            X.append(np.array(resized))
            X_ext.append(ext_data)
            if _cfg['augment'] > 0:
                for i in range(0, _cfg['augment']):
                    y.append(rak)
                    X.append(np.array(im.augment(resized)))
                    X_ext.append(ext_data)

        else:
            raise ValueError("Patient id_coi:", id_coi, 'not found!')

    X_train, X_test, X_ext_train, X_ext_test, y_train, y_test = train_test_split(X, X_ext, y, test_size=_test_ratio)
    X_train, X_val, X_ext_train, X_ext_val, y_train, y_val = train_test_split(X_train, X_ext_train, y_train, test_size=_val_ratio)
    

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_ratio)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=_val_ratio)
    
    
    train_size = len(X_train)
    test_size = len(X_test)
    val_size = len(X_val)
    
    X_train = np.array(X_train)
    X_ext_train = np.array(X_ext_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    X_ext_test = np.array(X_ext_test)
    y_test = np.array(y_test)
    
    X_val = np.array(X_val)
    X_ext_val = np.array(X_ext_val)
    y_val = np.array(y_val)
    
    im_width = X_train[0].shape[0]
    im_height = X_train[0].shape[1]
    
    X_train = X_train.reshape(train_size,im_width,im_height,_cfg['channels'])
    X_val = X_val.reshape(val_size,im_width,im_height,_cfg['channels'])
    X_test = X_test.reshape(test_size,im_width,im_height,_cfg['channels'])
    
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_val /= 255
    X_test /= 255

    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)


    return [X_train, X_ext_train], y_train, [X_val, X_ext_val], y_val, [X_test, X_ext_test], y_test  

def split_data_4cancer(_base_path, _cfg, _val_ratio, _test_ratio):
    
    # _base_path = g.IMG_PATH_BASE        
    # _config_file = "model_config.csv"
    # cfg = pd.read_csv(_config_file, sep=";")  
    # for _, c in cfg.iterrows():
    #     _cfg = c


    df = load_base_data_file()
    
    X = []
    y = []
    id_coi_list = []    
    images_list = []
   
    for f in os.listdir(_base_path):
        f_slit = f.split('_')
    
        id_coi = f_slit[4]
     
        if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
           
            rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
            id_coi_list.append(id_coi)
            images_list.append(f)
            y.append(rak)
            if _cfg['channels'] == 1:
                img = cv2.imread(_base_path + f, cv2.IMREAD_GRAYSCALE)
            else:
               img = cv2.imread(_base_path + f) 
               img = im.remove_background(img)
               
            resized = im.resize_with_aspect_ratio(img, _cfg['img_size'], _cfg['img_size'])
            
            X.append(np.array(resized))
            if _cfg['augment'] > 0:
                for i in range(0, _cfg['augment']):
                    y.append(rak)
                    X.append(np.array(im.augment(resized)))

        else:
            raise ValueError("Patient id_coi:", id_coi, 'not found!')

    
    # id_coi_list = list(set(id_coi_list))
    # len(id_coi_list)    
    # tmp = df[df['id_coi'].isin(id_coi_list)]
        

    # tmp = tmp[['id_coi', 'plec', 'rak', 'wiek', 'tirads_3', 'tirads_4', 'tirads_5']].drop_duplicates()
    # tmp.groupby(['rak']).agg(
    #     count=('wiek', 'count'),
    #     mean_wiek=('wiek', 'mean'),
    #     sd_wiek=('wiek', 'std')
    #     ).reset_index()    
    
    
    # tmp = tmp[['id_coi', 'rak','brzegi_mikrolobularne',  'echo_gleboko_hipo', 'echo_izoechogeniczna', 'granice_zatarte', 'zwapnienia_mikrozwapnienia']].drop_duplicates()
    # columns_to_sum = ['brzegi_mikrolobularne',  'echo_gleboko_hipo', 'echo_izoechogeniczna', 'granice_zatarte', 'zwapnienia_mikrozwapnienia']
    # all_sum = tmp[columns_to_sum].sum()
    
    # # Sum for rows where rak == 1
    # rak_1_sum = tmp[tmp['rak'] == 1][columns_to_sum].sum()
    
    # # Sum for rows where rak == 0
    # rak_0_sum = tmp[tmp['rak'] == 0][columns_to_sum].sum()
    
    # # Combine into a single DataFrame for clarity
    # result = pd.DataFrame({
    #     'all': all_sum,
    #     'rak=1': rak_1_sum,
    #     'rak=0': rak_0_sum
    # })
   
    # result_translated = result.rename(index=g.VARIABLES_TRANSLATE_PAPER)
    # result_translated.to_csv('res.csv', sep=';')
    
    X_train, X_test, y_train, y_test, id_coi_list_train, id_coi_list_test, _, images_list_test = train_test_split(X, y, id_coi_list, images_list, test_size=_test_ratio)
    X_train, X_val, y_train, y_val, id_coi_list_train, id_coi_list_val = train_test_split(X_train, y_train, id_coi_list_train, test_size=_val_ratio)
    
    
    train_size = len(X_train)
    test_size = len(X_test)
    val_size = len(X_val)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    im_width = X_train[0].shape[0]
    im_height = X_train[0].shape[1]
    X_train = X_train.reshape(train_size,im_width,im_height,_cfg['channels'])
    X_val = X_val.reshape(val_size,im_width,im_height,_cfg['channels'])
    X_test = X_test.reshape(test_size,im_width,im_height,_cfg['channels'])
    
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_val /= 255
    X_test /= 255

    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)


    return X_train, y_train, X_val, y_val, X_test, y_test, id_coi_list_train, id_coi_list_val, id_coi_list_test, images_list_test   
                    
def split_data_4feature(_base_path, _augument, _val_ratio, _test_ratio, _feature, _seed=123):
    
    df = load_base_data_file()

    X = []
    y = []
    
    for f in os.listdir(_base_path):
        f_slit = f.split('_')
    
        id_coi = f_slit[4]
     
        if len(df.loc[(df.id_coi==id_coi) ,_feature]) >0:
            feature_val = df.loc[(df.id_coi==id_coi) ,_feature].iloc[0]
            y.append(feature_val)
            X.append(np.array(cv2.imread(_base_path + f, cv2.IMREAD_GRAYSCALE)))
        else:
            raise ValueError("Patient id_coi:", id_coi, 'not found!')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_ratio, random_state=_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=_val_ratio, random_state=_seed)
    
    if _augument > 0:
        X_train_tmp = X_train
        y_train_tmp = y_train
    
        for i in range(0, _augument):
            X_train_augumented = im.augment(X_train)
            X_train_tmp = X_train_tmp + X_train_augumented
            y_train_tmp = y_train_tmp + y_train
            
        X_train = X_train_tmp
        y_train = y_train_tmp
    
    train_size = len(X_train)
    test_size = len(X_test)
    val_size = len(X_val)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    im_width = X_train[0].shape[0]
    im_height = X_train[0].shape[1]
    X_train = X_train.reshape(train_size,im_width,im_height,1)
    X_val = X_val.reshape(val_size,im_width,im_height,1)
    X_test = X_test.reshape(test_size,im_width,im_height,1)
    
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_val /= 255
    X_test /= 255
    
    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test        
                    

def combine_results(_in, _out):
    # _in = ['20250225_1417_results.csv', '20250227_2039_results.csv', '20250228_0849_results.csv']
    combined_df = pd.DataFrame();
    
    for file in _in:
        # print(file)
        combined_df = pd.concat([combined_df, pd.read_csv(f"results/{file}", sep=';')], ignore_index=True)
    
    combined_df.to_csv(f"results/{_out}", sep=';')


def load_img_to_predict(_file_path):

    X_val = np.array(cv2.imread(_file_path))
    
    X_val = im.resize_with_aspect_ratio(X_val, 140, 140)
    
    # im_width = X_val.shape[0]
    # im_height = X_val.shape[1]
    
    # X_val = X_val.reshape(1,im_width,im_height,3)
    X_val = X_val.astype('float32')
    X_val /= 255

    return X_val        
                    

def prepare_data_filters():
    
    res = im.extract_and_resize_images(g.ANNOTATION_INPUT_PATH, g.MODELING_INPUT_PATH, g.RAW_INPUT_PATH, g.IMG_WIDTH, g.IMG_HEIGHT)
    logger.info('good! ' + str(res) + ' images resized to: (' + str(g.IMG_WIDTH) + ', ' + str(g.IMG_HEIGHT) + ') saved to: ' + g.MODELING_INPUT_PATH)


    res = 0

    paths = [g.IMG_PATH_BASE, g.IMG_PATH_CANNY, g.IMG_PATH_HEAT,  g.IMG_PATH_SOBEL, g.IMG_PATH_BW, g.IMG_PATH_FELZEN]
    for p in paths:
        isExist = os.path.exists(p)
        if not isExist:
            os.makedirs(p)
   
    for f in os.listdir(g.MODELING_INPUT_PATH):
        filename = os.fsdecode(f)
        if filename.endswith(".png"):
            img = cv2.imread(g.MODELING_INPUT_PATH + f)
            canny = im.edges(img, 30, 105)
            heat = im.heatmap(img)
            sobel = im.sobel(img)
            bw = im.bw_mask(img)
            felzen = im.felzenszwalb(img, 110)
            
            cv2.imwrite(g.IMG_PATH_BASE +  f, img)
            cv2.imwrite(g.IMG_PATH_CANNY +  f, canny)
            cv2.imwrite(g.IMG_PATH_HEAT +  f, heat)
            cv2.imwrite(g.IMG_PATH_SOBEL +  f, sobel)
            cv2.imwrite(g.IMG_PATH_BW + f, bw)
            cv2.imwrite(g.IMG_PATH_FELZEN +  f, felzen)
            
            res = res + 1
    logger.info(f"{res} images transformed...")        

    return res

