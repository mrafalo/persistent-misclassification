import pandas as pd
from scipy.stats import norm

import numpy as np
import work.models as m
import work.data as d
import work.globals as g
import work.grad as gr
import work.image_manipulator as im
import work.custom_logger as cl
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
from collections import Counter
import glob
import random
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from tabulate import tabulate
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import kaleido
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_auc_score
import seaborn as sns

pio.renderers.default='svg'

logger = cl.get_logger()
              
def get_layers_info(_model):
    print("input layer:", _model.input_names)
    
    for layer in _model.layers:
        if 'conv' not in layer.name:
            print(layer.name)
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)

def plot_kernels(_model):
    
    filters, biases = _model.layers[2].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
    	# get the filter
    	f = filters[:, :, :, i]
    	# plot each channel separately
    	for j in range(3):
    		# specify subplot and turn of axis
    		ax = pyplot.subplot(n_filters, 3, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		# plot filter channel in grayscale
    		pyplot.imshow(f[:, :, j], cmap='gray')
    		ix += 1
    # show the figure
    pyplot.show()
    

def roc_plot(y_test, y_pred):
  
  m1_fpr, m1_tpr, _ = metrics.roc_curve(y_test,  y_pred)
  m1_auc = metrics.roc_auc_score(y_test, y_pred)

  plt.plot(m1_fpr,m1_tpr,label="model 1, auc="+str(round(m1_auc,2)))
  plt.legend(loc=4)
  plt.show()
  

def report_variables_vs_cancer():
    df = d.load_base_data_file()
    
    for z in g.BASE_VARIABLES:
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['label_cancer'], tmp[z])
        p_val = m.chi2(contigency)

def report_variables_vs_PTC():
    df = d.load_base_data_file()
    df1 = df[df.label_cancer.isin(['PTC'])]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])
    
    print("Liczba pacjentów: ", len(df), "liczba PTC:", len(df[df.rak == 1]))
        
    for z in g.BASE_VARIABLES:
        tmp = df.loc[df[z]>=0,]
        contigency = pd.crosstab(tmp['HP_PTC'], tmp[z])
        p_val = m.chi2(z, contigency)

        if p_val < 0.05:
            #print(z, round(p_val,3))
            x11=contigency[0][0]
            x12=contigency[1][0]
            x21=contigency[0][1]
            x22=contigency[1][1]
        
            #print(contigency)
            print("Wsród pacjentów z ",z, " ", round(x22/(x12+x22)*100), "% ma raka PTC", sep="")
            print("Wsród pacjentów z ",z, " ", round(x12/(x12+x22)*100), "% nie ma raka PTC", sep="")
            print("Wsród pacjentów z rakiem PTC ",round(x22/(x21+x22)*100), "% ma ", z, sep="")
            print("Wsród pacjentów bez raka ",round(x12/(x12+x11)*100), "% ma ", z, sep="")
            print("--------------------")


def report_overview(_latex = False):
    df = d.load_base_data_file() 
    print("Liczba pacjentów: ", len(df), "liczba nowotworow złoliwych:", len(df[df.rak == 1]), "liczba łagodnych:", len(df[df.label_cancer == "BENIGN"]))    
    
    cancer_list = []
    
    for img_file in glob.glob(os.path.join(g.IMG_PATH_HEAT, "*.png")):
        img_filename = os.path.basename(img_file)
        f_slit = img_filename.split('_')
        id_coi = f_slit[4]
        if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
            rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
            cancer_list.append(rak)     
            
            
    print(f"liczba obrazow: {len(cancer_list)}, benign: {len(cancer_list)- sum(cancer_list)}, malignant: {sum(cancer_list)} ")

    print("Pacjenci wg rodzaju nowotworu i Bethesda:")
    tmp = pd.crosstab(df.rak, df.BACC_Bethesda, margins = False) 
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='html'))
        html = tabulate(tmp,headers='firstrow',tablefmt='html')
        text_file = open("out.html", "w") 
        text_file.write(html) 
        text_file.close() 


    print("Pacjenci wg rodzaju nowotworu i tirads:")
    tmp = pd.crosstab(df.rak, df.tirads, margins = False) 
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='html'))
        html = tabulate(tmp,headers='firstrow',tablefmt='html')
        text_file = open("out.html", "w") 
        text_file.write(html) 
        text_file.close() 
        
        

    print("Pacjenci wg płci:")
    tmp = pd.crosstab(df.rak, df.plec, margins = False) 
    tmp
    print(tmp.to_string(index=True),'\n')
    
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='grid'))
    
    print("Pacjenci wg cech:")
    
  
    variables = []
    cancer1 = []
    cancer0 = []
    p_values = []
    i = 0

      
    for z in g.VARIABLES_DICT:
        
        tmp = df.loc[df[z]==1,].groupby('rak').size().reset_index(name='cnt')
        ct = pd.crosstab(df['rak'], df[z])
         
        if len(tmp) > 1:
            chi2_p_value = round(m.chi2(ct),3)
            #print(z, chi2_p_value)
            variables.append(g.VARIABLES_DICT[z])
            p_values.append(chi2_p_value)
            cancer1.append(tmp.at[1,"cnt"])
            cancer0.append(tmp.at[0,"cnt"])
        i = i + 1

    tmp = pd.DataFrame({
        'Feature': variables,
        'Benign': cancer0,
        'Malignant': cancer1,
        'chi2 p-value': p_values
        })
    
    if _latex:
        print(tmp.to_latex(),'\n')
    else:
        print(z,'\n')
        print(tmp.to_string(index=False),'\n')
        

        
def raport_images():

    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "reports/report_images_" + curr_date + ".pdf"       
    df = d.load_base_data_file()
    png_count = len(glob.glob(os.path.join(g.IMG_PATH_HEAT, "*.png")))
    
    pos = 0
    with PdfPages(filename) as pdf:   
        for img_file in glob.glob(os.path.join(g.IMG_PATH_HEAT, "*.png")):
            pos = pos + 1
            img_filename = os.path.basename(img_file)
            f_slit = img_filename.split('_')
            id_coi = f_slit[4]
            rak = -1
            df2 = df.loc[(df.id_coi==id_coi)]
            if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
                rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
                
                
            logger.info(f"processing: {pos}/{png_count} {img_filename}") 

            fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69))  
            
            fig.text(0.05, 0.95, f"id_coi: {id_coi} echo_izoechogeniczna: {df2['echo_izoechogeniczna'].values[0]} rak: {rak} plik {img_filename}", ha='left', fontsize=12, color='black')
               
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axs[0].imshow(img, cmap = "Reds")
            axs[0].set_axis_off()
            
            img = im.remove_background(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            channel_b, channel_g, channel_r = cv2.split(img)
            
            # Convert single-channel to 3-channel images for display
            channel_b = cv2.merge([channel_b, np.zeros_like(channel_b), np.zeros_like(channel_b)])  # Blue only
            channel_g = cv2.merge([np.zeros_like(channel_g), channel_g, np.zeros_like(channel_g)])  # Green only
            channel_r = cv2.merge([np.zeros_like(channel_r), np.zeros_like(channel_r), channel_r])  # Red only

            separator = np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 255  # White strip
            
            image_combined = np.hstack([channel_r, separator, channel_g, separator, channel_b])
            
            img = cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB)

            axs[1].imshow(img)
            axs[1].set_axis_off()
                
            plt.tight_layout(h_pad=2.0)
            plt.subplots_adjust(top=0.90) 
            pdf.savefig()
            plt.close()


def raport_image_details():

    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "reports/report_image_details_" + curr_date + ".pdf"       
    df = d.load_base_data_file()
    png_count = len(glob.glob(os.path.join(g.IMG_PATH_HEAT, "*.png")))
    
    pos = 0
    with PdfPages(filename) as pdf:   
        for img_file in glob.glob(os.path.join(g.IMG_PATH_HEAT, "*.png")):
            pos = pos + 1
            img_filename = os.path.basename(img_file)
            f_slit = img_filename.split('_')
            id_coi = f_slit[4]
            rak = -1
            # id_coi='2'
            df2 = df.loc[(df.id_coi==id_coi)]
            if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
                rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
                
                
            logger.info(f"processing: {pos}/{png_count} {img_filename}") 

            fig, axs = plt.subplots(1, 2, figsize=(8.27, 11.69), gridspec_kw={'width_ratios': [4, 1]})  
            
            fig.text(0.05, 0.95, f"numer pacjenta: {id_coi}", ha='left', fontsize=12, color='black')
            fig.text(0.05, 0.93, f"plik: {img_filename}", ha='left', fontsize=12, color='black')
            fig.text(0.05, 0.91, f"rak: {rak}", ha='left', fontsize=12, color='black')

            # img_file = g.IMG_PATH_HEAT + "heat_resized_out_shape_2_1580_20.png"
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img[:,:,0] = 0
            # axs[0].imshow(img, cmap = "Reds")
            # axs[0].set_axis_off()
            
            img = im.remove_background(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            channel_b, channel_g, channel_r = cv2.split(img)
            
            # Convert single-channel to 3-channel images for display
            channel_b = cv2.merge([channel_b, np.zeros_like(channel_b), np.zeros_like(channel_b)])  # Blue only
            channel_g = cv2.merge([np.zeros_like(channel_g), channel_g, np.zeros_like(channel_g)])  # Green only
            channel_r = cv2.merge([np.zeros_like(channel_r), np.zeros_like(channel_r), channel_r])  # Red only

            separator = np.ones((10,img.shape[0], 3), dtype=np.uint8) * 255  # White strip
            
            image_combined = np.vstack([channel_r, separator, channel_g, separator, channel_b])
            
            img = cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB)

            axs[0].imshow(img)
            axs[0].set_axis_off()
            
            
            df_pivot = df2[g.FEATURES_PL].iloc[0,:].T.reset_index()  # Transpose and reset index
            df_pivot.columns = ['Feature', 'Value']  # Rename columns

            axs[1].axis('tight')  # Remove white space
            axs[1].axis('off')  # Hide axes
            
            # Add table to plot
            table = axs[1].table(cellText=df_pivot.values, 
                             colLabels=df_pivot.columns, 
                             cellLoc='left', 
                             loc='right')
        
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width([0, 1, 2])
    

            plt.tight_layout(h_pad=2.0)
            plt.subplots_adjust(top=0.90) 
            pdf.savefig()
            plt.close()    
            
def raport_feature_importance():

    res_rf, feature_importance_rf = m.random_forest_cv(50)
    res_lr, feature_importance_lr = m.log_reg_cv(50)
    res_ci = m.compute_cm_metrics()
    res_ci.columns
    
    features_list = res_rf['feature'].unique().tolist()
    
    plt.figure(figsize=(10, 6))
    
    feature_importance_rf['feature2'] = feature_importance_rf['feature'].map(g.VARIABLES_TRANSLATE_PAPER)


    sns.barplot(data=feature_importance_rf, x="importance", y="feature2")
   
    plt.title("Feature Importance")
    plt.ylabel('US feature')
    plt.xlabel('importance')
    
    plt.savefig("fig4.png", dpi=300, bbox_inches='tight')
        
    plt.show()


def raport_results_summary():
    
    def confidence_interval_ppv(row):
        confidence = 0.95
        if row['TP'] + row['FP'] == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt((row['PPV'] * (1 - row['PPV'])) / (row['TP'] + row['FP']))
        return (round(max(0, row['PPV'] - z * se),3), round(min(1, row['PPV'] + z * se),3))
    
    def confidence_interval_npv(row):
        confidence = 0.95
        if row['TN'] + row['FN'] == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt((row['NPV'] * (1 - row['NPV'])) / (row['TN'] + row['FN']))
        return (round(max(0, row['NPV'] - z * se),3), round(min(1, row['NPV'] + z * se),3))



    def sensitivity_confidence_interval(row):
        confidence = 0.95
        if row['TP'] + row['FN'] == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt((row['sensitivity'] * (1 - row['sensitivity'])) / (row['TP'] + row['FN']))
        return (round(max(0, row['sensitivity'] - z * se),3), round(min(1, row['sensitivity'] + z * se),3))



    df = pd.read_csv('results/results.csv', sep=';')
    
    
    df['P'] = df['test_target_cases']
    df['N'] = df['test_dataset_size'] - df['test_target_cases']
    df['TP'] = round(df['sensitivity'] * df['P'])
    df['FN'] = df['P'] - df['TP']
    df['TN'] = round(df['specificity'] * df['N'])
    df['FP'] = df['test_dataset_size'] - df['TP'] - df['FN'] - df['TN']
    df['PPV'] = df['precision']
    df['NPV'] = round(df['TN']/(df['TN'] + df['FN']),2)

    df['PPV_CI'] = df.apply(confidence_interval_ppv, axis=1)
    df['NPV_CI'] = df.apply(confidence_interval_npv, axis=1)
    df['sensitivity_CI'] = df.apply(sensitivity_confidence_interval, axis=1)
    
    df[['PPV_CI_lower', 'PPV_CI_upper']] = pd.DataFrame(df['PPV_CI'].tolist(), index=df.index)
    df[['NPV_CI_lower', 'NPV_CI_upper']] = pd.DataFrame(df['NPV_CI'].tolist(), index=df.index)
    df[['sensitivity_CI_lower', 'sensitivity_CI_upper']] = pd.DataFrame(df['sensitivity_CI'].tolist(), index=df.index)

    df_top = df.sort_values(by=['model_name', 'auc'], ascending=[True, False])\
           .groupby('model_name').head(5)
           
    df_summary = df_top.groupby('model_name').agg(
        cnt = ('iteration','count'),
        
        sensitivity_mean=('sensitivity', 'mean'),
        sensitivity_max=('sensitivity', 'max'),
        sensitivity_median=('sensitivity', 'median'),
        sensitivity_std=('sensitivity', 'std'),
        
        specificity_mean=('specificity', 'mean'),
        specificity_max=('specificity', 'max'),
        specificity_median=('specificity', 'median'),
        
        PPV_mean=('PPV', 'mean'),
        PPV_max=('PPV', 'max'),
        PPV_median=('PPV', 'median'),
        PPV=('PPV', 'mean'),
        PPV_std=('PPV', 'std'),
        
        NPV_mean=('NPV', 'mean'),
        NPV_max=('NPV', 'max'),
        NPV_median=('NPV', 'median'),
        NPV=('NPV', 'mean'),
        NPV_std=('NPV', 'std'),
        
        AUC_mean=('auc', 'mean'),
        AUC_max=('auc', 'max'),
        AUC_median=('auc', 'median'),
        AUC_std=('auc', 'std'),
        
        PPV_CI_lower=('PPV_CI_lower', 'mean'),
        PPV_CI_upper=('PPV_CI_upper', 'mean'),
        NPV_CI_lower=('NPV_CI_lower', 'mean'),
        NPV_CI_upper=('NPV_CI_upper', 'mean'),
        sensitivity_CI_lower=('sensitivity_CI_lower', 'mean'),
        sensitivity_CI_upper=('sensitivity_CI_upper', 'mean')
    
    ).round(2) 
    

    df_summary['PPV_CI'] = list(zip(df_summary['PPV_CI_lower'], df_summary['PPV_CI_upper']))
    df_summary['NPV_CI'] = list(zip(df_summary['NPV_CI_lower'], df_summary['NPV_CI_upper']))
    df_summary['sensitivity_CI'] = list(zip(df_summary['sensitivity_CI_lower'], df_summary['sensitivity_CI_upper']))
    
    df_summary.to_csv('results/results_summary2.csv', sep=';')
    

def calc_U(y_true, y_score):
    n1 = np.sum(y_true==1)
    n0 = len(y_score)-n1
    
    ## Calculate the rank for each observation
    # Get the order: The index of the score at each rank from 0 to n
    order = np.argsort(y_score)
    # Get the rank: The rank of each score at the indices from 0 to n
    rank = np.argsort(order)
    # Python starts at 0, but statistical ranks at 1, so add 1 to every rank
    rank += 1
    
    # If the rank for target observations is higher than expected for a random model,
    # then a possible reason could be that our model ranks target observations higher
    U1 = np.sum(rank[y_true == 1]) - n1*(n1+1)/2
    U0 = np.sum(rank[y_true == 0]) - n0*(n0+1)/2
    
    # Formula for the relation between AUC and the U statistic
    AUC1 = U1/ (n1*n0)
    AUC0 = U0/ (n1*n0)
    
    return AUC1


def find_impact_images():
    
    file_list = glob.glob("results/20250425*_details.csv")
    df_details = pd.concat((pd.read_csv(file, sep=';') for file in file_list), ignore_index=True)

    df_details['is_correct'] = df_details['actual'] == df_details['predict_binary']
    df_details['FP'] = (df_details['actual'] == 0) & (df_details['predict_binary'] == 1)
    df_details['FN'] = (df_details['actual'] == 1) & (df_details['predict_binary'] == 0)
    

    # Group and aggregate
    res = df_details.groupby(['model_name','image']).agg(
        number_of_occurrences=('image', 'count'),
        good_predictions=('is_correct', 'sum'),
        ERRORS=('is_correct', lambda x: (~x).sum()),
        FP=('FP', 'sum'),
        FN=('FN', 'sum')
    ).reset_index()

    res['FP_RATIO'] = res['FP']/res['number_of_occurrences']
    res['FN_RATIO'] = res['FN']/res['number_of_occurrences']
    res['ERR_RATIO'] = res['ERRORS']/res['number_of_occurrences']
    
    
    res2 =  res[res['ERR_RATIO'] >= 0.85]
    res3 = res2.groupby(['image']).agg(number_of_occurrences=('image', 'count')).reset_index()
    res4 = res3[res3['number_of_occurrences'] > 2]
    
    res_ok_agg = df_details.groupby(['image']).agg(
        number_of_occurrences=('image', 'count'),
        good_predictions=('is_correct', 'sum'),
        ERRORS=('is_correct', lambda x: (~x).sum()),
        FP=('FP', 'sum'),
        FN=('FN', 'sum')
    ).reset_index()
    res_ok_agg['FP_RATIO'] = res_ok_agg['FP']/res_ok_agg['number_of_occurrences']
    res_ok_agg['FN_RATIO'] = res_ok_agg['FN']/res_ok_agg['number_of_occurrences']
    res_ok_agg['ERR_RATIO'] = res_ok_agg['ERRORS']/res_ok_agg['number_of_occurrences']
    res_ok =  res_ok_agg[res_ok_agg['ERR_RATIO'] <0.05]
    
    filtered_files_ok = res_ok['image'].unique().tolist()
    filtered_files_err = res4['image'].unique().tolist()
    
    return filtered_files_err, filtered_files_ok

def auc_impact():
    
    def compute_auc(group):
        try:
            return roc_auc_score(group['actual'], group['predict'])
        except ValueError:
            return None  # Handle cases with only one class present

    file_list = glob.glob("results/20250425*_details.csv")
    df_details = pd.concat((pd.read_csv(file, sep=';') for file in file_list), ignore_index=True)
        
    df_auc = (
        df_details
        .groupby(['model_name', 'iteration'])
        .apply(compute_auc)
        .reset_index()
        .rename(columns={0: 'AUC'})
    )
    

    top10_per_model = (
        df_auc
        .sort_values(['model_name', 'AUC'], ascending=[True, False])
        .groupby('model_name')
        .head(20)
        )

    top10_keys = top10_per_model[['model_name', 'iteration']]
    df_top10_detail = df_details.merge(top10_keys, on=['model_name', 'iteration'])

    results = []
    auc_values = df_top10_detail.groupby(['model_name', 'iteration']).apply(compute_auc).reset_index()
    auc_values = auc_values.rename(columns={0: 'AUC'})    
    auc_values['n_excluded_images'] = 0
    results.append(auc_values)


    # Start cumulative exclusion
    excluded_images = []
    filtered_files_err, _ = find_impact_images()

    for idx, image_to_exclude in enumerate(filtered_files_err, start=1):
        excluded_images.append(image_to_exclude)
        df_temp = df_top10_detail[~df_top10_detail['image'].isin(excluded_images)]
        auc_values = df_temp.groupby(['model_name', 'iteration']).apply(compute_auc).reset_index()
        auc_values = auc_values.rename(columns={0: 'AUC'})
        auc_values['n_excluded_images'] = idx
        results.append(auc_values)
    
    final_auc_df = pd.concat(results, ignore_index=True)
    
    # print(final_auc_df)
    
        
    df_max_auc = (
        final_auc_df[final_auc_df['model_name']!='VGG16']
        .groupby(['model_name', 'n_excluded_images'])['AUC']
        .mean()
        .reset_index()
    )
    
    # final_auc_df[final_auc_df['model_name']!='VGG16'].groupby(['model_name']).agg(
    #     mean_AUC=('AUC', 'mean'),
    #     std_AUC=('AUC', 'std'),
    #     min_AUC=('AUC', 'min'),
    #     max_AUC=('AUC', 'max')
    #     ).reset_index().round(2)
        
    # 2. Plot
    plt.figure(figsize=(12, 7))
    
    sns.lineplot(
        data=df_max_auc,
        x='n_excluded_images',
        y='AUC',
        hue='model_name',
        marker='o'
    )
    
    plt.xlabel('Number of excluded images')
    plt.ylabel('Mean AUC (100 iterations)')
    # plt.title('Max AUC vs Number of Excluded Images by Model')
    plt.grid(True)
    plt.legend(title='Model name')
    plt.tight_layout()
    plt.show()


def grad_for_impact_pdf():
    filtered_files_err, _ = find_impact_images()
    df = d.load_base_data_file() 

    cancer_features = ['hp_ptc', 'hp_ftc',
       'hp_hurthlea', 'hp_mtc', 'hp_dobrze_zroznicowane', 'hp_ana',
       'hp_plasko', 'hp_ruczolak', 'hp_guzek_rozrostowy', 'hp_zapalenie',
       'hp_nieokreslone', 'hp_niftp', 'hp_wdump', 'hp_ftump']
    
    total_features = g.BASE_VARIABLES #cancer_features + g.FEATURES_PL
    id_coi_list = []
    for fname in filtered_files_err:
        id_coi_list.append(re.search(r'out_shape_([^_]+)_\d+_\d+\.png$', fname).group(1))
    
  
    df_selected = df[df['id_coi'].isin(id_coi_list)][['id_coi'] + total_features]

    percentages = df[total_features].mean().round(2)
    for feature in total_features:
        df_selected[feature] = df_selected[feature].apply(lambda x: percentages[feature] if x == 1 else 0)

    df_selected['min_nonzero_percentage'] = df_selected[total_features].replace(0, np.nan).min(axis=1)
    df_selected['min_nonzero_percentage'] = df_selected['min_nonzero_percentage'].round(2)
    df_temp = df_selected[total_features].replace(0, np.nan)

    df_selected['min_feature_name'] = df_temp.idxmin(axis=1)
    df_selected = df_selected.groupby(['min_feature_name', 'min_nonzero_percentage']).count().reset_index()

    df_selected.to_csv('tmp.csv', sep=';')
    
    ### pdf part
    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "reports/report_grad_" + curr_date + ".pdf"       
    
    pos = 0
    
    with PdfPages(filename) as pdf:   
        for file in filtered_files_err:
            pos = pos + 1
            logger.info(f"processing: {pos}/{len(filtered_files_err)} {file}") 

            sgh_id = re.search(r'(\d+_\d+)\.png$', file).group(1)
            img_grad_file = gr.grad_for_image(file, 'cnn2')
            img_grad = cv2.imread(img_grad_file)
            
            fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69))  
            #fig.text(0.05, 0.95, f"id_coi: {id_coi} echo_izoechogeniczna: {df2['echo_izoechogeniczna'].values[0]} rak: {rak} plik {img_filename}", ha='left', fontsize=12, color='black')
            img = cv2.imread(g.IMG_PATH_BASE + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axs[0].imshow(img)
            axs[0].set_axis_off()
            

            axs[1].imshow(img_grad)
            axs[1].set_axis_off()
            
            
            plt.tight_layout(h_pad=2.0)
            plt.subplots_adjust(top=0.90) 
            pdf.savefig()
            plt.close()

        


def grad_for_impact_files(_dest_folder):
    filtered_files_err, filtered_files_ok = find_impact_images()
    
    df = d.load_base_data_file() 


    INPUT_PATH = g.IMG_PATH_BASE
    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    _dest_folder = 'ResNet/'   
    pos = 0
    
    for file in filtered_files_err:
        pos = pos + 1
        logger.info(f"processing: {pos}/{len(filtered_files_err)} {file}") 

        img_grad_file = gr.grad_for_image(file, _dest_folder, 'ResNet', 'err')
        
        img_base = d.load_img_to_predict(INPUT_PATH + file)
        # img_base = im.remove_background(img_base)
        
        dest_filename = f"figures/{_dest_folder}/err_{file}"  
        plt.imshow(img_base)
        plt.axis('off') 
        plt.savefig(dest_filename, bbox_inches='tight', pad_inches=0)
        plt.close()  
    
    pos = 0
    for file in filtered_files_ok:
        pos = pos + 1
        logger.info(f"processing: {pos}/{len(filtered_files_ok)} {file}") 
        img_grad_file = gr.grad_for_image(file, _dest_folder, 'ResNet', 'ok')
        
        img_base = d.load_img_to_predict(INPUT_PATH + file)
        # img_base = im.remove_background(img_base)
        dest_filename = f"figures/{_dest_folder}/ok_{file}"  
        plt.imshow(img_base)
        plt.axis('off') 
        plt.savefig(dest_filename, bbox_inches='tight', pad_inches=0)
        plt.close()  

def corr_features():
    df = d.load_base_data_file() 
    
    binary_df = df[g.VARIABLES_TRANSLATE.keys()]
    
    corr_matrix = binary_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Rename columns for visualization (Polish → English)
    corr_matrix_renamed = corr_matrix.rename(columns=g.VARIABLES_TRANSLATE_PAPER, index=g.VARIABLES_TRANSLATE_PAPER)
    
    # Plot heatmap with English labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_renamed, mask=mask, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.savefig("fig3.png", dpi=300, bbox_inches='tight')
    
    plt.show()
