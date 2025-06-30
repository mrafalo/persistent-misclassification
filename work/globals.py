import yaml
import pandas as pd

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    BASE_FILE_PATH = cfg['BASE_FILE_PATH']
    MODELING_PATH = cfg['MODELING_PATH']
    RAW_INPUT_PATH = cfg['RAW_INPUT_PATH']
    ANNOTATION_INPUT_PATH = cfg['ANNOTATION_INPUT_PATH']
    MODELING_INPUT_PATH = cfg['MODELING_INPUT_PATH']
    
    IMG_PATH_BASE = cfg['IMG_PATH_BASE']
    IMG_PATH_BW = cfg['IMG_PATH_BW']
    IMG_PATH_SOBEL = cfg['IMG_PATH_SOBEL']
    IMG_PATH_HEAT = cfg['IMG_PATH_HEAT']
    IMG_PATH_CANNY = cfg['IMG_PATH_CANNY']
    IMG_PATH_FELZEN = cfg['IMG_PATH_FELZEN']
    
    CV_ITERATIONS = cfg['CV_ITERATIONS']
    EPOCHS = cfg['EPOCHS']
    SEED = cfg['SEED']
    
    
    
    
BASE_VARIABLES = [
    'echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
    'echo_izoechogeniczna', 'echo_mieszana', 
    'budowa_lita', 'budowa_lito_plynowa', 'budowa_plynowo_lita', 
    'ksztalt_owalny', 'ksztalt_okragly', 'ksztalt_nieregularny', 
    'orientacja_rownolegla',
    'granice_rowne', 'granice_zatarte', 'granice_nierowne', 
    'brzegi_katowe', 'brzegi_mikrolobularne', 'brzegi_spikularne', 
    'halo', 'halo_cienka', 'halo_gruba', 
    'zwapnienia_mikrozwapnienia', 'zwapnienia_makrozwapnienia', 'zwapnienia_makro_obraczkowate', 'zwapnienia_artefakty_typu_ogona_komety', 
    'torbka_modelowanie', 'torebka_naciek', 
    'unaczynienie_brak', 'unaczynienie_obwodowe', 'unaczynienie_centralne', 'unaczynienie_mieszane', 
    'usg_azt',
    'wezly_chlonne_patologiczne',
    
    'lokalizacja_prawy_plat', 'lokalizacja_lewy_plat', 'lokalizacja_ciesn',
    
    'hp_ptc', 'hp_ftc', 'hp_hurthlea', 'hp_mtc', 'hp_dobrze_zroznicowane', 'hp_ana', 'hp_plasko',	
    'hp_ruczolak', 'hp_guzek_rozrostowy', 'hp_zapalenie', 'hp_nieokreslone', 'hp_niftp', 'hp_wdump','hp_ftump',
    'rak']



FEATURES_PL = [
    
    'echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_izoechogeniczna', 'echo_hiperechogeniczna','echo_mieszana',
    'budowa_lita', 'budowa_lito_plynowa', 'budowa_plynowo_lita',
    'ksztalt_owalny', 'ksztalt_okragly', 'ksztalt_nieregularny', 
    'granice_zatarte', 'brzegi_katowe', 'brzegi_mikrolobularne', 'brzegi_spikularne',
    'halo', 'halo_cienka', 'halo_gruba',
    'zwapnienia_mikrozwapnienia', 'zwapnienia_makro_obraczkowate', 'zwapnienia_makrozwapnienia', 'zwapnienia_artefakty_typu_ogona_komety',
    'torebka_naciek', 'torbka_modelowanie',
    'unaczynienie_brak', 'unaczynienie_obwodowe', 'unaczynienie_mieszane','unaczynienie_centralne',
    'usg_azt'
]

VARIABLES_PL = [
    'echo_gleboko_hipo', 'echo_izoechogeniczna', 'budowa_lita', 'budowa_lito_plynowa', 'ksztalt_owalny',
    'ksztalt_nieregularny', 'granice_zatarte', 'brzegi_mikrolobularne', 'halo', 'zwapnienia_mikrozwapnienia',
    'zwapnienia_makrozwapnienia', 'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe', 'unaczynienie_mieszane'
]


VARIABLES_ENG = [
    'hypoechoic', 'isoechoic', 'solid', 'fluid-filled', 'oval',
    'irregular', 'boundaries blurred', 'microlobular', 'halo', 'microcalcifications',
    'macrocalcifications', 'infiltrative capsule', 'no vascularity', 'peripheral vascularity', 'mixed vascularity'
]

VARIABLES_DICT = dict(zip(VARIABLES_PL, VARIABLES_ENG))


# VARIABLES_EXT = ['brzegi_mikrolobularne','echo_gleboko_hipo', 'granice_zatarte']
# VARIABLES_EXT = ['granice_zatarte']

def generte_model_config(_res_filename):
    
    cfg = get_model_config();
      
    histories = pd.DataFrame(columns =[
        "model_name", "learning_rate", "batch_size", "optimizer", "loss_function", "features", "img_size", "channels", "augment", "filter"])
        
    for _, r in cfg.iterrows():
        
        new_row = {
            "model_name": r["model_name"], 
            "learning_rate": r["learning_rate"], 
            "batch_size": r["batch_size"], 
            "optimizer": r["optimizer"], 
            "loss_function": r["loss_function"], 
            "features": r["features"], 
            "img_size": r["img_size"], 
            "channels": r["channels"], 
            "augment": 0, 
            "filter": r["filter"]
            }
        
        histories = pd.concat([histories, pd.DataFrame([new_row])], ignore_index=True)
    
    histories.to_csv(_res_filename, mode='w', header=True, index=False, sep=';')
                    

def get_model_config():
    
    #learning_rates = [0.01, 0.005]
    learning_rates = [0.005]
    
    #batch_sizes = [8, 16, 32]
    batch_sizes = [8]
        
    # optimizers = ['Adam', 'SGD']
    optimizers = ['SGD'] 

    #losses = ['focal_loss', 'binary_crossentropy', 'squared_hinge', 'categorical_hinge', 'kl_divergence', 'categorical_crossentropy' ]
    losses = ['categorical_crossentropy']
    
    img_sizes = [140]
    #img_sizes = [80]
    
    filters = ['heat']
    channels = [3]
    
    # features_sets = [
    #     ['brzegi_mikrolobularne'],
    #     ['brzegi_mikrolobularne','echo_gleboko_hipo'],
    #     ['brzegi_mikrolobularne','echo_gleboko_hipo', 'zwapnienia_mikrozwapnienia'],
    #     ['brzegi_mikrolobularne','echo_gleboko_hipo', 'granice_zatarte']
    #     ]
    
    features_sets = [["none"]]
    models = ['cnn1', 'cnn2', 'cnn3', 'VGG16', 'VGG19', 'denseNet121', 'denseNet201']
    # models = ['cnn3']
    res = pd.DataFrame(columns = ["model_name", "learning_rate", "batch_size", "optimizer", "loss_function", "img_size", "channels"])

    for model in models:
        for l in learning_rates:
            for b in batch_sizes:
                for o in optimizers:     
                    for lo in losses:
                        for fi in filters:
                            for img_size in img_sizes:
                                for channel in channels:
                                    for features in features_sets:
                                        new_row = {'model_name':model, 
                                                   'learning_rate':l, 
                                                   'batch_size':b, 
                                                   'optimizer':o, 
                                                   'loss_function': lo, 
                                                   'filter': fi, 
                                                   'features': features,
                                                   'img_size': img_size, 
                                                   'channels': channel}
                                        res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
                        
    return res


RESULT_COLUMNS = [
    "model_name", 
    "learning_rate", "batch_size", "optimizer", "loss_function", 
    "features",
    "img_size", "augment", "filter", "epochs", 
    "train_dataset_size", "test_dataset_size", "train_target_ratio", "test_target_ratio", "train_target_cases", "test_target_cases",
    "accuracy", "auc", "f1", "sensitivity", "specificity", "precision", "threshold",
    "run_date", "elapsed_mins", "iteration", "status"
    ]

RESULT_DETAILS_COLUMNS = ['model_name', 'iteration', 'id_coi', 'actual', 'threshold', 'predict', 'predict_binary', 'image', 'status']

VARIABLES_TRANSLATE = {
    "echo_nieznacznie_hipo": "mildly_hypoechoic",
    "echo_gleboko_hipo": "markedly_hypoechoic",
    "echo_hiperechogeniczna": "hyperechoic",
    "echo_izoechogeniczna": "isoechoic",
    "echo_mieszana": "mixed_echogenicity",

    "budowa_lita": "structure_solid",
    "budowa_lito_plynowa": "structure_mixed_solid_cystic",
    "budowa_plynowo_lita": "structure_mixed_cystic_solid",

    "ksztalt_owalny": "shape_ovoid",
    "ksztalt_okragly": "shape_round",
    "ksztalt_nieregularny": "shape_irregular",

    "orientacja_rownolegla": "orientation_parallel",

    "granice_rowne": "margins_smooth",
    "granice_zatarte": "margins_ill_defined",
    "granice_nierowne": "margins_irregular",

    "brzegi_katowe": "margins_angular",
    "brzegi_mikrolobularne": "margins_microlobulated",
    "brzegi_spikularne": "margins_spiculated",

    "halo_cienka": "halo_thin",
    "halo": "halo",
    "halo_gruba": "halo_thick",

    "zwapnienia_mikrozwapnienia": "microcalcifications",
    "zwapnienia_makrozwapnienia": "macrocalcifications",
    "zwapnienia_makro_obraczkowate": "calcifications_ring_shaped",

    "zwapnienia_artefakty_typu_ogona_komety": "calcifications_comet_tail_artifacts",

    "torbka_modelowanie": "shaping_of_the_gland_and_capsule",
    "torebka_naciek": "extrathyroidal_invasion",

    "unaczynienie_brak": "no_vascularity",
    "unaczynienie_obwodowe": "peripheral_vascularity",
    "unaczynienie_centralne": "central_vascularity",
    "unaczynienie_mieszane": "mixed_vascularity",

    "usg_azt": "autoimmune_disease",
    "wezly_chlonne_patologiczne": "metastatic_pathological_lymph_nodes",

    "lokalizacja_prawy_plat": "right_thyroid_lobe",
    "lokalizacja_lewy_plat": "left_thyroid_lobe",
    "lokalizacja_ciesn": "isthmus",
    
    'tirads_3': 'tirads_3', 
    'tirads_4': 'tirads_4',
    'tirads_5': 'tirads_5',
    'rak': 'cancer'
}

VARIABLES_TRANSLATE_PAPER = {
    "echo_nieznacznie_hipo": "mildly hypoechoic",
    "echo_gleboko_hipo": "markedly hypoechoic",
    "echo_hiperechogeniczna": "hyperechoic",
    "echo_izoechogeniczna": "isoechoic",
    "echo_mieszana": "mixed echogenicity",

    "budowa_lita": "solid structure",
    "budowa_lito_plynowa": "solidcystic structure",
    "budowa_plynowo_lita": "cysticsolid structure",

    "ksztalt_owalny": "ovoid shape",
    "ksztalt_okragly": "round shape",
    "ksztalt_nieregularny": "irregular shape",

    "orientacja_rownolegla": "parallel orientation",

    "granice_rowne": "smooth margins",
    "granice_zatarte": "ill-defined margins",
    "granice_nierowne": "irregular margins",

    "brzegi_katowe": "angular margins",
    "brzegi_mikrolobularne": "microlobulated margins",
    "brzegi_spikularne": "spiculated margins",

    "halo_cienka": "thin halo",
    "halo": "halo",
    "halo_gruba": "thick halo",

    "zwapnienia_mikrozwapnienia": "microcalcifications",
    "zwapnienia_makrozwapnienia": "macrocalcifications",
    "zwapnienia_makro_obraczkowate": "ringshaped calcifications",

    "zwapnienia_artefakty_typu_ogona_komety": "comet tail calcifications",

    "torbka_modelowanie": "no extrathyroidal invasion",
    "torebka_naciek": "extrathyroidal invasion",

    "unaczynienie_brak": "no vascularity",
    "unaczynienie_obwodowe": "peripheral vascularity",
    "unaczynienie_centralne": "central vascularity",
    "unaczynienie_mieszane": "mixed vascularity",

    "usg_azt": "autoimmune disease",
    "wezly_chlonne_patologiczne": "pathological lymph nodes",

    "lokalizacja_prawy_plat": "right lobe",
    "lokalizacja_lewy_plat": "left lobe",
    "lokalizacja_ciesn": "isthmus",
    
    'tirads_3': 'TIRADS 3', 
    'tirads_4': 'TIRADS 4',
    'tirads_5': 'TIRADS 5',
    'rak': 'cancer'
}
