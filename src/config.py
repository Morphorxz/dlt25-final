# Constants and configurations for the learning rate scheduler project

TRAIN_SET = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]

TEST_SET = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]

FILES = TRAIN_SET + TEST_SET

FOLDER_PATHS = {
    "25": "/home/kairong/lr_drop_law/csv_25_converted/",
    "100": "/home/kairong/lr_drop_law/csv_100_converted/",
    "400": "/home/kairong/lr_drop_law/csv_400_converted/",
}

PARAMS = {
    "25": [3.04045406, 0.52468604, 0.50786857, 363.78751622, 2.06560812, 0.58279013, 0.64142257],
    "100": [2.6514477, 0.60115152, 0.45295811, 437.9464276,  2.13245612, 0.59785199, 0.65523644],
    "400": [2.37474466, 0.65421216, 0.42878731, 523.42464371, 2.02462735, 0.59350493, 0.63472457],
}

OPT_PATH = "./optimized_schedules/"  # Directory for optimized LR schedules

HUBER_DELTA = 0.001