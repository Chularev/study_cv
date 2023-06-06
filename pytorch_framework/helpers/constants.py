from pathlib import Path



ROOT_DIR = str( Path(__file__).absolute().parent.parent )
IMG_DIR = ROOT_DIR + "/data/data/images"
LABEL_DIR = ROOT_DIR + "/data/data/labels"
TRAIN_CSV_FILE = ROOT_DIR + "/data/train.csv"
VAL_CSV_FILE = ROOT_DIR + "/data/test.csv"


if __name__ == "__main__":
    print('ROOT_DIR = ', ROOT_DIR)
    print("IMG_DIR = ", IMG_DIR)
    print("LABEL_DIR = ", LABEL_DIR)
    print("TRAIN_CSV_FILE = ", TRAIN_CSV_FILE)
    print("VAL_CSV_FILE", VAL_CSV_FILE)