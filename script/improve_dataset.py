import argparse, sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dlib_trainer import DlibPredictorTrainer

parser=argparse.ArgumentParser()

parser.add_argument('--path_csv', help='path to the csv landmark')
parser.add_argument('--path_video', help='path to the viddeo')
parser.add_argument('--path_xml', help='path to the xml dataset')


args=parser.parse_args()

dlib_predictor_trainer  = DlibPredictorTrainer(args.path_csv, args.path_video, args.path_xml)
dlib_predictor_trainer.manual_classify_image()