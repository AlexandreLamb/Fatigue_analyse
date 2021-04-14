import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from video_transforme import VideoToLandmarks

parser=argparse.ArgumentParser()


parser.add_argument('--path', help='path to the video folder')
parser.add_argument('--detector', help='name of detector : [mtcnn, hog]')
parser.add_argument('--min', help='number of min at begin and end to place landmarks')


args=parser.parse_args()

vl = VideoToLandmarks(args.path)

vl.load_and_transform_with_sec(args.detector, int(args.min))
