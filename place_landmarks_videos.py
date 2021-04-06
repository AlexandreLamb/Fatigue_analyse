import argparse, sys
from video_transforme import VideoToLandmarks

parser=argparse.ArgumentParser()


parser.add_argument('--path', help='path to the video folder')
parser.add_argument('--detector', help='name of detector : [mtcnn, hog]')

args=parser.parse_args()

print(type(args.path))
vl = VideoToLandmarks(args.path)

vl.load_and_transform_with_sec(args.detector)
