from video_transforme import  VideoToLandmarks

vl = VideoToLandmarks("data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H64.mp4")
vl.load_data_video()
vl.transoform_videos_with_sec_to_landmarks("hog", False, 5*60)
