from video_transforme import  VideoToLandmarks

path_array =  [
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_Go-NoGo_H71.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H63.mov",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H64.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H68.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H69.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H70.mp4",
"data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H71.mp4",
"data/data_in/videos/DESFAM_Semaine-2-Vendredi_PVT_H66_hog.mp4"]

for path in path_array:
    vl = VideoToLandmarks(path)
    vl.load_data_video()
    vl.transoform_videos_to_landmarks("hog", False)
