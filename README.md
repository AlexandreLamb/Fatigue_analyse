# Fatigue_analyse

## 1. Install project dependancie 

```bash
pip install -r requierments.txt
```
## 2. Create data folder
### 2.1 downlaod ressource from the drive
 (https://drive.google.com/drive/folders/1sE6V3zjRnm4mZ49aJVEsO0qn6r4kk9px?usp=sharing)[Lien ressource drive]
### 2.1 create the data folder with this arborecense 

 * [data](./data)
   * [video](./data/video)
       * [IRBA_extrait_1.mp4](./video/IRBA_extrait_1.mp4)
   * [images](./data/image)
   * [shape_predictor](./shape_predictor)
          * [shape_predictor_68_face_landmarks.dat](./data/video/shape_predictor_68_face_landmarks.dat)

## 3. Convert video to frame 
```bash 
python  video_to_frame.py -v /data/video/IRBA_extrait_1.mp4
```
## 4. Place landmarks and save it on all frame to csv file
```bash
python images_to_landmarks.py
```
