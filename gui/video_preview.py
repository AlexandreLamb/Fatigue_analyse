import time
import tkinter as tk
import imageio
from PIL import Image, ImageTk

global pause_video

# download video at: http://www.html5videoplayer.net/videos/toystory.mp4
video_name = "data/data_in/videos/IRBA_extrait_1.mp4"
video = imageio.get_reader(video_name)



class VideoReader(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=600, height=400, bg="yellow")
        tk.Frame.pack(self, padx=(0,300), pady=200)
        tk.Frame.pack_propagate(self,0)
        self.my_label = tk.Label(self)
        self.my_label.pack()
        self.start_button = tk.Button(self, text='start',  command=self._start).pack(side=tk.LEFT)
        self.pause_button = tk.Button(self, text='stop', command=self._stop).pack(side=tk.LEFT)
        self.pause_video = True
        self.video_name = ""
    
    # from https://stackoverflow.com/questions/57158514/how-to-display-video-preview-in-tkinter-window
    def video_frame_generator(self):
        def current_time():
            return time.time()

        start_time = current_time()
        _time = 0
        for frame, image in enumerate(video.iter_data()):

            # turn video array into an image and reduce the size
            image = Image.fromarray(image)
            image.thumbnail((750, 750), Image.ANTIALIAS)

            # make image in a tk Image and put in the label
            image = ImageTk.PhotoImage(image)

            # introduce a wait loop so movie is real time -- asuming frame rate is 24 fps
            # if there is no wait check if time needs to be reset in the event the video was paused
            _time += 1 / 24
            run_time = current_time() - start_time
            while run_time < _time:
                run_time = current_time() - start_time
            else:
                if run_time - _time > 0.1:
                    start_time = current_time()
                    _time = 0

            yield frame, image


    def _stop(self):
        self.pause_video = True


    def _start(self):
        self.pause_video = False


if __name__ == "__main__":

    root = tk.Tk()
    root.title('Video in tkinter')

    my_label = tk.Label(root)
    my_label.pack()
    tk.Button(root, text='start', command=_start).pack(side=tk.LEFT)
    tk.Button(root, text='stop', command=_stop).pack(side=tk.LEFT)

    pause_video = False
    movie_frame = video_frame_generator()

    while True:
        if not pause_video:
            frame_number, frame = next(movie_frame)
            my_label.config(image=frame)

        root.update()

    root.mainloop()