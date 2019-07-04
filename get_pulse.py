import argparse
import datetime
import sys

import numpy as np
from cv2 import moveWindow
import cv2 as cv

from lib.device import Camera
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from lib.processors_noopenmdao import FindFaceGetPulse


class GetPulseApp(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        self.flip = False
        self.flip_codes = [-1, 0, 1]
        self.flip_index = 0
        self.transpose = False

        # Imaging device - must be a connected camera
        # (not an ip camera or mjpeg stream)
        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break

        self.cameras.append(Camera('/home/mlima/Downloads/caren01.mp4'))
        self.cameras.append(Camera('/home/mlima/Downloads/caren02.mp4'))
        self.cameras.append(Camera('/home/mlima/Downloads/caren03.mp4'))
        self.cameras.append(Camera('/home/mlima/Downloads/caren04.mp4'))

        self.w, self.h = 0, 0
        self.pressed = 0
        # Containerized analysis of recieved image frames
        # (an openMDAO assembly) is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = FindFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv,
                             "r": self.toggle_flip,
                             "g": self.change_flip,
                             "t": self.toggle_transpose,
                             }

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True

            if self.bpm_plot:
                destroyWindow(self.plot_title)
            self.bpm_plot = False
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def toggle_flip(self):
        self.flip = not self.flip

    def change_flip(self):
        self.flip_index = (self.flip_index + 1) % len(self.flip_codes)

    def toggle_transpose(self):
        self.transpose = not self.transpose

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing csv")

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()

        if self.flip:
            frame = cv.flip(frame, self.flip_codes[self.flip_index])

        if self.transpose:
            frame = cv.transpose(frame)

        self.h, self.w, _c = frame.shape

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame

        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)

        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        # handle any key presses
        self.key_handler()

    def run(self):
        while True:
            self.main_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    args = parser.parse_args()
    App = GetPulseApp(args)

    App.run()
