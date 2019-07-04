import os
import sys
import time

import cv2
import numpy as np


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class FindFaceGetPulse(object):

    def __init__(self,
                 bpm_limits=[],
                 data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        # self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = None
        self.subface_rect = None
        self.set_face_rect([1, 1, 2, 2])
        self.last_frame = None
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def _get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained

    def _draw_main_menu(self, cam, color):
        cv2.putText(self.frame_out,
                    "Press 'C' to change camera (current: %s)" % str(cam),
                    (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, color)

        cv2.putText(self.frame_out,
                    "Press 'S' to lock face and begin",
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, color)

        cv2.putText(self.frame_out,
                    "Press 'Esc' to quit",
                    (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, color)

    def _draw_sub_menu(self, cam, color):
        cv2.putText(self.frame_out,
                    "Press 'C' to change camera (current: %s)" % str(cam),
                    (10, 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.25,
                    color)
        cv2.putText(self.frame_out, "Press 'S' to restart",
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, color)

        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                    (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, color)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, color)

    def do_find_faces(self, cam, col, gray):
        self._draw_main_menu(cam, col)

        self.data_buffer, self.times, self.trained = [], [], False
        detected = list(
            self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key=lambda a: a[-1] * a[-2])

            if self.shift(detected[-1]) > 10:
                self.set_face_rect(detected[-1])

        forehead1 = self.subface_rect
        self.draw_rect(self.face_rect, col=(255, 0, 0))

        x, y, w, h = self.face_rect
        cv2.putText(self.frame_out, "Face",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        self.draw_rect(forehead1)

        x, y, w, h = forehead1
        cv2.putText(self.frame_out, "Forehead",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        return

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        r, g, b = cv2.split(self.frame_in)
        gray = cv2.equalizeHist(g)
        color = (100, 255, 100)
        if self.find_faces:
            self.do_find_faces(cam, color, gray)
            return

        if set(self.face_rect) == {1, 1, 2, 2}:
            return

        self._draw_sub_menu(cam, color)

        if self.last_frame is not None:
            self.ajust_phase_correlation(gray)
        else:
            self.last_frame = np.float64(gray)

        forehead1 = self.subface_rect
        self.draw_rect(forehead1)

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals)
        data_buffer_len = len(self.data_buffer)
        if data_buffer_len > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            data_buffer_len = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if data_buffer_len > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(data_buffer_len) / (
                    self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1],
                                     data_buffer_len)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(data_buffer_len) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / data_buffer_len * \
                         np.arange(data_buffer_len / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            if len(pruned) == 0:
                return

            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.subface_rect
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            color = (100, 255, 100)
            gap = (self.buffer_size - data_buffer_len) / self.fps

            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())

            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % self.bpm

            tsize = 1
            cv2.putText(self.frame_out, text,
                        (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN,
                        tsize, color)

    def set_face_rect(self, face_rect):
        self.face_rect = face_rect
        self.subface_rect = self._get_subface_coord(0.5, 0.18, 0.25, 0.15)

    def ajust_phase_correlation(self, gray):
        current = np.float64(gray)
        ret = cv2.phaseCorrelate(self.last_frame, current)
        self.last_frame = current
        ajusted = np.array([np.int(self.face_rect[0] - ret[0][0]),
                            np.int(self.face_rect[1] - ret[0][1]),
                            self.face_rect[2], self.face_rect[3]],
                           dtype=np.int32)
        self.set_face_rect(ajusted)
