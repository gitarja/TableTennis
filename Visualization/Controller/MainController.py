import subprocess
import os
import sys

import pandas as pd

from Visualization.View.MainViewer import Ui_MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets
import numpy as np
from PyQt5 import QtCore, QtGui
from Utils.DataReader import ViconReader, TobiiReader, ECGReader, BallReader, SubjectObjectReader
from Visualization.View.UtilsViewer import showErrorMessage
import time
import glob
from pathlib import Path


class Feeder(QtCore.QThread):
    human_trajectories_points = QtCore.pyqtSignal(list)
    object_trajectories_points = QtCore.pyqtSignal(list)
    ball_trajectories_points = QtCore.pyqtSignal(list)
    ballarea_trajectories_points = QtCore.pyqtSignal(list)
    ecg_points = QtCore.pyqtSignal(list)
    gaze_points = QtCore.pyqtSignal(list)
    idx_value =  QtCore.pyqtSignal(int)

    def __init__(self):
        super(Feeder, self).__init__()
        self.pause = False


    def setPause(self, b):
        self.pause = b


    def setData(self, sub_data, obj_data, ecg_data, tobii_data, ball_data):
        self.sub_data = sub_data
        self.racket_2 = None
        self.racket_1 = None
        self.wall = None
        self.table = None
        for obj in obj_data:
            if (obj["name"] == 'Racket1'):
                self.racket_1 = obj
            elif (obj["name"] == 'Racket2') | (obj["name"] == 'Racket1a'):
                self.racket_2 = obj
            elif obj["name"] == 'Wall':
                self.wall_mean = np.nanmean(obj["trajectories"], 0)
            elif obj["name"] == 'Table':
                self.table_mean = np.nanmean(obj["trajectories"], 0)

        self.table_mean[4] = self.wall_mean[4]
        self.table_mean[7] = self.wall_mean[4]

        # self.ball_area = np.array([
        #     [-793.04703234, 500.82939009, 732.0650296],
        #     [774.58230762, 500.82939009, 749.83444349],
        #     [-766.44288645, 2407.78545434, 742.19295307],
        #     [797.48482387, 2327.98461769, 761.78510793],
        #
        #     [-793.04703234, 1007.82939009, 1666.323361],
        #     [774.58230762, 955.18120783, 1642.58802859],
        #     [-688.01776107, 2407.78545434, 2000.58802859],
        #     [876.92080693, 2348.65824315, 2000.58802859],
        #
        # ])


        # relocated table
        self.ball_area = np.array([
            [-749.966797, 117.712341, 726.281189],  # table pt1_x - 60, table pt1_y - 400, table pt1_z
            [817.196533, 104.012634, 746.193665],  # table pt4_x  - 60, table pt4_y - 400, table pt4_z
            [-749.386292, 1860.348145, 739.174377],  # table pt3_x, table pt3_y + 600, table pt3_z
            [814.946838, 1860.348145, 739.174377],  # table pt2_x, table pt2_y + 600, table pt2_z

            [-749.966797, 217.712341, 2036.201416],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
            [817.196533, 204.012634, 2036.201416],  # table pt4_x  - 60, table pt4_y, table pt4_z * 2
            [-690.061218, 1947.592773, 2036.201416],  # wall pt4_x, wall pt4_y, wall pt4_z + 400
            [877.275452, 1930.623779, 2036.201416],  # wall pt1_x, wall pt1_y, wall pt1_z + 400

        ])

        self.ecg_data = ecg_data
        self.tobii_data = tobii_data
        self.ball_data = ball_data
        self.n_data = len(sub_data[0]["trajectories"])

        self.idx_nexus = 0
        self.idx_ecg = 0
        self.idx_tobii = 0
        vicon_reader = ViconReader()
        self.tobii_reader = TobiiReader()

        # self.human_segments = [vicon_reader.constructSegments(d["segments"], vicon_reader.SEGMENTS_PAIR, vicon_reader.SEGMENTS_IDX) for d in self.sub_data]
        # self.segments_rot = [vicon_reader.getSegmentsRot(d["segments"]) for d in self.sub_data]

        self.human_segments = [vicon_reader.constructSegments(d["segments"].values, vicon_reader.SEGMENTS_PAIR, vicon_reader.SEGMENTS_IDX) for d in self.sub_data]
        self.segments_rot = [vicon_reader.getSegmentsRot(d["segments"].values) for d in self.sub_data]


    def run(self):
        """Long-running task."""
        while (True):
            if not self.pause:
                if self.idx_nexus < self.n_data:
                    self.sendData()
                    self.idx_nexus +=1
                    # ---------------sleep--------#
                    time.sleep(1. / 100.)
                else:
                    break
            else:
                time.sleep(1. / 100.)


    def sendData(self):
        human_trajectories = [d["trajectories"].values[self.idx_nexus, :123] / 100 for d in self.sub_data]
        human_segments = [s[self.idx_nexus] / 100 for s in self.human_segments]
        wall_trajectories = self.wall_mean / 100
        table_trajectories = self.table_mean/ 100
        racket1_trajectories = self.racket_1["trajectories"].values[self.idx_nexus]/ 100


        ball_trajectories = self.ball_data[self.idx_nexus] / 100
        ball_area_trajectories = self.ball_area / 100
        self.human_trajectories_points.emit([human_trajectories, human_segments])
        if self.racket_2 is not None:
            racket2_trajectories = self.racket_2["trajectories"].values[self.idx_nexus] / 100
        else:
            racket2_trajectories = None
        self.object_trajectories_points.emit([wall_trajectories, table_trajectories, racket1_trajectories, racket2_trajectories])

        # print(self.idx_nexus)

        self.ball_trajectories_points.emit(ball_trajectories.tolist())
        # self.ballarea_trajectories_points.emit(ball_area_trajectories.tolist())

        # if self.idx_nexus % 100 == 0:
        #     if self.idx_ecg < len(self.ecg_data):
        #         rr = self.ecg_data.iloc[self.idx_ecg]["RR"]
        #         rr_points = rr if len(rr) > 1 else [rr]
        #         self.ecg_points.emit(rr_points)
        #         self.idx_value.emit(self.idx_ecg)
        #         self.idx_ecg += 1




        tobii_segments = np.array([h[15] for h in human_segments])  # 15 is the index of tobii segment
        # print(tobii_segments)
        # gaze_inf = np.array([np.array(g)[2:17] / 100 for g in self.tobii_data.iloc[self.idx_tobii]["Gaze"]])
        gaze_inf = np.array([np.array(g[self.idx_nexus]) / 100 for g in self.tobii_data])
        # tobii_rot = np.array([np.array(h) for h in self.tobii_data.iloc[self.idx_tobii]["Head"]])
        tobii_rot = np.array([r[self.idx_nexus][-1] for r in self.segments_rot])

        gaze = self.tobii_reader.local2GlobalGaze(gaze_inf[:, 0:3], tobii_segments, tobii_rot)

        eye_left = self.tobii_reader.local2GlobalGaze(gaze_inf[:, 3:6], tobii_segments, tobii_rot)
        # print(eye_left)
        eye_right = self.tobii_reader.local2GlobalGaze(gaze_inf[:, 6:9], tobii_segments, tobii_rot)
        # print(eye_right)
        self.gaze_points.emit(
                    [np.array(gaze).tolist(), np.array(eye_left).tolist(), np.array(eye_right).tolist()])
        # self.gaze_points.emit((np.array(gaze) + np.array(tobii_segments)).tolist())
        self.idx_tobii += 1

    def setIndex(self, idx):
        self.idx_ecg = idx
        self.idx_nexus = idx * 100
        self.idx_tobii = idx * 50
        self.sendData()


class MainController:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.view = Ui_MainWindow()

        # # Feeder
        self.feeder = Feeder()
        self.feeder.human_trajectories_points.connect(self.view.updateSkeleton)
        self.feeder.object_trajectories_points.connect(self.view.updateObjects)
        self.feeder.ball_trajectories_points.connect(self.view.updateBall)
        self.feeder.ballarea_trajectories_points.connect(self.view.updateBallArea)

        self.feeder.gaze_points.connect(self.view.updateGaze)
        self.feeder.idx_value.connect(self.updateSliderValue)
        self.feeder.idx_value.connect(self.view.updateECG)

        # Reader
        # self.vicon_reader = ViconReader()
        # self.obj_reader = SubjectObjectReader()
        # self.tobii_reader = TobiiReader()
        # self.ecg_reader = ECGReader()
        # self.ball_reader = BallReader()
        self.reader = SubjectObjectReader()
        # self.c3d_reader = C3dReader()

        # menu action
        self.view.file_menu.triggered.connect(self.showFileDialog)
        self.view.play_button.clicked.connect(self.playPause)

        self.view.frame_slider.valueChanged.connect(self.updateGraphByClick)

        # play/pause
        self.isPause = False


    def updateSliderValue(self, value):
        self.view.frame_slider.setSliderPosition(value)
        self.view.statusbar.showMessage("no frame: "+ str(value))
        self.view.frame_slider.setTracking(False)

    def showFileDialog(self):
        fname = self.view.folderDialog(Path.home())
        if fname == "":
            return
        self.openTrial(fname)

    def updateGraphByClick(self):
        value = self.view.frame_slider.value()

        if value > -1:
            self.view.play_button.setIcon(self.view.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.isPause = True
            self.feeder.setPause(True)
            self.feeder.setIndex(int(value))






    def playPause(self):
        if self.isPause:
            self.view.play_button.setIcon(self.view.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self.isPause = False
            self.feeder.setPause(False)
        else:
            self.view.play_button.setIcon(self.view.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.isPause = True
            self.feeder.setPause(True)


    def openTrial(self, path):
        all_file = glob.glob(path + "\\*_complete_final.pkl")
        # nexus_file = glob.glob(path + "\\*_Nexus.pkl")
        # ecg_file = glob.glob(path + "\\*_ECG.csv")
        # tobii_file = glob.glob(path + "\\*_Tobii.tsv")
        # ball_file = glob.glob(path + "\\*_ball.csv")
        #
        # if len(ball_file) < 1:
        #     showErrorMessage("Ball file is not found")
        #     return
        # if len(nexus_file)< 1:
        #     showErrorMessage("Nexus file is not found")
        #     return
        # else:
        #     # obj, sub = self.vicon_reader.extractData(nexus_file[0])
        #     obj, sub = self.obj_reader.extractData(nexus_file[0])
        #     n_sub = len(sub)
        #     if len(ecg_file) != n_sub:
        #         showErrorMessage("Number of ECG files not match with the number of subjects")
        #         return
        #     if len(tobii_file) != n_sub:
        #         showErrorMessage("Number of Tobii files not match with the number of subjects")
        #         return
        #
        #     # read ECG and Tobii
        #     ecg_files = []
        #     tobii_files = []
        #     for s in sub:
        #         ecg_files.append(path + "\\"+s["name"]+"_ECG.csv")
        #         tobii_files.append(path+ "\\"+s["name"]+"_Tobii.tsv")
        #
        #     ecg_data = self.ecg_reader.extractData(ecg_files)
        #     tobii_data = self.tobii_reader.extractData(tobii_files)
        #     ball_data = self.ball_reader.extractData(ball_file[0])
        #     self.view.setECGPlot(n_sub, np.array(ecg_data["RR"].values.tolist()).T)

        obj, sub, ball, tobii = self.reader.extractData(all_file[0])
        ball_data = ball[0]["trajectories"].values
        # tobii_data = [tobii[0]["trajectories"].values[:, [3,4,5,12,13,14,15,16,17]], tobii[1]["trajectories"].values[:, [3,4,5,12,13,14,15,16,17]]]
        tobii_data = [tobii[0]["trajectories"].values[:, [3,4,5,12,13,14,15,16,17]]]
        ecg_data = np.ones((2, len(ball_data)))
        # Activate play button and slider
        self.view.play_button.setEnabled(True)
        self.view.frame_slider.setEnabled(True)
        self.view.frame_slider.setValue(0)
        # self.view.frame_slider.setRange(0, len(ecg_data))
        # # Feeder
        # # there is a 20 ms delay before the UDP is triggered
        self.feeder.setData(sub, obj, ecg_data, tobii_data, ball_data)
        self.feeder.start()

    def run(self):
        self.view.show()
        return self.app.exec_()

