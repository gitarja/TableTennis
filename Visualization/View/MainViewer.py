from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QSizePolicy
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from pyqtgraph import opengl as gl

import pyqtgraph as pg
import numpy as np
import time

from time import perf_counter
import glob


class Ui_MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.colors = [(0.4, 0.76078431, 0.64705882, 1.0), (0.98823529, 0.55294118, 0.38431373, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0),
                       (0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0),
                       (0.0, 1.0, 0.0, 1.0)]

        self.setupUi(self)
        self.setupGraph()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.central_widget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))

        # Add central widget
        self.plot_widget = QtWidgets.QWidget(self.central_widget)
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)

        # Add skeleton widget
        self.plot_widget.setGeometry(QtCore.QRect(0, 5, 1920, 700))
        self.graph = gl.GLViewWidget()
        self.plot_layout.addWidget(self.graph)

        # Add ecg plot widget
        self.ecgplot_widget = QtWidgets.QWidget(self.central_widget)
        self.ecgplot_layout = QtWidgets.QVBoxLayout(self.ecgplot_widget)

        self.ecgplot_widget.setGeometry(QtCore.QRect(0, 701, 1920, 250))


        # Play/Pause Button

        self.controller_widget = QtWidgets.QWidget(self.central_widget)
        self.controller_layout = QtWidgets.QHBoxLayout(self.plot_widget)
        self.controller_widget.setGeometry(QtCore.QRect(0, 950, 1920, 250))

        self.play_button = QtWidgets.QPushButton(self.controller_widget)
        self.play_button.setGeometry(QtCore.QRect(10, 0, 35, 30))
        self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self.play_button.setEnabled(False)
        self.controller_layout.addWidget(self.play_button)

        self.frame_slider = QtWidgets.QSlider(self.controller_widget)
        self.frame_slider.setGeometry(QtCore.QRect(46, 0, 1885, 30))
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("timeSlider")
        self.frame_slider.setEnabled(False)
        self.frame_slider.setValue(-1)
        self.frame_slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setSingleStep(1)

        self.controller_layout.addWidget(self.frame_slider)


        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.file_menu = QtWidgets.QMenu("File")
        self.menubar.addMenu(self.file_menu)

        self.project_menu = QtWidgets.QMenu("Project")
        self.menubar.addMenu(self.project_menu)

        self.analyze_menu = QtWidgets.QMenu("Analyze")
        self.menubar.addMenu(self.analyze_menu)

        '''
        Files action
         - Load single: load a single project consisting Vicon, ECG, and Tobii files
         - Load multiple: load multiple projects
        '''

        self.loadsingle_action = QtWidgets.QAction("Load Single Trial")
        self.file_menu.addAction(self.loadsingle_action)

        self.loadmul_action = QtWidgets.QAction("Load Multiple Trials")
        self.file_menu.addAction(self.loadmul_action)

        self.openproject_action = QtWidgets.QAction("Open")
        self.save_action = QtWidgets.QAction("Save")
        self.project_menu.addAction(self.openproject_action)
        self.project_menu.addAction(self.save_action)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainUI"))

    def setupGraph(self):

        try:
            # Add skeleton and joints
            self.skeleton_scttrplt = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=np.ones((1,)),
                                                          color=np.zeros((1, 4)), pxMode=False)
            self.graph.addItem(self.skeleton_scttrplt)
            self.joint_lineplt = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1., 1, 1., 1.), width=0.75,
                                                   antialias=True, mode="lines")
            self.graph.addItem(self.joint_lineplt)

            # Add eye
            self.eye_scttrplt = gl.GLScatterPlotItem(pos=[[0., 0., 10.]], size=np.ones((1,)),
                                                     color=(1.0, 0.0, 1.0, 1.0),
                                                     pxMode=False)
            self.graph.addItem(self.eye_scttrplt)
            self.eye_lineplt = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.0, 1.0, 1.0), width=0.5,
                                                 antialias=True,
                                                 mode="lines")
            self.graph.addItem(self.eye_lineplt)

            # Add wall
            self.wall_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.0, 0.0, 1.0, 1.0), width=0.5,
                                                 antialias=True,
                                                 mode="lines")
            self.graph.addItem(self.wall_line)

            # Add table
            self.table_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0., .0, 1.0), width=0.5,
                                               antialias=True,
                                               mode="lines")
            self.graph.addItem(self.table_line)
            # Add racket 1
            self.racket1_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.0, 1.0, 0.0, 1.0), width=0.5,
                                                     antialias=True,
                                                     mode="lines")
            self.graph.addItem(self.racket1_line)

            # Add racket 2
            self.racket2_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.0, 1.0, 0.0, 1.0), width=0.5,
                                                  antialias=True,
                                                  mode="lines")
            self.graph.addItem(self.racket2_line)

            # Add ball
            self.ball_scttrplt = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=0.7,
                                                          color=(1.0, 1.0, 1.0, 1.0), pxMode=False)
            self.graph.addItem(self.ball_scttrplt)

            # Add ball_area
            self.ball_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 1.0, 0.0, 1.0), width=0.5,
                                                  antialias=True,
                                                  mode="lines")
            self.graph.addItem(self.ball_line)

        except:
            pass

        # Add floor
        # gz = gl.GLGridItem(size=QtGui.QVector3D(100, 100, 1))
        # gz.translate(0, 0, 0)
        # self.graph.addItem(gz)

        # add ECG plotting

        self.ecg_graph = pg.PlotWidget(enableMouse=False)
        self.ecg_graph.setYRange(400, 900)
        self.ecg_graph.enableMouse(False)
        self.ecg_graph.getPlotItem().hideAxis('bottom')
        self.ecg_graph.getPlotItem().setMouseEnabled(x=False, y=False)
        self.ecgplot_layout.addWidget(self.ecg_graph)

        self.ecg_plot = None
        self.ecg_data = None



    def folderDialog(self, path):
        home_dir = str(path)
        fname = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open file', home_dir)
        return fname
    def setECGPlot(self, n_sub, ecg_data):

        self.ecg_plot = [self.ecg_graph.plot(pen=pg.mkPen((np.array(self.colors[i][:-1]) * 255).tolist(), width=3)) for i in
                         range(n_sub)]
        self.ecg_data = ecg_data

    def updateGaze(self, data):

        gaze_data, left_eye_data, right_eye_data = data
        gaze_positions = []
        gaze_directions = []
        colors_gaze = []
        for i in range(len(gaze_data)):
            gaze_positions.append([left_eye_data[i], right_eye_data[i], gaze_data[i]])
            gaze_directions.append([left_eye_data[i], gaze_data[i], right_eye_data[i], gaze_data[i]])
            colors_gaze.append(self.colors[i])

        self.eye_scttrplt.setData(pos=np.array(gaze_positions).reshape((-1, 3)), size=0.3)
        self.eye_lineplt.setData(pos=np.array(gaze_directions).reshape((-1, 3)))

    def updateSkeleton(self, data):
        trajectories_data, segments_data = data

        trajectories = []
        colors_trajectories = []

        segments = []
        colors_segments = []

        for i in range(len(trajectories_data)):
            trajectories.append(trajectories_data[i])
            segments.append(segments_data[i])
            colors_trajectories.append((self.colors[i],) * 37) # total number of marker is 37
            colors_segments.append((self.colors[i],) * 32) # total number of segments is 32

        self.skeleton_scttrplt.setData(pos=np.array(trajectories).reshape((-1, 3)), size=0.25,
                                       color=np.array(colors_trajectories).reshape((-1, 4)))
        self.joint_lineplt.setData(pos=np.array(segments).reshape((-1, 3)), width=1.5,
                                   color=np.array(colors_segments).reshape((-1, 4)))


    def updateObjects(self, data):
        wall_data, table_data, racket1_data, racket2_data = data

        wall_data =  np.array(wall_data).reshape((-1, 3))
        table_data = np.array(table_data).reshape((-1, 3))
        racket1_data = np.array(racket1_data).reshape((-1, 3))


        wall = wall_data[[0,2,2,1,1,3,3,0]]
        table = table_data[[0,2,2,1,1,3,3,0]]
        racket1 = racket1_data[[0,2,2,1,1,3,3,0]]


        self.wall_line.setData(pos=wall, width=1.5)
        self.table_line.setData(pos=table, width=1.5)
        self.racket1_line.setData(pos=racket1, width=1.5)
        if racket2_data is not None:
            racket2_data = np.array(racket2_data).reshape((-1, 3))
            racket2 = racket2_data[[0, 2, 2, 1, 1, 3, 3, 0]]
            self.racket2_line.setData(pos=racket2, width=1.5)

    def updateBallArea(self, data):
        bot_idx_line = [0, 1, 1, 3, 3, 2, 2, 0]
        up_idx_line = (4 + np.array(bot_idx_line)).tolist()
        idx_line = bot_idx_line + up_idx_line + [0, 4, 1, 5, 3, 7, 2, 6]
        data = np.array(data).reshape((-1, 3))[idx_line]
        self.ball_line.setData(pos=data)

    def updateBall(self, data):
        self.ball_scttrplt.setData(pos=np.array(data))
        # if np.sum(np.isnan(data)) == 0:
        #     self.ball_scttrplt.setData(pos=np.array(data))
        # else:
        #     self.ball_scttrplt.setData(pos=np.array(np.zeros((1, 3))))



    def updateECG(self, idx):
        idx_start = idx - 30
        if idx < 31:
            idx_start = 0

        for i in range(len(self.ecg_data)):
            self.ecg_plot[i].setData(self.ecg_data[i][idx_start:idx+1])
        # for i in range(len(data)):
        #     if len(self.ecg_data[i]) > 30:
        #         del self.ecg_data[i][0]
        #     self.ecg_data[i].append(data[i])
        #     self.ecg_plot[i].setData(self.ecg_data[i])


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    # ui.feeder.start()
    sys.exit(app.exec_())
