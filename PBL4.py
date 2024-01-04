import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPainter, QColor, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolButton,
    QPushButton,
)
from PySide6.QtCore import QFile, Qt, QRect, Signal
from PySide6.QtSerialPort import QSerialPortInfo, QSerialPort
import serial
import math
import numpy as np
import cv2
import sympy as sp
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Ui_PBL4(object):
    def __init__(self):
        super(Ui_PBL4, self).__init__()
        self.ui = QUiLoader().load(QFile("PBL4.ui"))

        self.mode = False
        self.on = False
        self.move = True
        self.write = True

        self.serial1 = serial.Serial()

        self.target_size = 80, 80
        self.ui.icon_bk.setPixmap(
            QPixmap("./download.png").scaled(
                self.target_size[0], self.target_size[1], Qt.KeepAspectRatio
            )
        )
        self.ui.icon_ck.setPixmap(
            QPixmap("./download (1).png").scaled(
                self.target_size[0], self.target_size[1], Qt.KeepAspectRatio
            )
        )

        self.dis_d1 = 171
        self.dis_a2 = 211
        self.dis_a3 = 34
        self.dis_d4 = 228
        self.dis_d6 = 130

        self.joint_limits = [
            [math.radians(-90), math.radians(90)],
            [math.radians(-30), math.radians(110)],
            [math.radians(-84), math.radians(90)],
            [math.radians(-180), math.radians(180)],
            [math.radians(-90), math.radians(90)],
            [math.radians(-90), math.radians(90)],
        ]

        self.robot = DHRobot(
            [
                RevoluteDH(
                    a=0,
                    alpha=math.radians(90),
                    d=self.dis_d1,
                    qlim=[math.radians(-90), math.radians(90)],
                    offset=0,
                ),
                RevoluteDH(
                    a=self.dis_a2,
                    alpha=0,
                    d=0,
                    qlim=[math.radians(-30), math.radians(110)],
                    offset=0,
                ),
                RevoluteDH(
                    a=self.dis_a3,
                    alpha=math.radians(90),
                    d=0,
                    qlim=[math.radians(-84), math.radians(90)],
                    offset=0,
                ),
                RevoluteDH(
                    a=0,
                    alpha=math.radians(90),
                    d=self.dis_d4,
                    qlim=[math.radians(-180), math.radians(180)],
                    offset=0,
                ),
                RevoluteDH(
                    a=0,
                    alpha=math.radians(-90),
                    d=0,
                    qlim=[math.radians(-90), math.radians(90)],
                    offset=0,
                ),
                RevoluteDH(
                    a=0,
                    alpha=0,
                    d=self.dis_d6,
                    qlim=[math.radians(-90), math.radians(90)],
                    offset=0,
                ),
            ],
            name="robot",
        )

        self.list_write = []
        self.list_detect = []

        self.background = [0, 0, 40, 255, 255, 255]

        self.myColors = {
            "Red": [0, 70, 150, 22, 255, 255],
            "Yello": [21, 71, 156, 72, 255, 215],
            "Green": [43, 70, 40, 140, 255, 255],
        }
        self.step1 = 0
        self.step2 = 0
        self.step3 = 0
        self.step4 = 0

        self.theta1_current = 0
        self.theta2_current = 110
        self.theta3_current = -84
        self.theta4_current = 0

        self.theta1 = 0
        self.theta2 = 110
        self.theta3 = -84
        self.theta4 = 0
        self.theta5 = 0
        self.theta6 = 0
        self.gripper = 0

        self.ui.run.clicked.connect(self.click_run)
        self.ui.stop.clicked.connect(self.click_stop)

        self.ui.check.clicked.connect(self.check_port)
        self.ui.port.currentIndexChanged.connect(self.connect_serial)

    def check_port(self):
        if self.on == False:
            self.ui.port.clear()
            for ports in QSerialPortInfo.availablePorts():
                self.ui.port.addItem(ports.portName())

    def connect_serial(self):
        if self.on == False:
            if self.serial1.isOpen():
                self.serial1.close()
            self.serial1.port = self.ui.port.currentText()
            self.serial1.baudrate = 115200
            self.serial1.timeout = 1

            if not self.serial1.isOpen():
                self.serial1.open()

    def click_run(self):
        self.on = True
        self.mode = self.ui.checkBox.isChecked()
        self.control()
        self.auto()

    def click_stop(self):
        self.on = False

    def control(self):
        if self.on and self.mode == False:
            self.ui.Slider_theta1.valueChanged.connect(self.update_label)
            self.ui.Slider_theta2.valueChanged.connect(self.update_label)
            self.ui.Slider_theta3.valueChanged.connect(self.update_label)
            self.ui.Slider_theta4.valueChanged.connect(self.update_label)
            self.ui.Slider_theta5.valueChanged.connect(self.update_label)
            self.ui.Slider_theta6.valueChanged.connect(self.update_label)

            self.ui.linetheta1.editingFinished.connect(self.update_slider)
            self.ui.linetheta2.editingFinished.connect(self.update_slider)
            self.ui.linetheta3.editingFinished.connect(self.update_slider)
            self.ui.linetheta4.editingFinished.connect(self.update_slider)
            self.ui.linetheta5.editingFinished.connect(self.update_slider)
            self.ui.linetheta6.editingFinished.connect(self.update_slider)

            self.ui.nhap_thuan.clicked.connect(self.donghocthuan)
            self.ui.nhap_nghich.clicked.connect(self.donghocnghich)

    def auto(self):
        while self.on and self.mode:
            cap = cv2.VideoCapture(0)
            success, self.img = cap.read()
            self.img = self.img[70:325, 200:470]
            if self.img is None:
                break
            imgResult = self.img.copy()
            self.img = self.getContours(imgResult, self.img)
            self.show_wedcam(self.img)
            if cv2.waitKey(1) & self.ui.stop.isChecked():
                break

    def findColor(self, result, color):
        imgHSV = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(result, result, mask=mask)
        return imgResult

    def get_key_from_value(self, value):
        for key, val in self.myColors.items():
            if val == value:
                return key
        return None

    def getContours(self, imgResult, img):
        imgHSV1 = cv2.cvtColor(imgResult, cv2.COLOR_BGR2HSV)
        lower1 = np.array(self.background[0:3])
        upper1 = np.array(self.background[3:6])
        mask1 = cv2.inRange(imgHSV1, lower1, upper1)
        imgResult1 = cv2.bitwise_and(imgResult, imgResult, mask=mask1)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours1, hierarchy1 = cv2.findContours(
            img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for cnt1 in contours1:
            area1 = cv2.contourArea(cnt1)
            if area1 > 200:
                peri1 = cv2.arcLength(cnt1, True)
                approx1 = cv2.approxPolyDP(cnt1, 0.02 * peri1, True)
                x1, y1, w1, h1 = cv2.boundingRect(approx1)

                for color in self.myColors.values():
                    imgResult = self.findColor(imgResult1, color)
                    imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(
                        imgResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area1 - area > 100 and area > 100:
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                            x, y, w, h = cv2.boundingRect(approx)
                            objCor = len(approx)

                            if objCor > 6:
                                aspRatio = w / float(h)
                                if aspRatio > 0.98 and aspRatio < 1.03:
                                    objectType = "Square"
                                else:
                                    objectType = "Rectangle"
                            elif objCor < 6:
                                objectType = "Circles"
                            else:
                                objectType = "None"

                            cv2.rectangle(
                                img,
                                (x1, y1),
                                (x1 + w1, y1 + h1),
                                (0, 255, 0),
                                2,
                            )
                            cv2.putText(
                                img,
                                self.get_key_from_value(color),
                                (x + (w // 2), y + (h // 2) - 10),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )

                            cv2.putText(
                                img,
                                objectType,
                                (x + (w // 2), y + (h // 2) + 20),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            box = cv2.minAreaRect(cnt1)
                            box = cv2.boxPoints(box)
                            box = np.array(box, dtype="int")
                            for a, b in box:
                                cv2.putText(
                                    img,
                                    str(a) + "+" + str(b),
                                    (a, b),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2,
                                )
                            if (
                                objectType == "Circles"
                                or objectType == "Rectangle"
                                or objectType == "Square"
                            ):
                                self.data_object = [
                                    x + (w // 2),
                                    y + (h // 2),
                                    self.get_key_from_value(color),
                                    objectType,
                                ]
                            if len(self.list_detect) == 0:
                                self.list_detect.append(self.data_object)
                            else:
                                for (
                                    x_dis,
                                    y_dis,
                                    color_dis,
                                    object_dis,
                                ) in self.list_detect:
                                    if (
                                        x_dis - self.data_object[0] > 50
                                        or y_dis - self.data_object[1] > 50
                                    ):
                                        self.list_detect.append(self.data_object)

                                for x_obj, y_obj, col_obj, obj in self.list_detect:
                                    self.ui.px.setText(f"{round(125+(y_obj*20)/19,2)}")
                                    self.ui.py.setText(
                                        f"{round((x_obj*20)/19-158.5,2)}"
                                    )
                                    self.ui.pz.setText(f"{30}")
                                    self.ui.gripper.setValue(60)
                                    self.list_detect.pop(0)
                                    self.donghocnghich()
        return img

    """
    def Distance(x1, y1, x2, y2):
        x = x2 - x1
        y = y2 - y1
        distance = math.sqrt(x * x + y * y)
        return distance
    """

    def show_wedcam(self, cv_img):
        if self.on and self.mode:
            qt_img = self.convert(cv_img)
            self.ui.frame.setPixmap(qt_img)

    def convert(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_format = QImage(
            rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_format.scaled(300, 300, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_label(self):
        if self.on and self.mode == False and self.move:
            self.theta1 = self.ui.Slider_theta1.value()
            self.ui.linetheta1.setText(f"{self.theta1}")
            self.theta2 = self.ui.Slider_theta2.value()
            self.ui.linetheta2.setText(f"{self.theta2}")
            self.theta3 = self.ui.Slider_theta3.value()
            self.ui.linetheta3.setText(f"{self.theta3}")
            self.theta4 = self.ui.Slider_theta4.value()
            self.ui.linetheta4.setText(f"{self.theta4}")
            self.theta5 = self.ui.Slider_theta5.value()
            self.ui.linetheta5.setText(f"{self.theta5}")
            self.theta6 = self.ui.Slider_theta6.value()
            self.ui.linetheta6.setText(f"{self.theta6}")

    def update_slider(self):
        if self.on and self.mode == False and self.move:
            # slider1
            self.theta1 = float(self.ui.linetheta1.text())
            if (
                self.ui.Slider_theta1.minimum()
                <= self.theta1
                <= self.ui.Slider_theta1.maximum()
            ):
                self.ui.Slider_theta1.setValue(self.theta1)
            else:
                self.ui.linetheta1.setText(f"{self.ui.Slider_theta1.value()}")
            # slider2
            self.theta2 = float(self.ui.linetheta2.text())
            if (
                self.ui.Slider_theta2.minimum()
                <= self.theta2
                <= self.ui.Slider_theta2.maximum()
            ):
                self.ui.Slider_theta2.setValue(self.theta2)
            else:
                self.ui.linetheta2.setText(f"{self.ui.Slider_theta2.value()}")
            # slider3
            self.theta3 = float(self.ui.linetheta3.text())
            if (
                self.ui.Slider_theta3.minimum()
                <= self.theta3
                <= self.ui.Slider_theta3.maximum()
            ):
                self.ui.Slider_theta3.setValue(self.theta3)
            else:
                self.ui.linetheta3.setText(f"{self.ui.Slider_theta3.value()}")
            # slider4
            self.theta4 = float(self.ui.linetheta4.text())
            if (
                self.ui.Slider_theta4.minimum()
                <= self.theta4
                <= self.ui.Slider_theta4.maximum()
            ):
                self.ui.Slider_theta4.setValue(self.theta4)
            else:
                self.ui.linetheta4.setText(f"{self.ui.Slider_theta4.value()}")
            # slider5
            self.theta5 = float(self.ui.linetheta5.text())
            if (
                self.ui.Slider_theta5.minimum()
                <= self.theta5
                <= self.ui.Slider_theta5.maximum()
            ):
                self.ui.Slider_theta5.setValue(self.theta5)
            else:
                self.ui.linetheta5.setText(f"{self.ui.Slider_theta5.value()}")
            # slider6
            self.theta6 = float(self.ui.linetheta6.text())
            if (
                self.ui.Slider_theta6.minimum()
                <= self.theta6
                <= self.ui.Slider_theta6.maximum()
            ):
                self.ui.Slider_theta6.setValue(self.theta6)
            else:
                self.ui.linetheta6.setText(f"{self.ui.Slider_theta6.value()}")

    def tinhdonghoc(self):
        if self.on and self.mode == False:
            theta11 = math.radians(self.theta1)
            theta22 = math.radians(self.theta2)
            theta33 = math.radians(self.theta3)
            theta44 = math.radians(self.theta4)
            theta55 = math.radians(self.theta5)
            theta66 = math.radians(self.theta6)

            joint_angles = [
                theta11,
                theta22,
                theta33,
                theta44,
                theta55,
                theta66,
            ]

            self.T6 = self.robot.fkine(joint_angles)

            self.px = self.T6.t[0]
            self.py = self.T6.t[1]
            self.pz = self.T6.t[2]

            self.nx = self.T6.n[0]
            self.ny = self.T6.n[1]
            self.nz = self.T6.n[2]

            self.ox = self.T6.o[0]
            self.oy = self.T6.o[1]
            self.oz = self.T6.o[2]

            self.ax = self.T6.a[0]
            self.ay = self.T6.a[1]
            self.az = self.T6.a[2]

    def donghocthuan(self):
        if self.on and self.mode == False:
            self.tinhdonghoc()

            self.step1 = self.theta1 - self.theta1_current
            self.step2 = self.theta2 - self.theta2_current
            self.step3 = self.theta3 - self.theta3_current
            self.step4 = self.theta4 - self.theta4_current
            self.theta1_current = self.theta1
            self.theta2_current = self.theta2
            self.theta3_current = self.theta3
            self.theta4_current = self.theta4

            self.ui.px.setText(f"{round(self.px,2)}")
            self.ui.py.setText(f"{round(self.py,2)}")
            self.ui.pz.setText(f"{round(self.pz,2)}")

            self.ui.nx.setText(f"{round(self.nx,2)}")
            self.ui.ny.setText(f"{round(self.ny,2)}")
            self.ui.nz.setText(f"{round(self.nz,2)}")

            self.ui.ox.setText(f"{round(self.ox,2)}")
            self.ui.oy.setText(f"{round(self.oy,2)}")
            self.ui.oz.setText(f"{round(self.oz,2)}")

            self.ui.ax.setText(f"{round(self.ax ,2)}")
            self.ui.ay.setText(f"{round(self.ay ,2)}")
            self.ui.az.setText(f"{round(self.az ,2)}")

            self.add_list()

            self.write_stm()

            """
            data = f"{self.list_write[0][0]}/{self.list_write[0][1]}/{self.list_write[0][2]}/{self.list_write[0][3]}/{self.list_write[0][4]}/{self.list_write[0][5]}."
            self.serial1.write(data.encode("utf-8"))
            self.list_write.pop(0)
            """

    def donghocnghich(self):
        if self.on:
            self.move = False

            q0 = [
                math.radians(0),
                math.radians(114),
                math.radians(-80),
                math.radians(0),
                math.radians(0),
                math.radians(0),
            ]

            desired_position = SE3(
                float(self.ui.px.text()),
                float(self.ui.py.text()),
                float(self.ui.pz.text()),
            )

            mask = np.array([1, 1, 1, 0, 0, 0])

            joint_angles = self.robot.ikine_LM(
                desired_position,
                q0=q0,
                slimit=2000,
                ilimit=6000,
                joint_limits=self.joint_limits,
                tol=0.01,
                mask=mask,
            )

            self.theta1 = round(math.degrees(joint_angles.q[0]), 2)
            self.theta2 = round(math.degrees(joint_angles.q[1]), 2)
            self.theta3 = round(math.degrees(joint_angles.q[2]), 2)
            self.theta4 = round(math.degrees(joint_angles.q[3]), 2)
            self.theta5 = round(math.degrees(joint_angles.q[4]), 2)
            self.theta6 = round(math.degrees(joint_angles.q[5]), 2)

            while joint_angles.success == False:
                joint_angles = self.robot.ikine_LM(
                    desired_position,
                    q0=q0,
                    slimit=1000,
                    ilimit=3000,
                    joint_limits=self.joint_limits,
                    tol=0.01,
                    mask=mask,
                )
                self.theta1 = round(math.degrees(joint_angles.q[0]), 2)
                self.theta2 = round(math.degrees(joint_angles.q[1]), 2)
                self.theta3 = round(math.degrees(joint_angles.q[2]), 2)
                self.theta4 = round(math.degrees(joint_angles.q[3]), 2)
                self.theta5 = round(math.degrees(joint_angles.q[4]), 2)
                self.theta6 = round(math.degrees(joint_angles.q[5]), 2)
            print(
                f"{self.theta1} {self.theta2} {self.theta3} {self.theta4} {self.theta5} {self.theta6}"
            )

            self.step1 = self.theta1 - self.theta1_current
            self.step2 = self.theta2 - self.theta2_current
            self.step3 = self.theta3 - self.theta3_current
            self.step4 = self.theta4 - self.theta4_current
            self.theta1_current = self.theta1
            self.theta2_current = self.theta2
            self.theta3_current = self.theta3
            self.theta4_current = self.theta4

            self.ui.linetheta1.setText(f"{self.theta1}")
            self.ui.linetheta2.setText(f"{self.theta2}")
            self.ui.linetheta3.setText(f"{self.theta3}")
            self.ui.linetheta4.setText(f"{self.theta4}")
            self.ui.linetheta5.setText(f"{self.theta5}")
            self.ui.linetheta6.setText(f"{self.theta6}")

            self.ui.Slider_theta1.setValue(int(self.theta1))
            self.ui.Slider_theta2.setValue(int(self.theta2))
            self.ui.Slider_theta3.setValue(int(self.theta3))
            self.ui.Slider_theta4.setValue(int(self.theta4))
            self.ui.Slider_theta5.setValue(int(self.theta5))
            self.ui.Slider_theta6.setValue(int(self.theta6))

            self.T6 = self.robot.fkine(joint_angles.q)

            self.nx = self.T6.n[0]
            self.ny = self.T6.n[1]
            self.nz = self.T6.n[2]

            self.ox = self.T6.o[0]
            self.oy = self.T6.o[1]
            self.oz = self.T6.o[2]

            self.ax = self.T6.a[0]
            self.ay = self.T6.a[1]
            self.az = self.T6.a[2]

            self.ui.nx.setText(f"{round(self.nx,2)}")
            self.ui.ny.setText(f"{round(self.ny,2)}")
            self.ui.nz.setText(f"{round(self.nz,2)}")

            self.ui.ox.setText(f"{round(self.ox,2)}")
            self.ui.oy.setText(f"{round(self.oy,2)}")
            self.ui.oz.setText(f"{round(self.oz,2)}")

            self.ui.ax.setText(f"{round(self.ax ,2)}")
            self.ui.ay.setText(f"{round(self.ay ,2)}")
            self.ui.az.setText(f"{round(self.az ,2)}")

            self.add_list()

            self.write_stm()
            """
            data = f"{self.list_write[0][0]}/{self.list_write[0][1]}/{self.list_write[0][2]}/{self.list_write[0][3]}/{self.list_write[0][4]}/{self.list_write[0][5]}."
            self.serial1.write(data.encode("utf-8"))
            self.list_write.pop(0)
            """
        self.move = True

    def add_list(self):
        if self.on:
            self.step1 = int(((self.step1 * (200 / 360)) * 97) / 16)
            self.step2 = int(((self.step2 * (200 / 360)) * 132) / 16)
            self.step3 = int(((self.step3 * (200 / 360)) * 126) / 16)
            self.step4 = int(self.step4 * (200 / 360))
            self.gripper = self.ui.gripper.value()

            self.list_write.append(
                [
                    self.step1,
                    self.step2,
                    -self.step3,
                    self.step4,
                    -int(self.theta5),
                    int(self.theta6),
                    self.gripper,
                ]
            )
            print(self.list_write)

    def write_stm(self):
        text = str(self.serial1.read(20))
        print(text)
        if "Xong" in text:
            self.write = True
        if self.write:
            if len(self.list_write) > 0:
                data = f"{self.list_write[0][0]}/{self.list_write[0][1]}/{self.list_write[0][2]}/{self.list_write[0][3]}/{self.list_write[0][4]}/{self.list_write[0][5]}/{self.list_write[0][6]}."
                try:
                    self.serial1.write(data.encode("utf-8"))
                except:
                    print("Khong gui dc")
                self.list_write.pop(0)
                self.write = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    PBL4 = Ui_PBL4()
    PBL4.ui.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
    PBL4.ui.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
    PBL4.ui.show()
    sys.exit(app.exec())
