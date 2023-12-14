from ultralytics import YOLO
import numpy as np

bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

model = YOLO(r"C:\Users\doant\PycharmProjects\pythonProject7\best.pt")
metric = model.val(data =r"C:\Users\doant\PycharmProjects\pythonProject7\coco128.yaml", iou=0.5)
confus = metric.confusion_matrix.matrix
print(confus)
TP_motorcycle = confus[0][0]
FP_motorcycle = confus[0][1] + confus[0][2] + confus[0][3] + confus[0][4]
FN_motorcycle = confus[1][0] + confus[2][0] + confus[3][0] + confus[4][0]
TN_motorcycle = 14817 - (TP_motorcycle + FP_motorcycle + FN_motorcycle)

confusion_matrix_Motorcycle = np.array([TP_motorcycle, FP_motorcycle, FN_motorcycle, TN_motorcycle])

print("TP_motorcycle: ", TP_motorcycle)
print("FP_motorcycle: ", FP_motorcycle)
print("FN_motorcycle: ", FN_motorcycle)
print("TN_motorcycle: ", TN_motorcycle)

TP_Car = confus[1][1]
FP_Car = confus[1][0] + confus[1][2] + confus[1][3] + confus[1][4]
FN_Car = confus[0][1] + confus[2][1] + confus[3][1] + confus[4][1]
TN_Car = 4049 - (TP_Car + FP_Car + FN_Car)

confusion_matrix_Car = np.array([TP_Car, FP_Car,FN_Car,TN_Car])

print("TP_Car: ", TP_Car)
print("FP_Car: ", FP_Car)
print("FN_Car: ", FN_Car)
print("TN_Car: ", TN_Car)

TP_Truck = confus[3][3]
FP_Truck = confus[3][0] + confus[3][1] + confus[3][2] + confus[3][4]
FN_Truck = confus[0][3] + confus[1][3] + confus[2][3] + confus[4][3]
TN_Truck = 494 - (TP_Truck + FP_Truck + FN_Truck)

print("TP_Truck", TP_Truck)
print("FP_Truck", FP_Truck)
print("FN_Truck", FN_Truck)
print("TN_Truck", TN_Truck)

confusion_matrix_Truck = np.array([TP_Truck, FP_Truck, FN_Truck, TN_Truck])

import matplotlib.pyplot as plt

plt.subplot(1,3,1)
plt.bar(["TP","FP","FN","TN"], confusion_matrix_Motorcycle, color=bar_colors)
plt.title("Motorcycle")


plt.subplot(1,3,2)
plt.bar(["TP","FP","FN","TN"], confusion_matrix_Car, color=bar_colors)
plt.title("Car")


plt.subplot(1,3,3)
plt.bar(["TP","FP","FN","TN"], confusion_matrix_Truck, color=bar_colors)
plt.title("Truck")

plt.show()
