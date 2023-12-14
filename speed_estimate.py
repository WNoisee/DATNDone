import math
import numpy as np

class SpeedEstimator:
    def __init__(self, posList, fps):
        self.x = posList[0]
        self.y = posList[1]
        self.fps = fps

    def estimateSpeed(self):
        # Distance / Time -> Speed
        d_pixels = math.sqrt(self.x + self.y)
        ppm = 8.8
        d_meters = int(d_pixels * ppm)
        speed = d_meters / self.fps * 3.6
        speedInKM = np.average(speed)
        return int(speedInKM)
