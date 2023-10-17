"""
"""
import airsim
import numpy as np
import math
from scipy import ndimage

class Agent(object):
    def __init__(self, useAPI=True):
        self.client = airsim.CarClient()
        self.controller = airsim.CarControls()

        self._setConnection(useAPI)
        self._setSegmentation()

    def _setConnection(self, useAPI):
        self.client.confirmConnection()
        self.client.enableApiControl(useAPI)

    def _setSegmentation(self):
        # self.client.simSetSegmentationObjectID("SM_SkySphere", 195, True)
        # self.client.simSetSegmentationObjectID("Terrain", 250, True)
        # self.client.simSetSegmentationObjectID("Prop", 198, True)
        # self.client.simSetSegmentationObjectID("Road[\w]*", 171, True)
        # self.client.simSetSegmentationObjectID("Marking[\w]*", 238, True)
        # self.client.simSetSegmentationObjectID("Cube[\w]*", 164, True)

        self.client.simSetSegmentationObjectID("SM_SkySphere", 195, True)
        self.client.simSetSegmentationObjectID("Landscape[\w]*", 250, True)
        self.client.simSetSegmentationObjectID("SM_Path[\w]*", 171, True)
        self.client.simSetSegmentationObjectID('barrier[\w]*', 198, True)
        self.client.simSetSegmentationObjectID('Fir_01_Plant[\w]*', 151, True)
        self.client.simSetSegmentationObjectID('Fir_03_Medium[\w]*', 151, True)
        self.client.simSetSegmentationObjectID('Poplar_04_Small[\w]*', 151, True)
        self.client.simSetSegmentationObjectID('Poplar_05_Small[\w]*', 151, True)
        self.client.simSetSegmentationObjectID('SM_Big_Rock_01[\w]*', 141, True)
        self.client.simSetSegmentationObjectID('SM_Big_Rock_02[\w]*', 141, True)
        self.client.simSetSegmentationObjectID('SM_clover_01[\w]*', 131, True)
        self.client.simSetSegmentationObjectID('SM_clover_01_flower[\w]*', 131, True)
        self.client.simSetSegmentationObjectID('SM_grass_bush_01[\w]*', 131, True)
        self.client.simSetSegmentationObjectID('SM_grass_bush_simple_04[\w]*', 131, True)



    def getAction(self):
        return np.array([
            self.controller.steering,
            self.controller.throttle,
            self.controller.brake
        ])

    def extractState(self):
        rawState = self.client.getCarState('Car1')

        angular_acc_x = rawState.kinematics_estimated.angular_acceleration.x_val
        angular_acc_y = rawState.kinematics_estimated.angular_acceleration.x_val
        angular_acc_z = rawState.kinematics_estimated.angular_acceleration.x_val

        angular_vel_x = rawState.kinematics_estimated.angular_velocity.x_val
        angular_vel_y = rawState.kinematics_estimated.angular_velocity.y_val
        angular_vel_z = rawState.kinematics_estimated.angular_velocity.z_val

        linear_acc_x = rawState.kinematics_estimated.linear_acceleration.x_val
        linear_acc_y = rawState.kinematics_estimated.linear_acceleration.y_val
        linear_acc_z = rawState.kinematics_estimated.linear_acceleration.z_val

        linear_vel_x = rawState.kinematics_estimated.linear_velocity.x_val
        linear_vel_y = rawState.kinematics_estimated.linear_velocity.y_val
        linear_vel_z = rawState.kinematics_estimated.linear_velocity.z_val

        speed = rawState.speed
        rpm = rawState.rpm
        gear = rawState.gear

        return np.array([
            angular_acc_x, angular_acc_y, angular_acc_z, angular_vel_x, angular_vel_y, angular_vel_z,
            linear_acc_x, linear_acc_y, linear_acc_z, linear_vel_x, linear_vel_y, linear_vel_z,
            speed, rpm, gear
        ])

    def getCarGPS(self):
        rawData = self.client.simGetVehiclePose('Car1').position
        gpsData = np.array(
            [rawData.x_val, rawData.y_val, rawData.z_val], dtype=np.float32
        ).round(3)
        return gpsData

    def getDistance(self):
        pointCloud = self.client.getLidarData('LidarSensor2', 'Car1').point_cloud
        carGPS = self.getCarGPS()
        try:
            pointCloud = np.reshape(pointCloud, (int(len(pointCloud)/3), 3)).round(3)
            distance = pointCloud - carGPS
            distance = np.square(distance)
            distance = np.sum(distance, axis=1)
            distance = np.sqrt(distance).round(3)
            distance = np.sort(distance)
            meanDistance = int(np.mean(distance[:10]))
            return meanDistance
        except ValueError as e:
            return 10000

    def getCamImage(self, camCategories):
        numCam = len(camCategories)
        imgArray = np.zeros((numCam, 144, 256, 3), dtype=np.uint8)

        for idx, camName, imgType in zip(list(range(numCam)), camCategories.keys(), camCategories.values()):
            res = self.client.simGetImages([
                airsim.ImageRequest(camName, imgType, False, False)
            ])[0]
            img1d = np.fromstring(res.image_data_uint8, dtype=np.uint8)
            img = np.reshape(img1d, (144, 256, 3))
            imgArray[idx] = img
        return imgArray

    def getLidarVis(self):
        mapping = np.zeros((200, 200), dtype=np.uint8)

        carPosVec = self.getCarGPS()

        camPos = self.client.simGetCameraInfo('F_myCamSeg').pose
        camPosVec = np.array([camPos.position.x_val, camPos.position.y_val, camPos.position.z_val], dtype=np.float32)

        pointCloud = self.client.getLidarData('LidarSensor1', 'Car1')
        pointCloud = np.array(pointCloud.point_cloud, dtype=np.float32)

        try:
            lidarData = np.reshape(pointCloud, (int(pointCloud.shape[0]/3), 3))
            lidarData = lidarData - carPosVec
            lidarData = lidarData + 100.0
            lidarData = np.clip(lidarData.astype(np.uint8), 0, 199)

            for i in range(lidarData.shape[0]):
                mapping[lidarData[i, 0], lidarData[i, 1]] = 255
        
        except ValueError as e:
            print(e)

        delta = carPosVec - camPos
        theta = math.atan2(delta[1], delta[0]) * 180 / np.pi
        visLidar = np.fliplr(ndimage.rotate(mapping, -theta))
        return visLidar