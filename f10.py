import pybullet as p
import time
import math
import numpy as np
#from Semestralka.track import getdataset
import os


# our main env class

class f10RaceCarEnv():

    #initialization

    def __init__(self, animate=False, max_steps=1000):
        self.animate = animate
        self.max_steps = max_steps

        # Initialize simulation
        if (animate):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        # Set simulation world params
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / 120.)

        # Input and output dimensions defined in the environment
        self.obs_dim = 12
        self.act_dim = 2

        # sum rewards
        self.lastrew = 0

        # parametrs for the track
        self.ctr = 0
        self.iteration = 100
        self.first = True
        self.X = []
        self.Y = []
        self.index = 0
        self.amount = 0
        self.reset_index = 0
        env_width = 100
        env_height = 100

        # generating track
        heightfieldData = [0] * (env_height * env_width)
        # if we would like to generate new track every time
        '''
        datasetx, datasety, x1,y1,cx,cy = getdataset()
        datasetx = np.int64(datasetx)
        datasety = np.int64(datasety)
        x1 = np.int64(x1)
        y1 = np.int64(y1)'''
        # for training we used pregenerated tracks
        if self.animate:
            self.tracking = 5
        else:
            self.tracking = 1
        exx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outx0.npy')
        exy = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outy0.npy')
        inx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inx0.npy')
        iny = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'iny0.npy')
        checkx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpointsx0.npy')
        checky = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpointsy0.npy')

        datasetx = np.load(exx)
        datasety = np.load(exy)
        x1 = np.load(inx)
        y1 = np.load(iny)
        cx = np.load(checkx)
        cy = np.load(checky)

        first = True
        for i in range(len(datasetx)):
            x = (datasetx[i])
            y = (datasety[i])
            xnew = x1[i]
            ynew = y1[i]

            if not first and xold == x and yold == y:
                continue
            else:
                heightfieldData.pop(env_height * x + y)
                heightfieldData.insert(env_height * x + y, 1)
                heightfieldData.pop(env_height * xnew + ynew)
                heightfieldData.insert(env_height * xnew + ynew, 1)

                xold = x
                yold = y
                first = False

        self.X = cy
        self.Y = cx
        self.amount = len(cx)

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                              meshScale=[1, 1,
                                                         1],
                                              heightfieldTextureScaling=(env_width - 1) / 2,
                                              heightfieldData=heightfieldData,
                                              numHeightfieldRows=env_height,
                                              numHeightfieldColumns=env_width,
                                              physicsClientId=self.client_ID)

        mass = 0
        terrain = p.createMultiBody(mass, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [49.5, 49.5, 0], [0, 0, 0, 1])

        # generating car in the right angle

        if (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 2 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > -2):
            angle = 100
        elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 2 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > -2):
            angle = 0
        elif (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0 and (
                (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 2 and (
                self.X[self.reset_index + 5] - self.X[self.reset_index]) > -2):
            angle = -1
        elif (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0 and (
                (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 2 and (
                self.X[self.reset_index + 5] - self.X[self.reset_index]) > -2):
            angle = 1
        elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0:
            angle = 0.5
        elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0:
            angle = -0.5
        elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0:
            angle = -2
        elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0:
            angle = 2

        else:
            angle = 1

        self.car = p.loadURDF("f10_racecar/racecar_differential.urdf", [self.X[0], self.Y[0], .3], [0, 0, angle, 1])
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=270,
                                     cameraTargetPosition=[self.X[0], self.Y[0], 0])

        # Input and output dimensions defined in the environment
        for wheel in range(p.getNumJoints(self.car)):
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            p.changeDynamics(self.car, wheel, mass=1, lateralFriction=1.0)

        self.wheels = [8, 15]

        # spojuje klouby přes ramena dohromady a tím vytváří celek auta
        self.c = p.createConstraint(self.car, 9, self.car, 11, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=1, maxForce=10000)

        self.c = p.createConstraint(self.car, 10, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, maxForce=10000)

        self.c = p.createConstraint(self.car, 9, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, maxForce=10000)

        self.c = p.createConstraint(self.car, 16, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=1, maxForce=10000)

        self.c = p.createConstraint(self.car, 16, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, maxForce=10000)

        self.c = p.createConstraint(self.car, 17, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, maxForce=10000)

        self.c = p.createConstraint(self.car, 1, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        self.c = p.createConstraint(self.car, 3, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        p.changeConstraint(self.c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

        self.steering = [0, 2]

        # Limits of our joints. When using the * (multiply) operation on a list, it repeats the list that many times
        self.joints_rads_low = -1.57
        self.joints_rads_high = 4.71
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        # self.Velocity = 1

        self.stepCtr = 0
        self.hokuyo_joint = 4

        # Lidar rays initialisation
        self.replaceLines = True
        self.numRays = 10
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]
        self.rayLen = 7
        self.rayStartLen = 0.25
        for i in range(self.numRays):

            self.rayFrom.append(
                [self.rayStartLen * math.sin(
                    -0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays),
                 self.rayStartLen * math.cos(
                     -0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays), 0])
            self.rayTo.append(
                [self.rayLen * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays),
                 self.rayLen * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays), 0])
            if (self.replaceLines):
                self.rayIds.append(
                    p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, parentObjectUniqueId=self.car,
                                       parentLinkIndex=self.hokuyo_joint))
            else:
                self.rayIds.append(-1)

    # two functions for converting either from normalised data to radians or radians to normalised data

    def rads_to_norm(self, joints):
        sjoints = ((joints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        return (action * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    #function for getting the state of the car in the specific time

    def getObs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.car)
        obs = p.getJointStates(self.car, range(19))
        joint_angles = obs[0][0]

        return torso_pos, joint_angles

    # function for one step in simulation

    def step(self, steeringAngle, velocity):
        if self.animate and self.stepCtr == 0:
            self.start = time.time()

        angles = self.norm_to_rads(steeringAngle)
        hitArray = []
        numThreads = 0

        velocity = velocity * 100

        results = p.rayTestBatch(self.rayFrom, self.rayTo, numThreads, parentObjectUniqueId=self.car,
                                 parentLinkIndex=self.hokuyo_joint)
        for i in range(self.numRays):
            hitFraction = results[i][2]
            hitArray.append(hitFraction)

            if (hitFraction == 1.):
                p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor,
                                   replaceItemUniqueId=self.rayIds[i],
                                   parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)

            else:
                localHitTo = [self.rayFrom[i][0] + hitFraction * (self.rayTo[i][0] - self.rayFrom[i][0]),
                              self.rayFrom[i][1] + hitFraction * (self.rayTo[i][1] - self.rayFrom[i][1]),
                              self.rayFrom[i][2] + hitFraction * (self.rayTo[i][2] - self.rayFrom[i][2])]
                p.addUserDebugLine(self.rayFrom[i], localHitTo, self.rayHitColor, replaceItemUniqueId=self.rayIds[i],
                                   parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)

        # changing velocity and angle of the wheels
        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=velocity, force=50.0)
        p.setJointMotorControl2(self.car, self.wheels[1], p.VELOCITY_CONTROL, targetVelocity=velocity, force=50.0)

        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=angles)
        p.setJointMotorControl2(self.car, self.steering[1], p.POSITION_CONTROL, targetPosition=angles)

        # Step the simulation.
        p.stepSimulation()
        if self.animate: time.sleep(0.004)

        torsoPosition, newAngle = self.getObs()
        x, y, z = torsoPosition

        # reward function
        if self.index < (self.amount - 5):
            velocityRew = ((self.X[self.index + 5] - self.X[self.index]) ** 2 + (
                        self.Y[self.index + 5] - self.Y[self.index]) ** 2) - (
                                      (self.X[self.index + 5] - x) ** 2 + (self.Y[self.index + 5] - y) ** 2)
            dist = (self.X[self.index + 5] - self.X[self.index]) ** 2 + (
                        self.Y[self.index + 5] - self.Y[self.index]) ** 2
            currdist = ((self.X[self.index + 5] - x) ** 2 + (self.Y[self.index + 5] - y) ** 2)
            # normalization
            r_pos = self.lastrew + velocityRew / (dist + 0.0001)
            if currdist < 0.7:
                self.lastrew = r_pos
                if self.index < (self.amount - 5):
                    self.index += 5
                else:
                    self.index = 0
        else:
            velocityRew = ((self.X[0] - self.X[self.index]) ** 2 + (self.Y[0] - self.Y[self.index]) ** 2) - (
                        (self.X[0] - x) ** 2 + (self.Y[0] - y) ** 2)
            dist = ((self.X[0] - self.X[self.index]) ** 2 + (self.Y[0] - self.Y[self.index]) ** 2)
            currdist = ((self.X[0] - x) ** 2 + (self.Y[0] - y) ** 2)
            # normalization
            r_pos = self.lastrew + velocityRew / (dist + 0.0001)
            if currdist < 0.7:
                self.lastrew = r_pos
                if self.index < (self.amount - 5):
                    self.index += 5
                else:
                    self.index = 0

        # Scale joint angles and make the policy observation
        scaled_joint_angles = self.rads_to_norm(newAngle)
        velocity = velocity/100
        velocity_cliped = np.clip(velocity, 0, 1)
        env_obs = [scaled_joint_angles, velocity_cliped]
        for i in range(10):
            env_obs.append(hitArray[i])

        self.stepCtr += 1

        # This condition terminates the episode
        done = self.stepCtr > self.max_steps

        return env_obs, r_pos, done

    # function for generating track

    def generate_track(self):
        env_width = 100
        env_height = 100

        heightfieldData = [0] * (env_height * env_width)
        exx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outx' + str(self.tracking) + '.npy')
        exy = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outy' + str(self.tracking) + '.npy')
        inx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inx' + str(self.tracking) + '.npy')
        iny = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'iny' + str(self.tracking) + '.npy')
        checkx = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpointsx' + str(self.tracking) + '.npy')
        checky = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpointsy' + str(self.tracking) + '.npy')

        if self.tracking < 4:
            self.tracking += 1
        elif self.tracking == 5:
            self.tracking = 5
        else:
            self.tracking = 0

        datasetx = np.load(exx)
        datasety = np.load(exy)
        x1 = np.load(inx)
        y1 = np.load(iny)
        cx = np.load(checkx)
        cy = np.load(checky)
        '''
        datasetx, datasety, x1, y1, cx, cy = getdataset()
        datasetx = np.int64(datasetx)
        datasety = np.int64(datasety)
        x1 = np.int64(x1)
        y1 = np.int64(y1)
        '''
        first = True
        for i in range(len(datasetx)):
            x = (datasetx[i])
            y = (datasety[i])
            xnew = x1[i]
            ynew = y1[i]

            if not first and xold == x and yold == y:
                continue
            else:
                heightfieldData.pop(env_height * x + y)
                heightfieldData.insert(env_height * x + y, 1)
                heightfieldData.pop(env_height * xnew + ynew)
                heightfieldData.insert(env_height * xnew + ynew, 1)

                xold = x
                yold = y
                first = False

        self.X = cy
        self.Y = cx
        self.amount = len(cx)

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                              meshScale=[1, 1,
                                                         1],
                                              heightfieldTextureScaling=(env_width - 1) / 2,
                                              heightfieldData=heightfieldData,
                                              numHeightfieldRows=env_height,
                                              numHeightfieldColumns=env_width,
                                              physicsClientId=self.client_ID)

        mass = 0
        terrain = p.createMultiBody(mass, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [49.5, 49.5, 0], [0, 0, 0, 1])

    #reset function

    def reset(self):

        # measuring time

        if not self.first and self.animate:
            stop = time.time()
            elapsed = stop - self.start
            print("TIME=", elapsed)

        self.stepCtr = 0  # Counts the amount of steps done in the current episode

        self.ctr += 1

        if not self.animate and self.ctr % self.iteration == 0:
            try:
                self.generate_track()
                self.reset_index = 0
                self.index = 0
            except:
                print("Couldn't generate new track")

        self.first = False
        # Reset the robot to initial position and orientation and null the motors

        self.lastrew = 0  # resets rewards

        if self.reset_index < (self.amount - 5):
            if (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                    (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 2 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > -2):
                angle = 100
            elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                    (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 2 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > -2):
                angle = 0
            elif (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0 and (
                    (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 2 and (
                    self.X[self.reset_index + 5] - self.X[self.reset_index]) > -2):
                angle = -1
            elif (self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0 and (
                    (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 2 and (
                    self.X[self.reset_index + 5] - self.X[self.reset_index]) > -2):
                angle = 1
            elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0:
                angle = 0.5
            elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) > 0 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0:
                angle = -0.5
            elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) < 0:
                angle = -2
            elif (self.X[self.reset_index + 5] - self.X[self.reset_index]) < 0 and (
                    self.Y[self.reset_index + 5] - self.Y[self.reset_index]) > 0:
                angle = 2

            else:
                angle = 1
        else:
            if (self.X[0] - self.X[self.reset_index]) < 0 and (
                    (self.Y[0] - self.Y[self.reset_index]) < 2 and (self.Y[0] - self.Y[self.reset_index]) > -2):
                angle = 100
            elif (self.X[0] - self.X[self.reset_index]) > 0 and (
                    (self.Y[0] - self.Y[self.reset_index]) < 2 and (self.Y[0] - self.Y[self.reset_index]) > -2):
                angle = 0
            elif (self.Y[0] - self.Y[self.reset_index]) < 0 and (
                    (self.X[0] - self.X[self.reset_index]) < 2 and (self.X[0] - self.X[self.reset_index]) > -2):
                angle = -1
            elif (self.Y[0] - self.Y[self.reset_index]) > 0 and (
                    (self.X[0] - self.X[self.reset_index]) < 2 and (self.X[0] - self.X[self.reset_index]) > -2):
                angle = 1
            elif (self.X[0] - self.X[self.reset_index]) > 0 and (self.Y[0] - self.Y[self.reset_index]) > 0:
                angle = 0.5
            elif (self.X[0] - self.X[self.reset_index]) > 0 and (self.Y[0] - self.Y[self.reset_index]) < 0:
                angle = -0.5
            elif (self.X[0] - self.X[self.reset_index]) < 0 and (self.Y[0] - self.Y[self.reset_index]) < 0:
                angle = -2
            elif (self.X[0] - self.X[self.reset_index]) < 0 and (self.Y[0] - self.Y[self.reset_index]) > 0:
                angle = 2

            else:
                angle = 1

        p.resetBasePositionAndOrientation(self.car, [self.X[self.reset_index], self.Y[self.reset_index], .3],
                                          [0, 0, angle, 1])

        self.index = self.reset_index

        self.reset_index += 5

        if self.reset_index >= self.amount:
            self.reset_index = 0

        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=0)

        # Step a few times so stuff settles down
        for i in range(10):
            p.stepSimulation()
        if self.animate: time.sleep(0.04)

        # Return initial obs
        obs, _, _ = self.step(0, 10)
        return obs

    #demo function

    def demo(self):
        while (True):
            a = -0.75
            for i in range(self.max_steps):
                if i < self.max_steps / 5.5:
                    if a < -0.45:
                        a = a + 0.001
                else:
                    a = 0.5
                self.step(-a, 10)
            self.reset()

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)


if __name__ == "__main__":
    env = f10RaceCarEnv(animate=True)
    env.demo()
