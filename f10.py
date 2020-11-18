import pybullet as p
import time
import math
import numpy as np

class f10RaceCarEnv():
    def __init__(self, animate=False, max_steps=100):
        self.animate = animate
        self.max_steps = max_steps

        # Initialize simulation
        if (animate):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        # Set simulation world params
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setTimeStep(1. / 120.) #mozna neni nutny
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)

        #track = p.loadURDF("plane.urdf")
        self.track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1, physicsClientId=self.client_ID)
        # otherCar = p.loadURDF("f10_racecar/racecar_differential.urdf", [0,1,.3])
        self.car = p.loadURDF("f10_racecar/racecar_differential.urdf", [0, 0, .3], physicsClientId=self.client_ID)

        # Input and output dimensions defined in the environment
        #todo

        for wheel in range(p.getNumJoints(self.car)):
            print("joint[", wheel, "]=", p.getJointInfo(self.car, wheel))
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            p.getJointInfo(self.car, wheel)

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

        self.lastControlTime = time.time()

        self.sim_steps_per_iter = 24

        # Limits of our joints. When using the * (multiply) operation on a list, it repeats the list that many times
        self.joints_rads_low = -1.57
        self.joints_rads_high = 4.71
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        #self.targetVelocity = 10

        self.stepCtr = 0

    def getCarYaw(self):
        carPos, carOrn = p.getBasePositionAndOrientation(self.car)
        carEuler = p.getEulerFromQuaternion(carOrn)
        carYaw = carEuler[2] * 360 / (2. * math.pi) - 90
        return carYaw

    def rads_to_norm(self, joints):
        '''
        :param joints: list or array of joint angles in radians
        :return: array of joint angles normalized to [-1,1]
        '''
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        '''
        :param action: list or array of normalized joint target angles (from your control policy)
        :return: array of target joint angles in radians (to be published to simulator)
        '''
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def getObs(self):
        torsoVel, torsoAngVel = p.getBaseVelocity(self.car, physicsClientId=self.client_ID)
        return torsoVel

    def step(self, targetVelocity, steeringAngle):
        angles = self.norm_to_rads(steeringAngle)

       #for wheel in self.wheels:
        #    print("wheel", wheel, self.car, targetVelocity)
        #    p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=targetVelocity, force=0)
        # otáčí koly - nastavení úhlu zahnutí
        #for steer in self.steering:
        #   p.setJointMotorControl2(self.car, steer, p.POSITION_CONTROL, targetPosition=-angles)

        # Step the simulation.
        for i in range(self.sim_steps_per_iter):
            p.stepSimulation(physicsClientId=self.client_ID)
            if self.animate: time.sleep(0.004)

        #todo: get new observations
        torsoVelocity = self.getObs()
        x, y, z = torsoVelocity

        velocityRew = np.minimum(y, self.targetVelocity) / self.targetVelocity

        # Scale joint angles and make the policy observation
        scaled_joint_angles = self.rads_to_norm(angles)
        env_obs = np.concatenate((scaled_joint_angles).astype(np.float32))

        self.stepCtr += 1

        # This condition terminates the episode
        done = self.stepCtr > self.max_steps

        return env_obs, velocityRew, done

    def reset(self):
        self.stepCtr = 0


    def demo(self):
        p.resetSimulation(physicsClientId=self.client_ID)
        while (True):
            for i in range (self.max_steps):
                self.step(0, -0.5)

if __name__ == "__main__":
    env = f10RaceCarEnv(animate=True)
    env.demo()
