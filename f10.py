import pybullet as p
import time
import math
import numpy as np

class f10RaceCarEnv():
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
        p.setTimeStep(1. / 120.) #mozna neni nutny
        p.setRealTimeSimulation(0)

        # Input and output dimensions defined in the environment
        self.obs_dim = 4
        self.act_dim = 4


        #track = p.loadURDF("plane.urdf")
        self.track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
        self.car = p.loadURDF("f10_racecar/racecar_differential.urdf", [0, 0, .3])

        # Input and output dimensions defined in the environment
        for wheel in range(p.getNumJoints(self.car)):
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

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

        self.Velocity = 1

        self.stepCtr = 0
        self.hokuyo_joint = 4

        # nastaveni paprsku
        self.replaceLines = True
        self.numRays = 10
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]
        self.rayLen = 2.5
        self.rayStartLen = 0.25
        for i in range(self.numRays):

            self.rayFrom.append(
                [self.rayStartLen * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays),
                 self.rayStartLen * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays), 0])
            self.rayTo.append([self.rayLen * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays),
                          self.rayLen * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self.numRays), 0])
            if (self.replaceLines):
                self.rayIds.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, parentObjectUniqueId=self.car,
                                                 parentLinkIndex=self.hokuyo_joint))
            else:
                self.rayIds.append(-1)

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
        #todo: write better getObs function
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.car)
        torsoVel, torsoAngVel = p.getBaseVelocity(self.car)

        obs = p.getJointStates(self.car, range(19))
        joint_angles = []
        i=0
        for o in obs:
            if i == 0 or i == 2:
                joint_angles.append(o[0])
            i +=1
        return torso_pos, joint_angles

    def step(self, steeringAngle, velocity):
        angles = self.norm_to_rads(steeringAngle)

        numThreads = 0
        results = p.rayTestBatch(self.rayFrom, self.rayTo, numThreads, parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)
        for i in range(self.numRays):
            hitObjectUid = results[i][0]
            hitFraction = results[i][2]
            hitPosition = results[i][3]

            if (hitFraction == 1.):
                p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, replaceItemUniqueId=self.rayIds[i],
                                   parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)

            else:
                localHitTo = [self.rayFrom[i][0] + hitFraction * (self.rayTo[i][0] - self.rayFrom[i][0]),
                              self.rayFrom[i][1] + hitFraction * (self.rayTo[i][1] - self.rayFrom[i][1]),
                              self.rayFrom[i][2] + hitFraction * (self.rayTo[i][2] - self.rayFrom[i][2])]
                p.addUserDebugLine(self.rayFrom[i], localHitTo, self.rayHitColor, replaceItemUniqueId=self.rayIds[i],
                                   parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)


        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=velocity[0], force=50.0)
        p.setJointMotorControl2(self.car, self.wheels[1], p.VELOCITY_CONTROL, targetVelocity=velocity[1], force=50.0)

        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=angles[0])
        p.setJointMotorControl2(self.car, self.steering[1], p.POSITION_CONTROL, targetPosition=angles[1])

        # Step the simulation.
        p.stepSimulation()
        if self.animate: time.sleep(0.004)

        #todo: get new observations
        torsoVelocity, newAngle = self.getObs()
        x, y, z = torsoVelocity

        #todo: write better reward function
        #velocityRew = np.minimum(y+x, self.Velocity) / self.Velocity
        velocityRew = np.sin(x*y)
        r_pos = (velocityRew * 1.0) / self.max_steps * 100

        # Scale joint angles and make the policy observation
        scaled_joint_angles = self.rads_to_norm(newAngle)
        env_obs = np.concatenate((scaled_joint_angles, velocity)).astype(np.float32)
        #env_obs = [scaled_joint_angles, velocity]

        self.stepCtr += 1

        # This condition terminates the episode
        done = self.stepCtr > self.max_steps
        return env_obs, r_pos, done

    def reset(self):
        self.stepCtr = 0  # Counts the amount of steps done in the current episode

        # Reset the robot to initial position and orientation and null the motors
        #joint_init_pos_list = self.norm_to_rads([0] * 19)
        #[p.resetJointState(self.car, i, joint_init_pos_list[i], 0) for i in range(19)]
        p.resetBasePositionAndOrientation(self.car, [0, 0, .3], [0, 0, 0, 1])

        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.car, self.wheels[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(self.car, self.steering[0], p.POSITION_CONTROL, targetPosition=0)

        # Step a few times so stuff settles down
        for i in range(10):
            p.stepSimulation()
        if self.animate: time.sleep(0.04)

        # Return initial obs
        obs, _, _ = self.step([0,0],[10, 10])
        return obs


    def demo(self):
        while (True):
            a = -0.75
            for i in range (self.max_steps):
                if i < self.max_steps/5.5:
                    if a < -0.45:
                        a = a + 0.001
                else:
                    a = 0.5
                self.step([-a,-a],[10,10])
            self.reset()

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = f10RaceCarEnv(animate=True)
    env.demo()
