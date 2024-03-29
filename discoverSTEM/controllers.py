import numpy as np
import numpy.linalg as la

   


class moveSphereTo:
    def __init__(self,target_x,target_y,Kp=3,Kd=7,tolerance=0.1):
        self.target = np.array([target_x,target_y])
        self.Kp = Kp
        self.Kd = Kd
        self.Done = False
        self.posErr = np.zeros(2)
        self.velErr = np.zeros(2)
        self.tolerance = tolerance
        
    def update(self,measurement):
        pos,vel = measurement
        self.posErr = pos - self.target
        self.velErr = vel
        if (la.norm(self.posErr) < self.tolerance) and (la.norm(self.velErr) < self.tolerance):
            self.Done = True
        
    def value(self):
        return -self.Kp * self.posErr - self.Kd * self.velErr

class controllerSequence:
    def __init__(self,controllers):
        self.NumControllers = len(controllers)
        self.controllers = controllers
        self.index = 0
        self.Done = False
        
    def update(self,measurement):
        if (self.controllers[self.index].Done) and (self.index < self.NumControllers -1):
            self.index += 1
        self.controllers[self.index].update(measurement)

        if (self.index == self.NumControllers - 1) and self.controllers[self.index].Done:
            self.Done = True
        

    def value(self):
        return self.controllers[self.index].value()
        
        
class turnCar:
    def __init__(self,target_angle,Kp=3,Kd=7,tolerance=0.001):
        self.target = target_angle
        self.Kp = Kp
        self.Kd = Kd
        self.tol = tolerance
        self.Done = False
        
    def update(self,measurement):
        x,y,theta,v,omega = measurement
        self.posError = ((theta - self.target + np.pi) % (2*np.pi)) - np.pi
        self.velError = omega

        if np.abs(self.posError) < self.tol and np.abs(self.velError) < self.tol:
            self.Done = True
        
    def value(self):
        domega = -self.Kp * self.posError - self.Kd * self.velError
        return np.array([0.,domega]) 

class carForward:
    def __init__(self,distance,Kp=3,Kd=7,tolerance=.001):
        # Distance Must be positive
        self.startPosition = None
        self.Kp = Kp
        self.Kd = Kd
        self.tol = tolerance
        self.Done = False
        self.goalPosition = None
        self.distance = np.abs(distance)
    def update(self,measurement):
        x,y,theta,v,omega = measurement
        curPos = np.array([x,y])
        if self.goalPosition is None:
            self.goalPosition = curPos + self.distance * np.array([np.cos(theta),np.sin(theta)])
            self.startPosition = np.copy(curPos)
            self.projector = (self.goalPosition - self.startPosition) / self.distance
        
        self.d_err = np.dot(curPos - self.startPosition,self.projector) -self.distance
        self.v_err = np.dot(np.array([v * np.cos(theta),v*np.sin(theta)]),self.projector)


        if (np.abs(self.d_err) < self.tol) and (np.abs(self.v_err) < self.tol):
            self.Done = True

    def value(self):
        return np.array([-self.Kp * self.d_err-self.Kd * self.v_err,0.]) 
        

    
class carJoint:
    def __init__(self,distance,Kp=3,Kd=7,Kp_ang=.01,Kd_ang=.01,tolerance=.1):
        # Distance Must be positive
        self.startPosition = None
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang
        self.tol = tolerance
        self.Done = False
        self.goalPosition = None
        self.distance = np.abs(distance)
    def update(self,measurement):
        x,y,theta,v,omega = measurement
        curPos = np.array([x,y])
        if self.goalPosition is None:
            self.goalPosition = curPos + self.distance * np.array([np.cos(theta),np.sin(theta)])
            self.startPosition = np.copy(curPos)
            self.goalAngle = theta
            self.projector = (self.goalPosition - self.startPosition) / self.distance
        
        self.d_err = np.dot(curPos - self.startPosition,self.projector) -self.distance
        self.v_err = np.dot(np.array([v * np.cos(theta),v*np.sin(theta)]),self.projector)

        self.theta_err = ((theta - self.goalAngle + np.pi) % (2*np.pi)) - np.pi
        self.omega_err = omega

        if np.abs(self.d_err) < self.tol and np.abs(self.v_err) < self.tol:
            self.Done = True

    def value(self):
        return np.array([-self.Kp * self.d_err-self.Kd * self.v_err,-self.Kp_ang * self.theta_err - self.Kd_ang*self.omega_err]) 
        
