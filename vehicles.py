import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import pyglet as pg
from pyglet.gl import *
from pyglet.window import key, mouse
## Rolling Sphere ## 
numPrimes = 20
numMeridians = 20


def sphere_vertices(x,y,z,n):

    Vertices = [] 
    top = np.array([x,y+n,z])

    Vertices.append(top)

    alpha = np.pi/(numPrimes+1)
    beta = 2*np.pi / (numMeridians)
    for p in range(numPrimes):
        theta = alpha * (p + 1)
        y_val = y+n*np.cos(theta)
        for m in range(numMeridians):
            phi = beta * m
            x_val = x + n* np.cos(phi) * np.sin(theta)
            z_val = z + n * np.sin(phi) * np.sin(theta)

            Vertices.append(np.array([x_val,y_val,z_val]))
            
            
    bottom = np.array([x,y-n,z])
    Vertices.append(bottom)

    Vertices = np.array(Vertices)
    return Vertices

def sphere_sequence():
    nv = numPrimes*numMeridians + 2
    Seq = []
    # Top triangles
    for p in range(numMeridians-1):
        Seq.extend([p+1,p+2,0])
    Seq.extend([numMeridians,1,0])

    # Middle Triangles
    # (1,2,5), (2,5,6),  (2,3,6), (3,6,7),  (3,4,7), (4,7,8),  (4,1,8), (1,8,5)
    # (5,6,9), (6,9,10), (6,7,10),(7,10,11),(7,8,11),(8,11,12),(8,5,12),(5,12,9)

    for p in range(numPrimes-1):
        offset = p * numMeridians
        for m in range(numMeridians-1):
            Seq.extend([offset+m+1,offset+m+2,offset+m+1+numMeridians])
            Seq.extend([offset+m+2,offset+m+1+numMeridians,offset+m+2+numMeridians])
        Seq.extend([offset+numMeridians,offset+1,offset+2*numMeridians])
        Seq.extend([offset+1,offset+2*numMeridians,offset+numMeridians+1])

    # 9 = 2*4+1 = (numPrimes-1) * numMeridians + 1
    # Bottom triangles
    # (13,9,10),(13,10,11),(13,11,12),(13,12,9)
    offset = (numPrimes-1) * numMeridians
    for p in range(numMeridians-1):
        Seq.extend([nv-1, offset + p+1, offset + p+2])
    Seq.extend([nv-1,offset+numMeridians,offset + 1])
    return Seq

    
sphereVertsColors = np.zeros((2+numPrimes*numMeridians,3),dtype=int)
sphereVertsColors[:,0] = rnd.randint(0,256,size=len(sphereVertsColors))
sphereVertsColors = sphereVertsColors.flatten()

class rollingSphere:
    def __init__(self,position,radius,SPEED):
        self.R = np.eye(3)
        self.position = np.array(position)
        self.vertexColors = sphereVertsColors
        self.radius = radius
        self.Seq = sphere_sequence()
        # Assuming that y velocity is always zero
        # Just using a normalized velocity
        self.velocity = np.zeros(3)
        self.SPEED = SPEED
        

    def get_vertices(self):
        x,y,z = self.position
        V = sphere_vertices(0,0,0,self.radius)
        VR = V@self.R.T
        
        return VR + np.outer(np.ones(len(V)),self.position)


    def get_angular_velocity(self):
        M = np.array([[0,0,-1],
                      [0,1,0],
                      [1,0,0]])

        return M@self.velocity*self.SPEED /self.radius

    def update(self,dt):
        self.position = self.position + dt * self.velocity * self.SPEED

        omega = self.get_angular_velocity()
        Omega = np.cross(omega,np.eye(3))

        self.R = la.expm(Omega * dt) @ self.R 

    def draw(self):
        Verts = self.get_vertices()
        Seq = self.Seq
        colors = self.vertexColors
        pg.graphics.draw_indexed(len(Verts),GL_TRIANGLES,
                                 Seq,
                                 ('v3f',Verts.flatten()),
                                 ('c3B',colors))


    def on_key_press(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.velocity[0] += 1
        elif symbol == key.RIGHT:
            self.velocity[0] -= 1
        elif symbol == key.UP:
            self.velocity[2] += 1
        elif symbol == key.DOWN:
            self.velocity[2] -= 1

    def on_key_release(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.velocity[0] -= 1
        elif symbol == key.RIGHT:
            self.velocity[0] += 1
        elif symbol == key.UP:
            self.velocity[2] -= 1
        elif symbol == key.DOWN:
            self.velocity[2] += 1

carBotY = -.1
carMidY = 0.075
carTopY = .15
cw = .08
cl = .08
carVertices = np.array([[cw*4.,carBotY,-4*cl],
                        [cw*4,carBotY,-2*cl],
                        [cw*4,carBotY,-1*cl],
                        [cw*4,carBotY,1*cl],
                        [cw*4,carBotY,2*cl],
                        [cw*4,carBotY,4*cl],
                        [cw*2,carBotY,5.5*cl],
                        [cw*-2,carBotY,5.5*cl],
                        [cw*-4,carBotY,4*cl],
                        [cw*-4,carBotY,2*cl],
                        [cw*-4,carBotY,1*cl],
                        [cw*-4,carBotY,-1*cl],
                        [cw*-4,carBotY,-2*cl],
                        [cw*-4,carBotY,-4*cl],
                        [cw*4.,carMidY,-4*cl],
                        [cw*4,carMidY,-2*cl],
                        [cw*3.,carTopY,-1*cl],
                        [cw*3.,carTopY,1*cl],
                        [cw*4,carMidY,2*cl],
                        [cw*4,carMidY,4*cl],
                        [cw*2,carMidY,5.*cl],
                        [cw*-2,carMidY,5.*cl],
                        [cw*-4,carMidY,4*cl],
                        [cw*-4,carMidY,2*cl],
                        [cw*-3.,carTopY,1*cl],
                        [cw*-3.,carTopY,-1*cl],
                        [cw*-4,carMidY,-2*cl],
                        [cw*-4,carMidY,-4*cl],
                        [cw*4,carMidY,-1*cl],
                        [cw*4,carMidY,1*cl],
                        [cw*-4,carMidY,-1*cl],
                        [cw*-4,carMidY,1*cl]])

carVertices = carVertices @ np.array([[0,0,-1],
                                      [0,1,0],
                                      [1,0,0]])

carSeq = [0,1,13,
          1,2,12,
          2,3,11,
          3,4,10,
          4,5,9,
          5,6,8,
          6,7,8,
          5,8,9,
          4,9,10,
          3,10,11,
          2,11,12,
          1,12,13,
          14,15,27,
          15,16,26,
          16,17,25,
          17,18,24,
          18,19,23,
          19,20,22,
          20,21,22,
          19,22,23,
          18,23,24,
          17,24,25,
          16,25,26,
          15,26,27,
          0,1,15,
          0,14,15,
          1,2,28,
          1,15,28,
          15,16,28,
          2,3,29,
          2,28,29,
          17,28,29,
          16,17,28,
          3,4,18,
          3,18,29,
          17,18,29,
          4,5,19,
          4,18,19,
          5,6,20,
          5,19,20,
          7,8,21,
          8,21,22,
          8,9,22,
          9,22,23,
          9,10,23,
          10,23,31,
          23,24,31,
          10,11,31,
          11,30,31,
          24,30,31,
          24,25,30,
          11,12,30,
          12,26,30,
          25,26,30,
          12,13,26,
          13,26,27,
          6,7,21,
          6,20,21,
          0,13,14,
          13,14,27]


carColors = np.array([[0,0,255],#0 
                      [0,0,255],#1
                      [0,0,255],#2
                      [0,0,255],#3
                      [0,0,255],#4
                      [0,0,255],#5
                      [0,0,255],#6
                      [0,0,255],#7
                      [0,0,255],#8
                      [0,0,255],#9
                      [0,0,255],#10
                      [0,0,255],#11
                      [0,0,255],#12
                      [0,0,255],#13
                      [35,45,155],#14
                      [35,45,155],#15
                      [155,100,70],#16
                      [155,100,70],#17
                      [35,45,155],#18
                      [35,45,155],#19
                      [70,50,155],#20
                      [70,50,155],#21
                      [35,45,155],#22
                      [35,45,155],#23
                      [155,100,70],#24
                      [155,100,70],#25
                      [35,45,155],#26
                      [35,45,155],#27
                      [0,0,0],#28
                      [0,0,0],#29
                      [0,0,0],#30
                      [0,0,0]]).flatten()                      
class car:
    def __init__(self,position,orientation,scale,SPEED):
        self.SPEED = SPEED
        self.position = np.array(position)
        self.theta = orientation
        self.scale = scale

        self.v = 0
        self.omega = 0

        
    def get_vertices(self): 
        V = carVertices
        nv = len(V)
        theta = self.theta
        R = np.array([[np.cos(theta),0,-np.sin(theta)],
                      [0,1,0],
                      [np.sin(theta),0,np.cos(theta)]])
        V = V @ R.T * self.scale + np.tile(self.position,(nv,1))
        return V 

    def draw(self):
        Verts = self.get_vertices() 
        Seq = carSeq
        colors = carColors
        pg.graphics.draw_indexed(len(Verts),GL_TRIANGLES,
                                 Seq,
                                 ('v3f',Verts.flatten()),
                                 ('c3B',colors))


    def update(self,dt):
        theta = self.theta
        v = self.v
        self.position = self.position + np.array([[np.cos(theta),0,np.sin(theta)]])*v*dt*self.SPEED
        self.theta = theta + dt * self.omega * self.SPEED
    
    def on_key_press(self,symbol,modifiers):
        if symbol == key.UP:
            self.v += 1
        elif symbol == key.DOWN:
            self.v -= 1
        elif symbol == key.LEFT:
            self.omega -= 1
        elif symbol == key.RIGHT:
            self.omega += 1


            
    def on_key_release(self,symbol,modifiers):
        if symbol == key.UP:
            self.v -= 1
        elif symbol == key.DOWN:
            self.v += 1
        elif symbol == key.LEFT:
            self.omega += 1
        elif symbol == key.RIGHT:
            self.omega -= 1



        
        
