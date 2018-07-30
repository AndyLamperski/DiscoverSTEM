import numpy as np
import numpy.random as rnd
import pyglet
from pyglet.window import key
import matplotlib.pyplot as plt

def basicDynamics(x,u):
    return np.array([u[0]*np.cos(x[2]),u[0]*np.sin(x[2]),u[1]]) 

class unicycleSim(pyglet.window.Window):
    def __init__(self,dynamics):
        super().__init__()

        self.dynamics = dynamics

        label_exit = pyglet.text.Label('Press Esc to exit',
                                       font_size = 16,
                                       x=self.width//2,y=self.height//4,
                                       anchor_x = 'center',anchor_y = 'center')
        self.Labels = [label_exit]    

        
        pyglet.clock.schedule_interval(self.update,1./60.)

        self.car_width = 50
        self.car_length = 80
        vertexArray = np.array([0,-self.car_width/2,
                                self.car_length,0,
                                0,self.car_width/2])

        numVertices = len(vertexArray)//2
        vertexMat = np.reshape(vertexArray,(numVertices,2))
        pos = np.mean(vertexMat,axis=0)

        self.vertexMatDiff = vertexMat - np.tile(pos,(numVertices,1))

        
        pos = np.array([self.width//8,(self.height//8)])

        self.x = np.hstack([pos,0.])
        
        # Make the target
        vertexTuple = (0,0,
                       1,0,
                       1,1,
                       0,1)

        numVertices = int(len(vertexTuple)/2)
        vertexMat = np.array(vertexTuple,dtype='float').reshape((numVertices,2))

        pos = np.mean(vertexMat,axis=0)
        self.targetVertexMat = vertexMat - np.tile(pos,(numVertices,1))


        self.target_pos = np.array([7*self.width/8, 7* self.height/8])


        self.target_width = 70


        

        self.u = np.zeros(2)

        self.omegaMax = 1.
        self.vMax = 100.

           
    def update(self,dt):
        pos = self.x[:2]
        onTarget = np.linalg.norm(pos-self.target_pos,ord=np.inf) < (self.target_width-self.car_width)/2
        if onTarget:
            self.close()
        self.x = self.x + dt * self.dynamics(self.x,self.u)

        
        
    def rotationMatrix(self):
        theta = self.x[2]
        R = np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]])
        return R
                                 
                                 
    def drawCar(self):
                                 
        numVertices = len(self.vertexMatDiff)

        R = self.rotationMatrix()
        pos = self.x[:2]
        vertexMat = np.dot(self.vertexMatDiff,R.T) +\
                    np.tile(pos,(numVertices,1))
        pyglet.graphics.draw_indexed(numVertices,pyglet.gl.GL_TRIANGLES,
                                     [0,1,2],
                                     ('v2f',vertexMat.flatten()),
                                     ('c4B',(34, 139, 34, 1) * 3))

    def drawBox(self,pos,color = (135, 206, 235, 1)):
        numVertices = len(self.targetVertexMat)
        vertexMat = self.target_width * self.targetVertexMat + np.tile(pos,(numVertices,1))
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                                     [0, 1, 2, 0, 2, 3],
                                     ('v2f', vertexMat.flatten()),
                                     ('c4B',color * 4))
 

    def on_draw(self):
        self.clear()

        for label in self.Labels:
            label.draw()

        self.drawBox(self.target_pos)
        self.drawCar()

class unicycleGame(unicycleSim):
    def __init__(self,dynamics=basicDynamics):
        super().__init__(dynamics)

        label_top = pyglet.text.Label('Drive the triangle to the box',
                                      font_size=16,
                                      x= self.width//2,y=3*self.height//4,
                                      anchor_x = 'center',anchor_y = 'center')

        label_bot = pyglet.text.Label('Use the arrows and space bar',
                                      font_size=16,
                                      x=self.width//2,y=3*self.height//4 - 24,
                                      anchor_x = 'center',anchor_y = 'center')
        self.Labels.extend([label_top,label_bot])


    def on_key_press(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.u[1] = self.omegaMax
        elif symbol == key.RIGHT:
            self.u[1] = -self.omegaMax
        elif symbol == key.SPACE:
            self.u[0] = self.vMax
        elif symbol == key.ESCAPE:
            self.close()

    def on_key_release(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.u[1] = 0.
        elif symbol == key.RIGHT:
            self.u[1] = 0.
        elif symbol == key.SPACE:
            self.u[0] = 0.

def closedLoopDynamics(x,u,dt,target_pos):
    err = x[:2] - target_pos
    alpha = np.arctan2(err[1],err[0])
    theta = x[2]

    angleError = theta - alpha - np.pi
    # Shift it
    angleError = ((angleError + np.pi) % (2*np.pi)) - np.pi 
    
    if np.cos(theta-alpha) < 0:
        v = 100.
    else:
        v = 0.

    omega = np.clip(-angleError,-1,1) 

    # omega = 0.
    
    u = np.array([v,omega])
    
    return basicDynamics(x,u,dt,target_pos)

def runGame(dynamics):
    window = unicycleGame(dynamics)
    pyglet.app.run()

if __name__ == '__main__':
    # window = unicycleGame()
    window = unicycleSim(closedLoopDynamics)
    pyglet.app.run()
