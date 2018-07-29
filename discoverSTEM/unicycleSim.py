import numpy as np
import numpy.random as rnd
import pyglet
from pyglet.window import key
import matplotlib.pyplot as plt

def basicDynamics(x,u,dt):
    return x + dt* np.array([u[0]*np.cos(x[2]),u[0]*np.sin(x[2]),u[1]]) 

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

        pos = np.array([self.width//2,self.height//2])

        self.x = np.hstack([pos,0])
        
        self.v = 0.
        self.omega = 0.

        self.omegaMag = 1.
        self.vMag = 100.

    def on_key_press(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.omega = self.omegaMag
        elif symbol == key.RIGHT:
            self.omega = -self.omegaMag
        elif symbol == key.SPACE:
            self.v = self.vMag
        elif symbol == key.ESCAPE:
            self.close()

    def on_key_release(self,symbol,modifiers):
        if symbol == key.LEFT:
            self.omega = 0.
        elif symbol == key.RIGHT:
            self.omega = 0.
        elif symbol == key.SPACE:
            self.v = 0.
            
    def update(self,dt):
        u = np.array([self.v,self.omega])
        self.x = self.dynamics(self.x,u,dt)

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
                                     ('v2f',vertexMat.flatten()))
    def on_draw(self):
        self.clear()

        for label in self.Labels:
            label.draw()

        self.drawCar()

if __name__ == '__main__':
    window = unicycleSim(basicDynamics)
    pyglet.app.run()
