import numpy as np
import numpy.random as rnd
import pyglet
from pyglet.window import key
import matplotlib.pyplot as plt


class gantrySim(pyglet.window.Window):
    def __init__(self,damping):
        super().__init__()

        # Here are some nice vertices.

        self.cursor_width = 50
        self.target_width = 60
        vertexTuple = (0,0,
                       1,0,
                       1,1,
                       0,1)

        numVertices = int(len(vertexTuple)/2)
        vertexMat = np.array(vertexTuple,dtype='float').reshape((numVertices,2))

        pos = np.mean(vertexMat,axis=0)
        self.vertexMatDiff = vertexMat - np.tile(pos,(numVertices,1))


        self.target_pos = np.array([7*self.width/8, self.height/2])

        # Let's subtract this off so that we start in the middle / left. 
        self.pos = np.array([self.width/8,self.height/2])
        self.noise_std = 5.
        self.vel = np.zeros(2)
        self.force = np.zeros(2)
        self.damping = 0.01

        label_exit = pyglet.text.Label('Press Esc to exit',
                                       font_size = 16,
                                       x=self.width//2,y=self.height//4,
                                       anchor_x = 'center',anchor_y = 'center')
        self.Labels = [label_exit]


        self.TimeTraj = [0.]
        self.PosTraj = [self.pos]
        pyglet.clock.schedule_interval(self.update,1./60.)

    def on_window_close(self):
        plt.plot(self.TimeTraj,np.array(self.PosTraj))
    def update(self,dt):
        
        stopped = np.linalg.norm(self.vel) < 2.
        onTarget = np.linalg.norm(self.pos-self.target_pos,ord=np.inf) < (self.target_width-self.cursor_width)/2
        if onTarget and stopped:
            self.close()

        self.update_force(dt)

        noise = self.noise_std * rnd.randn(2) * np.sqrt(dt)
        self.pos = self.pos + self.vel * dt
        self.vel = self.vel + dt * (self.force - self.damping * self.vel) + noise

        self.TimeTraj.append(self.TimeTraj[-1]+dt)
        self.PosTraj.append(self.pos)
    def update_force(self,dt):
        pass
    
    def on_draw(self):
        self.clear()

        for label in self.Labels:
            label.draw()

        self.drawBox(self.target_pos,self.target_width)
        self.drawBox(self.pos,self.cursor_width,(34, 139, 34, 1))

    def drawBox(self,pos,scale,color = (255,255,255,255)):
        numVertices = len(self.vertexMatDiff)
        vertexMat = scale * self.vertexMatDiff + np.tile(pos,(numVertices,1))
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                                     [0, 1, 2, 0, 2, 3],
                                     ('v2f', vertexMat.flatten()),
                                     ('c4B',color * 4))
 

class gantryGame(gantrySim):
    def __init__(self,damping):
        
        super().__init__(damping)
        
        label_top = pyglet.text.Label('Make the green box stop on the white box',
                                      font_size=16,
                                      x= self.width//2,y=3*self.height//4,
                                      anchor_x = 'center',anchor_y = 'center')

        label_bot = pyglet.text.Label('Use the arrows',
                                      font_size=16,
                                      x=self.width//2,y=3*self.height//4 - 24,
                                      anchor_x = 'center',anchor_y = 'center')
        self.Labels.extend([label_top,label_bot])

        
        self.forceMag = 100.

        
        

    def on_key_press(self,symbol, modifiers):
        if symbol == key.LEFT:
            self.force[0] = -self.forceMag
        
        elif symbol == key.RIGHT:
            self.force[0] = self.forceMag

        elif symbol == key.UP:
            self.force[1] = self.forceMag
        elif symbol == key.DOWN:
            self.force[1] = -self.forceMag
        elif symbol == key.ESCAPE:
            self.close()
            
    def on_key_release(self,symbol, modifiers):
        if symbol == key.LEFT:
            self.force[0] = 0.
        elif symbol == key.RIGHT:
            self.force[0] = 0.
        elif symbol == key.UP:
            self.force[1] = 0.
        elif symbol == key.DOWN:
            self.force[1] = 0.


class gantryPD(gantrySim):
    def __init__(self,damping,kP,kD):
        super().__init__(damping)

        self.pos_last = np.copy(self.pos)

        self.kP = kP
        self.kD = kD
        # To properly simulate the effect of a step, you need this.
        # This says that initially, the error is actually zero.
        self.error_last = np.zeros(2)

        self.forceMax = np.inf
        
    def update_force(self,dt):
        self.error = self.target_pos - self.pos
        error_deriv = (self.error - self.error_last) / dt
        self.error_last = self.error

        
        self.force = self.kP * self.error + self.kD * error_deriv
        self.force = np.clip(self.force,-self.forceMax,self.forceMax)

    
    
        
damping = 0.01
kP = 5.
kD = 2.
# This is a hack. I've hardcoded the initial condtion and error 
pos = np.array([80.,240.])
error = np.array([480.,0.])


def runWindow(window):
   pyglet.app.run()
   plt.plot(window.TimeTraj,np.array(window.PosTraj))
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.legend(['X','Y'],loc='best')
def manualControl():
    window = gantryGame(damping)
    runWindow(window)
def pdControl(kP,kD):
    window = gantryPD(damping,kP,kD)
    runWindow(window)

if __name__ == '__main__':
    window = gantryPD(damping,kP,kD)
    # window = gantryGame(damping)
    pyglet.app.run()
