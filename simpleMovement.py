import numpy as np
import pyglet
from pyglet.window import key


class oneDGame(pyglet.window.Window):
    def __init__(self):
        
        super(oneDGame, self).__init__()
        
        self.label_top = pyglet.text.Label('Make the green box stop on the white box',
                                      font_size=16,
                                      x= self.width//2,y=3*self.height//4,
                                      anchor_x = 'center',anchor_y = 'center')

        self.label_bot = pyglet.text.Label('Use the arrows',
                                           font_size=16,
                                           x=self.width//2,y=3*self.height//4 - 24,
                                           anchor_x = 'center',anchor_y = 'center')

        self.label_exit = pyglet.text.Label('Press Esc to exit',
                                            font_size = 16,
                                            x=self.width//2,y=self.height//4,
                                            anchor_x = 'center',anchor_y = 'center')
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
        self.vel = np.zeros(2)
        self.force = np.zeros(2)
        self.damping = .01

        self.forceMag = 100.

        
        pyglet.clock.schedule_interval(self.update,1./60.)
        
    def drawBox(self,pos,scale,color = (255,255,255,255)):
        numVertices = len(self.vertexMatDiff)
        vertexMat = scale * self.vertexMatDiff + np.tile(pos,(numVertices,1))
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                                     [0, 1, 2, 0, 2, 3],
                                     ('v2f', vertexMat.flatten()),
                                     ('c4B',color * 4))
 

    def update(self,dt):

        stopped = np.linalg.norm(self.vel) < .5
        onTarget = np.linalg.norm(self.pos-self.target_pos,ord=np.inf) < (self.target_width-self.cursor_width)/2
        if onTarget and stopped:
            self.close()
    
        self.pos = self.pos + self.vel * dt
        self.vel = self.vel + dt * (self.force - self.damping * self.vel)

    
    def on_draw(self):
        self.clear()

        self.label_top.draw()
        self.label_bot.draw()
        self.label_exit.draw()
        self.drawBox(self.target_pos,self.target_width)
        self.drawBox(self.pos,self.cursor_width,(34, 139, 34, 1))


    def on_key_press(self,symbol, modifiers):
        if symbol == key.LEFT:
            self.force[0] = -self.forceMag
        
        elif symbol == key.RIGHT:
            self.force[0] = self.forceMag

    def on_key_release(self,symbol, modifiers):
        if symbol == key.LEFT:
            self.force[0] = 0.
        elif symbol == key.RIGHT:
            self.force[0] = 0.

def runOneD():
    window = oneDGame()
    pyglet.app.run()

if __name__ == '__main__':
    window = oneDGame()
    pyglet.app.run()
