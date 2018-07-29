import numpy as np
import numpy.random as rnd
import pyglet
from pyglet.window import key
import matplotlib.pyplot as plt

def basicDynamics(x,u,dt):
    return x

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

    def update(self,dt):
        pass

    def on_draw(self):
        self.clear()

        for label in self.Labels:
            label.draw()


if __name__ == '__main__':
    window = unicycleSim(basicDynamics)
    pyglet.app.run()
