from model import *
import glm
from random import random

def clamp(val, low, high):
    return max(low, min(high,val))

class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        n, s = 3, 1
        height = 0
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                if random() < 0.3:
                    height += 1
                if random() < 0.3:
                    height -= 1
                height = clamp(height, 0, 3)
                add(Cube(app, tex_id=int(0.8+1.4*random()),pos=(2*x, 2*height -s, 2*z)))

        add(Cat(app, pos=(0, -2, -10)))

    def render(self):
        for obj in self.objects:
            obj.render()
        self.skybox.render()