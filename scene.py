from model import *
from random import random

class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        n, s = 2, 1
        height = 0
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                if(random() < 0.3):
                    height += 1
                if(random() < 0.3):
                    height -= 1
                height = max(0, height)
                height = min(2,height)
                add(Cube(app, tex_id=int(1.2*random()), pos=(2*x, 2*height - s, 2*z)))

        #add(Cat(app,rot=(0,0,0), pos=(0, 5, -10),scale=(8,8,8)))

    def render(self):
        for obj in self.objects:
            obj.render()