from model import *
import glm
from random import random
from vbo import *
from texture import get_program, get_texture, get_texture_cube, get_vao

def clamp(val, low, high):
    return max(low, min(high,val))


#
# Scene: holds models.
# Mesh: holds Model, texture
# Texture: holds textures
# Model: holds vao number, transformations, uniforms.
# VAO: holds VBO container, shaderprograms, turns them into vaos
# ShaderProgram: holds programs
# VBO: holds different vbos of objects

# basically:     ,- Texture
#    Main - Mesh            ,- ShaderProgram
#                `-   VAO`
#                           `- VBO
# it's stupid.
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


class VAO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = VBO(ctx)
        self.program = ShaderProgram(ctx)
        self.vaos = {}

        # cube vao
        self.vaos['cube'] = get_vao(ctx, 
            program=self.program.programs['default'],
            vbo = self.vbo.vbos['cube'])

        # cat vao
        self.vaos['cat'] = get_vao(ctx, 
            program=self.program.programs['default'],
            vbo=self.vbo.vbos['cat'])

        # skybox vao
        self.vaos['skybox'] = get_vao(ctx, 
            program=self.program.programs['skybox'],
            vbo=self.vbo.vbos['skybox'])

        # advanced_skybox vao
        self.vaos['advanced_skybox'] = get_vao(ctx, 
            program=self.program.programs['advanced_skybox'],
            vbo=self.vbo.vbos['advanced_skybox'])


    def destroy(self):
        self.vbo.destroy()
        self.program.destroy()


class ShaderProgram:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}
        self.programs['default'] = get_program(ctx, 'default')
        self.programs['skybox'] = get_program(ctx, 'skybox')
        self.programs['advanced_skybox'] = get_program(ctx, 'advanced_skybox')

    def destroy(self):
        [program.release() for program in self.programs.values()]


class VBO:
    def __init__(self, ctx):
        self.vbos = {}
        self.vbos['cube'] = CubeVBO(ctx)
        self.vbos['cat'] = FileVBO(ctx, 'objects/bunny/bunny.obj')
        self.vbos['skybox'] = SkyBoxVBO(ctx)
        self.vbos['advanced_skybox'] = AdvancedSkyBoxVBO(ctx)

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]


class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}
        self.textures[0] = get_texture(ctx, path='textures/img.png')
        self.textures[1] = get_texture(ctx, path='textures/img_1.png')
        self.textures[2] = get_texture(ctx, path='textures/img_2.png')
        self.textures['cat'] = get_texture(ctx, path='objects/bunny/UVMap.png')
        self.textures['skybox'] = get_texture_cube(ctx, dir_path='textures/skybox1/', ext='png')
    

    def destroy(self):
        [tex.release() for tex in self.textures.values()]


class Mesh:
    def __init__(self, app):
        self.app = app
        self.vao = VAO(app.ctx)
        self.texture = Texture(app.ctx)

    def destroy(self):
        self.vao.destroy()
        self.texture.destroy()