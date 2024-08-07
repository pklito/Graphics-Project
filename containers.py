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
# Light: holds color values

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
        #self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        n, s = 10, 1
        height = 0
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                if random() < 0.3:
                    height += 1
                if random() < 0.3:
                    height -= 1
                height = clamp(height, 0, 2)
                add(Cube(app, tex_id=int(0.8+1.4*random()),pos=(2*x, 2*height -s, 2*z)))

        add(Cat(app, pos=(0, -2, -10)))

    def render(self):
        for obj in self.objects:
            obj.render()
        #self.skybox.render()

class Mesh:
    # Load all the files and generate buffers for the openGL context.
    def __init__(self, ctx):
        self.ctx = ctx
        
        self.vbos = {}
        self.programs = {}
        self.vaos = {}
        self.textures = {}

        self.gen_textures(ctx)
        self.gen_vbos(ctx)
        self.gen_programs(ctx)
        self.gen_vaos(ctx)

    def gen_textures(self, ctx):
        self.textures[0] = get_texture(ctx, path='textures/img.png')
        self.textures[1] = get_texture(ctx, path='textures/img_1.png')
        self.textures[2] = get_texture(ctx, path='textures/img_2.png')
        self.textures['cat'] = get_texture(ctx, path='objects/bunny/UVMap.png')
        self.textures['skybox'] = get_texture_cube(ctx, dir_path='textures/skybox1/', ext='png')
        self.textures['opencv'] = ctx.texture((ctx.screen.width,ctx.screen.height),4)
    
    def gen_vbos(self,ctx):
        self.vbos['cube'] = CubeVBO(ctx)
        self.vbos['cat'] = FileVBO(ctx, 'objects/bunny/bunny.obj')
        self.vbos['skybox'] = SkyBoxVBO(ctx)
        self.vbos['advanced_skybox'] = AdvancedSkyBoxVBO(ctx)
    
    def gen_programs(self, ctx):
        self.programs['default'] = get_program(ctx, 'default')
        self.programs['flat'] = get_program(ctx, 'default', 'default_flat')
        self.programs['skybox'] = get_program(ctx, 'skybox')
        self.programs['advanced_skybox'] = get_program(ctx, 'advanced_skybox')
        self.programs['opencv'] = get_program(ctx, 'screen')    #draw texture on screen for opencv.

    def gen_vaos(self, ctx):
        # cube vao
        self.vaos['cube'] = get_vao(ctx, 
            program=self.programs['flat'],
            vbo = self.vbos['cube'])

        # cat vao
        self.vaos['cat'] = get_vao(ctx, 
            program=self.programs['default'],
            vbo=self.vbos['cat'])

        # skybox vao
        self.vaos['skybox'] = get_vao(ctx, 
            program=self.programs['skybox'],
            vbo=self.vbos['skybox'])

        # advanced_skybox vao
        self.vaos['advanced_skybox'] = get_vao(ctx, 
            program=self.programs['advanced_skybox'],
            vbo=self.vbos['advanced_skybox'])
        
        self.vaos['opencv'] = ctx.vertex_array(self.programs['opencv'], [])
        self.vaos['opencv'].vertices = 3

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]
        [tex.release() for tex in self.textures.values()]
        [program.release() for program in self.programs.values()]


class Light:
    def __init__(self, position=(30, 30, 10), color=(1, 1, 1)):
        self.position = glm.vec3(position)
        self.color = glm.vec3(color)
        # intensities
        self.Ia = 0.06 * self.color  # ambient
        self.Id = 0.8 * self.color  # diffuse
        self.Is = 1.0 * self.color  # specular