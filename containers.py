from model import *
import glm
from random import random
from vbo import *
from texture import get_program, get_texture, get_texture_cube, get_vao
from types import SimpleNamespace
from constants import GLOBAL_CONSTANTS

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

    def clear_objects(self, obj):
        self.objects = [o for o in self.objects if type(o) != type(obj)]

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
                height = clamp(height, 0, 2)
                if(random()< 0.7):
                    y = (random()>0.5)
                    add(Cube(app, tex_id=int(0.8+1.4*random()),pos=(x, y, z)))
                        
        add(Cube(app, tex_id=1,pos=(10, 0, 0)))
        add(Cube(app, tex_id=1,pos=(0, 0, 10)))
        add(Cube(app, tex_id=1,pos=(10, 1, 0)))
        add(Cube(app, tex_id=1,pos=(0, -1, 0)))
        add(Cube(app, tex_id=1,pos=(10, 1, 3)))
        add(Cube(app, tex_id=1,pos=(10, 1, 5)))
        add(Cube(app, tex_id=1,pos=(10, 1, 4)))


                
        # for x in range(n):
        #     add(Cube(app, tex_id=1,pos=(x**3+50, 0, 0)))
        #     add(Cube(app, tex_id=1,pos=(0, 0, x**3+50)))
        #     add(Cube(app, tex_id=1,pos=(0, -x**3-50, 0)))



        # add(Cat(app, pos=(0, 0, 0)))

    def render(self):
        for obj in self.objects:
            obj.render()
        #self.skybox.render()


class Mesh:
    # Load all the files and generate buffers for the openGL context.
    def __init__(self, ctx: mgl.Context):
        self.ctx = ctx
        
        self.vbos = {}
        self.programs = {}
        self.vaos = {}
        self.textures = {}
        self.buffers = SimpleNamespace()

        self.gen_textures(ctx)
        self.gen_vbos(ctx)
        self.gen_programs(ctx)
        self.gen_vaos(ctx)
        self.gen_buffers(ctx)

    def gen_textures(self, ctx: mgl.Context):
        self.textures[0] = get_texture(ctx, path='textures/img.png')
        self.textures[1] = get_texture(ctx, path='textures/img_1.png')
        self.textures[2] = get_texture(ctx, path='textures/img_2.png')
        self.textures[3] = get_texture(ctx, path='textures/img_3.png')
        self.textures['cat'] = get_texture(ctx, path='objects/bunny/UVMap.png')
        self.textures['skybox'] = get_texture_cube(ctx, dir_path='textures/skybox1/', ext='png')
    
    def gen_vbos(self, ctx: mgl.Context):
        self.vbos['cube'] = CubeVBO(ctx)
        self.vbos['cat'] = FileVBO(ctx, 'objects/bunny/bunny.obj')
        self.vbos['skybox'] = SkyBoxVBO(ctx)
        self.vbos['advanced_skybox'] = AdvancedSkyBoxVBO(ctx)
    
    def gen_programs(self, ctx: mgl.Context):
        self.programs['default'] = get_program(ctx, 'default')
        self.programs['flat'] = get_program(ctx, 'default', 'default_flat')
        self.programs['alpha'] = get_program(ctx, 'default', 'default_alpha')
        self.programs['skybox'] = get_program(ctx, 'skybox')
        self.programs['advanced_skybox'] = get_program(ctx, 'advanced_skybox')
        self.programs['blit'] = get_program(ctx, 'screen')    #draw texture on screen for opencv.
        self.programs['sobel'] = get_program(ctx, 'screen', 'processing/sobel')
        self.programs['1d_gaussian'] = get_program(ctx, 'screen', 'processing/1d_gaussian')
        self.programs['draw_over'] = get_program(ctx, 'screen', 'draw_over')

    def gen_vaos(self, ctx: mgl.Context):
        # cube vao
        self.vaos['flat_cube'] = get_vao(ctx, 
            program=self.programs['flat'],
            vbo = self.vbos['cube'])
        
        self.vaos['cube'] = get_vao(ctx, program=self.programs['default'], vbo=self.vbos['cube'])

        self.vaos['alpha_cube'] = get_vao(ctx, program=self.programs['alpha'], vbo=self.vbos['cube'])

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
        
        self.vaos['blit'] = ctx.vertex_array(self.programs['blit'], [])
        self.vaos['blit'].vertices = 3

        self.vaos['sobel'] = ctx.vertex_array(self.programs['sobel'], [])
        self.vaos['sobel'].vertices = 3

        self.vaos['1d_gaussian'] = ctx.vertex_array(self.programs['1d_gaussian'], [])
        self.vaos['1d_gaussian'].vertices = 3

        self.vaos['draw_over'] = ctx.vertex_array(self.programs['draw_over'], [])
        self.vaos['draw_over'].vertices = 3

    def gen_buffers(self, ctx: mgl.Context):
        self.buffers.screen = ctx.screen
        self.buffers.fb_render_tex = ctx.texture((ctx.screen.size),4)
        self.buffers.fb_render_tex_depth = ctx.depth_renderbuffer(ctx.screen.size)
        self.buffers.fb_render = ctx.framebuffer(color_attachments=self.buffers.fb_render_tex,depth_attachment=self.buffers.fb_render_tex_depth)
        
        ssaa_size = (ctx.screen.width*2, ctx.screen.height*2)
        self.buffers.fb_ssaa_render_tex = ctx.texture(ssaa_size,4)
        self.buffers.fb_ssaa_render_tex_depth = ctx.depth_renderbuffer(ssaa_size)
        self.buffers.fb_ssaa_render = ctx.framebuffer(color_attachments=self.buffers.fb_ssaa_render_tex,depth_attachment=self.buffers.fb_ssaa_render_tex_depth)
        
        self.buffers.fb_aux_tex = ctx.texture((ctx.screen.size),4)
        self.buffers.fb_aux = ctx.framebuffer(color_attachments=self.buffers.fb_aux_tex)

        self.buffers.fb_binary_tex = ctx.texture((ctx.screen.size),1)
        self.buffers.fb_binary = ctx.framebuffer(color_attachments=self.buffers.fb_binary_tex)

        self.buffers.fb_screen_mix_tex = ctx.texture((ctx.screen.size),4)
        self.buffers.fb_screen_mix = ctx.framebuffer(color_attachments=self.buffers.fb_aux_tex)

        self.buffers.opencv_tex = ctx.texture((ctx.screen.width,ctx.screen.height),4)
        self.buffers.opencv = ctx.framebuffer(color_attachments=self.buffers.opencv_tex)


    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]
        [tex.release() for tex in self.textures.values()]
        [program.release() for program in self.programs.values()]
        [texorfbo.release() for texorfbo in self.buffers.__dict__.values()]


class Light:
    def __init__(self, position=(30, 30, 10), color=(1, 1, 1)):
        self.position = glm.vec3(position)
        self.color = glm.vec3(color)
        # intensities
        self.Ia = GLOBAL_CONSTANTS.light._K_AMBIENT * self.color  # ambient
        self.Id = GLOBAL_CONSTANTS.light._K_DIFFUSE * self.color  # diffuse
        self.Is = GLOBAL_CONSTANTS.light._K_SPECULAR * self.color  # specular