import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera
from containers import *
from opencv import postProcessFbo, exportFbo
from constants import loadConstants
from texture import do_pass

class GraphicsEngine:
    def __init__(self, win_size=(600, 400)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)

        #self.ctx.enable(flags=mgl.AA)
        # increase line width
        self.ctx.line_width = 3.0
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0

        # configs
        self.SHOW_HOUGH = True
        self.PAUSED = False
        self.EXPORT = False

        # light
        self.light = Light()
        # camera
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self.ctx)
        # buffers
        self.buffers = self.mesh.buffers
        # scene
        self.scene = Scene(self)

    def key_down(self, event):
        if event.key == pg.K_v:
            self.SHOW_HOUGH = not self.SHOW_HOUGH
        if event.key == pg.K_r:
            loadConstants()
        if event.key == pg.K_t:
            self.EXPORT = True
        if event.key == pg.K_p:
            self.PAUSED = not self.PAUSED
            self.opencv_pipeline()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                self.key_down(event)

    def clear_buffers(self):
        # clear framebuffers
        self.ctx.clear(red=0.5, green=0.6, blue=0.95)
        self.buffers.fb_render.clear(color=(1,1,1))
        self.buffers.fb_aux.clear()
        self.buffers.fb_binary.clear()

    def render(self, target = None):
        if target is None:
            target = self.buffers.fb_render
        # Render world
        target.use()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.scene.render()

    def render_shaders(self, source = None, target = None):
        if(source is None):
            source = self.buffers.fb_render
        if(target is None):
            target = self.ctx.screen
        
        #blit
        do_pass(self.buffers.fb_screen_mix, source, self.mesh.vaos['blit'])

        # Do gaussian blur:
        do_pass(self.buffers.fb_aux, self.buffers.fb_render, self.mesh.vaos['1d_gaussian'], {"is_x" : 1})
        do_pass(self.buffers.fb_render, self.buffers.fb_aux, self.mesh.vaos['1d_gaussian'], {"is_x" : 0})

        do_pass(self.buffers.fb_binary, self.buffers.fb_render, self.mesh.vaos['sobel'])
        do_pass(self.buffers.fb_screen_mix, self.buffers.fb_binary, self.mesh.vaos['draw_over'])

        do_pass(target, self.buffers.fb_screen_mix, self.mesh.vaos['blit'])

    def flip_buffers(self):
        """Flip the buffers, useful if you want to do something before the flip (like exporting)"""
        if self.EXPORT:
            self.EXPORT = False
            print("camera proj:", self.camera.m_view)
            print("cubes (and rabbit):" + str([[x/2 for x in b.pos] for b in self.scene.objects]))
            exportFbo(self.buffers.screen, "output.png")
        pg.display.flip()

    def render_pipeline(self):
        self.clear_buffers()
        self.render(target=self.buffers.screen)
        #self.render_shaders(source=self.buffers.fb_render, target=self.buffers.screen)
        self.flip_buffers()

    def opencv_pipeline(self):
        self.clear_buffers()
        self.render()
        self.do_overlay(source=self.buffers.fb_render)
        
        self.flip_buffers()

    def do_overlay(self, target = None, source = None):
        if source is None:
            source = self.buffers.screen
        if target is None:
            target = self.buffers.screen
                # do overlay
        postProcessFbo(self, source)
        do_pass(target, self.buffers.opencv, self.mesh.vaos['blit'], {"flip_y": 1})



    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def run(self):
        while True:
            self.get_time()
            self.check_events()
            if not self.PAUSED:
                self.camera.update()
                if GLOBAL_CONSTANTS.opencv.DO_POST_PROCESS:
                    self.opencv_pipeline()
                else:
                    self.render_pipeline()
            
            self.delta_time = self.clock.tick(60)


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()
