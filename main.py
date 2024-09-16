import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera
from containers import *
from opencv import opencv_process_fbo
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
        # increase line width
        self.ctx.line_width = 3.0
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0

        # configs
        self.SHOW_HOUGH = True
        self.PAUSED = False

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
        if event.key == pg.K_p:
            self.PAUSED = not self.PAUSED
            self.render()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                self.key_down(event)

    def render(self):
        # clear framebuffers
        self.ctx.clear()
        self.buffers.fb_render.clear(color=(0.1,0.1,0.2))
        self.buffers.fb_aux.clear()
        self.buffers.fb_binary.clear()

        # Render world
        self.buffers.fb_render.use()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.scene.render()

        #blit
        do_pass(self.buffers.fb_screen_mix, self.buffers.fb_render, self.mesh.vaos['blit'])

        # Do gaussian blur:
        do_pass(self.buffers.fb_aux, self.buffers.fb_render, self.mesh.vaos['1d_gaussian'], {"is_x" : 1})
        do_pass(self.buffers.fb_render, self.buffers.fb_aux, self.mesh.vaos['1d_gaussian'], {"is_x" : 0})

        do_pass(self.buffers.fb_binary, self.buffers.fb_render, self.mesh.vaos['sobel'])
        do_pass(self.buffers.fb_screen_mix, self.buffers.fb_binary, self.mesh.vaos['draw_over'])

        do_pass(self.ctx.screen, self.buffers.fb_screen_mix, self.mesh.vaos['blit'])
        # do overlay
        #self.buffers.screen.use()
        #opencv_process_fbo(self, self.buffers.fb_binary)
        # swap buffers
        pg.display.flip()



    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def run(self):
        while True:
            self.get_time()
            self.check_events()
            if not self.PAUSED:
                self.camera.update()
                self.render()
            self.delta_time = self.clock.tick(60)


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()
