import pygame as pg
import moderngl as mgl
import sys

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

        self.ctx = mgl.create_context()

        self.clock = pg.time.Clock()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # swap buffers
        pg.display.flip()

    def run(self):
        while True:
            self.check_events()
            self.render()
            self.delta_time = self.clock.tick(60)


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()