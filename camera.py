import glm
import pygame as pg
import json

FOV = 90  # deg
NEAR = 0.1
FAR = 10000
SPEED = 0.008
SENSITIVITY = 0.04

CAMERA_PRINTS = False
def jsonprint(key, thing, comma = True):
    print(f'"{key}": {json.dumps(str(thing))}' + ("," if comma else ""))

def toEuclidian(vec4):
    return vec4/vec4.w
class Camera:
    def __init__(self, app, position=(0, 0, 4), yaw=30, pitch=0):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.forward_level = self.forward #forward without the pitch.
        self.yaw = yaw
        self.pitch = pitch
        # view matrix
        self.m_view = self.get_view_matrix()
        # projection matrix
        self.m_proj = self.get_projection_matrix()

    def rotate(self):
        rel_x, rel_y = pg.mouse.get_rel()
        self.yaw += rel_x * SENSITIVITY
        self.pitch -= rel_y * SENSITIVITY
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)

        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.right.x = glm.cos(yaw + glm.pi() / 2)
        self.right.y = 0
        self.right.z = glm.sin(yaw + glm.pi() / 2)

        self.up.x = -glm.cos(yaw) * glm.sin(pitch)
        self.up.y = glm.cos(pitch)
        self.up.z = -glm.sin(yaw) * glm.sin(pitch)

        self.forward_level = glm.vec3(glm.cos(yaw),0,glm.sin(yaw))

        # self.forward = glm.normalize(self.forward)
        # self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        # self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        self.move()
        self.rotate()
        self.update_camera_vectors()
        self.m_view = self.get_view_matrix()
        if CAMERA_PRINTS:
            print("{")
            jsonprint("yaw", self.yaw)
            jsonprint("pitch", self.pitch)
            jsonprint("m_view", self.m_view)
            jsonprint("up", self.up)
            jsonprint("right", self.right)
            jsonprint("forward", self.forward)
            jsonprint("x_inf", toEuclidian(self.m_proj*self.m_view*glm.vec4(100000,0,0,1)))
            jsonprint("y_inf", toEuclidian(self.m_proj*self.m_view*glm.vec4(0,100000,0,1)))
            jsonprint("z_inf", toEuclidian(self.m_proj*self.m_view*glm.vec4(0,0,100000,1)))
            jsonprint("proj", self.m_proj, False)
            print("},")


    def move(self):
        velocity = SPEED * self.app.delta_time
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.position += self.forward_level * velocity
        if keys[pg.K_s]:
            self.position -= self.forward_level * velocity
        if keys[pg.K_a]:
            self.position -= self.right * velocity
        if keys[pg.K_d]:
            self.position += self.right * velocity
        if keys[pg.K_SPACE]:
            self.position += glm.vec3(0,1,0) * velocity
        if keys[pg.K_LSHIFT]:
            self.position -= glm.vec3(0,1,0) * velocity

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.forward, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(FOV), self.aspect_ratio, NEAR, FAR)




















