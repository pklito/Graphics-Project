import pygame as pg
import moderngl as mgl
import struct

def get_texture_cube(ctx, dir_path, ext='png'):
        faces = ['right', 'left', 'top', 'bottom'] + ['front', 'back'][::-1]
        # textures = [pg.image.load(dir_path + f'{face}.{ext}').convert() for face in faces]
        textures = []
        for face in faces:
            texture = pg.image.load(dir_path + f'{face}.{ext}').convert()
            if face in ['right', 'left', 'front', 'back']:
                texture = pg.transform.flip(texture, flip_x=True, flip_y=False)
            else:
                texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
            textures.append(texture)

        size = textures[0].get_size()
        texture_cube = ctx.texture_cube(size=size, components=3, data=None)

        for i in range(6):
            texture_data = pg.image.tostring(textures[i], 'RGB')
            texture_cube.write(face=i, data=texture_data)

        return texture_cube


def get_texture(ctx, path):
        texture = pg.image.load(path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = ctx.texture(size=texture.get_size(), components=4,
                                   data=pg.image.tostring(texture, 'RGBA'))
        # mipmaps
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        # AF
        texture.anisotropy = 32.0
        return texture


def get_program(ctx: mgl.Context, shader_program_name, shader_frag_name = None):
        if shader_frag_name is None:
              shader_frag_name = shader_program_name
        with open(f'shaders/{shader_program_name}.vert') as file:
            vertex_shader = file.read()

        with open(f'shaders/{shader_frag_name}.frag') as file:
            fragment_shader = file.read()

        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program


def get_vao(ctx : mgl.Context, program, vbo):
        vao = ctx.vertex_array(program, [(vbo.vbo, vbo.format, *vbo.attribs)])
        return vao

def do_pass(fb_dest : mgl.Framebuffer, fb_prev: mgl.Framebuffer, vao : mgl.VertexArray, uniforms : dict = None):
    fb_dest.use()
    fb_prev.color_attachments[0].use()
    if uniforms is not None:
        for name, value in uniforms.items():
            vao.program[name].write(struct.pack('i', value))
    vao.render()
      