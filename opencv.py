import cv2 as cv
import numpy as np
import moderngl as mgl

fps = 0.0
def opencv_process(app):
    ctx = app.ctx

    # Get the screen as a buffer
    buffer = ctx.screen.read(components=3,dtype="f4")
    raw = np.frombuffer(buffer,dtype="f4")
    image = raw.reshape((ctx.screen.height,ctx.screen.width,3))[::-1,:,::-1]
    
    ### CALCULATIONS ###
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #blue_channel, green_channel, red_channel = cv.split(image)
    # Apply GaussianBlur to reduce noise and improve contour detection

    def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(blurred.astype(np.uint8), thresh1, thresh2)

    #canny_split = cv.cvtColor(cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)]),cv.COLOR_BGR2GRAY)
    canny = get_edges_image(gray)
    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)
    lines = None
    ### OVERLAY DRAW ###
    overlay = np.zeros((ctx.screen.height,ctx.screen.width,4),dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(overlay, (x1, y1), (x2, y2), (0, 0, 255,255), 2)
    global fps
    fps = app.clock.get_fps()
    cv.putText(overlay,"fps: " + str(round(fps,2)),(0,50),cv.FONT_HERSHEY_PLAIN,1,(0,0,0,255),2)
    ### DRAW ON SCREEN ###
    buffer = overlay.tobytes()
    app.mesh.textures['opencv'].write(buffer)
    ctx.enable_only(ctx.BLEND)
    app.mesh.textures['opencv'].use()
    app.mesh.vaos['opencv'].render()
    
