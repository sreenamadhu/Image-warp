import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import os

OUT_DIR  = 'exported_data/'
r = 5
frame_width = 500
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

for task in ['crop', 'warp']:
    if not os.path.isdir(os.path.join(OUT_DIR,task)):
        os.makedirs(os.path.join(OUT_DIR,task))


def init_state():

    if 'points' not in st.session_state:
        st.session_state['points'] = []

    if 'idx' not in st.session_state:
        st.session_state['idx'] = 1

    if 'task' not in st.session_state:
        st.session_state['task'] = 'None'
        st.session_state['task_data'] = None

    return

def warp(path, input_points):
    im = Image.open(path)
    w,h = im.size
    frame_height = frame_width*(h/w)
    out_points = []
    for point in input_points:
        out_points.append(np.array([(w/frame_width)*point[0], (h/frame_height)*point[1]]))

    top_left, top_right, bottom_right, bottom_left = out_points
    out_points = np.array(out_points)
    top_width = np.sqrt(np.dot(top_left - top_right,
                                    top_left - top_right))
    bottom_width = np.sqrt(np.dot(bottom_left - bottom_right,
                                bottom_left - bottom_right))
    left_height = np.sqrt(np.dot(top_left - bottom_left,
                                top_left - bottom_left))
    right_height = np.sqrt(np.dot(top_right - bottom_right,
                                top_right - bottom_right))
    max_w, max_h = int(max(top_width, bottom_width)), int(max(left_height,right_height))
    tg_loc = np.array([[0,0],[max_w, 0],[max_w,max_h],[0,max_h]])
    tg_loc = np.float32(tg_loc)
    M = cv2.getPerspectiveTransform(out_points.astype(np.float32), tg_loc)
    static_crop = cv2.warpPerspective(np.asarray(im),M,(max_w, max_h),flags=cv2.INTER_LINEAR)
    static_crop = Image.fromarray(np.uint8(static_crop)).convert('RGB')
    return static_crop

def crop(path, input_points):
    im = Image.open(path)
    w,h = im.size
    frame_height = frame_width*(h/w)
    top_left_x, top_left_y, bot_right_x, bot_right_y = [float('inf'),float('inf'),float('-inf'),float('-inf')]
    out_points = []
    for point in input_points:
        point = [(w/frame_width)*point[0], (h/frame_height)*point[1]]
        top_left_x = min(point[0],top_left_x)
        bot_right_x = max(point[0], bot_right_x)
        top_left_y = min(point[1], top_left_y)
        bot_right_y = max(point[1], bot_right_y)
        out_points.append(point)

    cropped_im = im.crop((top_left_x, top_left_y,bot_right_x, bot_right_y))
    return cropped_im

def export_result(image, out_path):
    image.save(out_path) 
    if st.session_state['task'] != 'None':
        #Crop/Warp
        image.save(os.path.join(OUT_DIR, st.session_state['task'], out_path))
    return 
    
def load_image(path):
    im = Image.open(path)
    w,h = im.size
    frame_height = frame_width*(h/w)
    im = im.resize((int(frame_width),int(frame_height)))
    return im

def format():
    st.session_state['points'] = []
    st.session_state['task'] = 'None'
    st.session_state['task_data'] = None
    st.experimental_rerun()
    return


init_state()
##Headers
st.header("Image Transformation :crystal_ball:")
st.text("")
st.markdown("You can perform in-plane image transformations like crop or warp using this tool")
st.markdown("Instructions: :pencil:")
st.markdown("1. If the image needs to be warped, select 4 coordinates on the image and click on **:red[Warp]** button  \n"
             "2. If the image needs to be cropped, select 4 coordinates on the image and click on **:red[Crop]** button  \n"
             "3. Select the **:blue[Reset]** button to clear the coordinate selection  \n"
            "4. Select the **:blue[Save]** button to save the transformation")
st.text("")

with st.container():
    col1, col2 = st.columns([1,4])
    with col1:
        st.write('Transform:')
    with col2:
        st.write('Input Image')

with st.container():
    col3, col4 = st.columns([1,4])
    with col3:
        needs_warp = st.button('Warp', type = 'primary')
        needs_crop = st.button('Crop', type = 'primary')
        clear_points = st.button('Reset', type = 'secondary')
        if clear_points:
            format()
        save = st.button('Save')

    im = load_image('{}.jpg'.format(st.session_state['idx']))
    draw = ImageDraw.Draw(im)
    for point in st.session_state['points']:
        coords = [point[0]-r, point[1] -r,
                    point[0]+r, point[1]+r]
        draw.ellipse(coords, fill = 'green')
    with col4:
        a,b,c = st.columns([1,1,1])
        with a:
            prev = st.button('<< Prev')
            if prev:
                st.session_state['idx'] -= 1
                format()
        with b:
            st.text("{}.jpg".format(st.session_state['idx']))
        with c:
            prev = st.button('Next >>')
            if prev:
                st.session_state['idx'] += 1
                format()    
        value = streamlit_image_coordinates(im, key = 'pil')

if value is not None:
    point = value['x'], value['y']
    if point not in st.session_state['points']:
        st.session_state['points'].append(point)
        st.experimental_rerun()


if needs_warp:
    if len(st.session_state['points'])==4:
        message = st.text('Warpped around selected 4 points of the image')
        with col3:
            st.session_state['task'] = 'warp'
            st.session_state['task_data'] = warp('{}.jpg'.format(st.session_state['idx']), st.session_state['points'])
            im_warp = st.image(st.session_state['task_data'])
    elif len(st.session_state['points']) < 4:
        st.session_state['task_data'] = None
        st.session_state['task'] = 'invalid'
        message = st.text('Needs 4 points for warping')
    else:
        st.session_state['task_data'] = None
        st.session_state['task'] = 'invalid'
        message = st.text('You have more than 4 points. Redo!!')

if needs_crop:
    if len(st.session_state['points'])==4:
        message = st.text('Cropped around selected 4 points of the image')
        st.session_state['task'] = 'crop'
        st.session_state['task_data']  = crop('{}.jpg'.format(st.session_state['idx']), st.session_state['points'])
        im_crop = st.image(st.session_state['task_data'])
    elif len(st.session_state['points']) < 4:
        st.session_state['task_data'] = None
        st.session_state['task'] = 'invalid'
        message = st.text('Needs 4 points for cropping')
    else:
        st.session_state['task_data'] = None
        st.session_state['task'] = 'invalid'
        message = st.text('You have more than 4 points. Redo!!')

if save:
    if st.session_state['task'] == 'invalid':
        message = st.text('File cannot be exported. No crop or warp found. Please process again.')
    elif st.session_state['task'] == 'None':
        export_result(Image.open('{}.jpg'.format(st.session_state['idx'])), '{}.jpg'.format(st.session_state['idx']))
        message = st.text('Exported the image successfully!  :)')
    else:
        export_result(st.session_state['task_data'], '{}.jpg'.format(st.session_state['idx']))
        message = st.text('Exported the {} successfully!  :)'.format(st.session_state['task']))
