import cv2
import numpy as np
def lucas_kanade_method(video_path):
    method = 'lucas_kenade'
    cap = cv2.VideoCapture(video_path)
    
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    f = 0
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
        width  = cap.get(3) // 2 # float `width`
        height = cap.get(4) // 2
        # Using resizeWindow() 
        cv2.resizeWindow("frame", (int(width), int(height))) 
        cv2.imshow("frame", img)
        f = f+1
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        if k == ord("s"):
            cv2.imwrite(f'{method}_{f}.png',img)
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()
    width  = cap.get(3) // 2 # float `width`
    height = cap.get(4) // 2
    old_frame = cv2.resize(old_frame, (int(width), int(height)))


    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
    m = 'rlof'
    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    f = 0
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        new_frame = cv2.resize(new_frame, (int(width), int(height)))
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.namedWindow("optical flow", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
        
        # Using resizeWindow() 
        cv2.resizeWindow("optical flow", (int(width), int(height))) 
        cv2.resizeWindow("frame", (int(width), int(height))) 

        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        f = f +1
        if k == 27:
            break
        if k == ord("s"):
            cv2.imwrite(f'{m}_{f}.png',bgr)
            cv2.imwrite(f'{m}_{f}_s.png',frame_copy)
        old_frame = new_frame

video_path = '/home/andrea/MCC/Repos/Computer Vision/video_test.mp4'

#----------LUCAS KANADE METHOD-----------
lucas_kanade_method(video_path)
#----------DENSE LUCAS KANADE METHOD-----------
#method = cv2.optflow.calcOpticalFlowSparseToDense
#dense_optical_flow(method, video_path, to_gray=True)
#----------FARNEBACK METHOD-----------
#method = cv2.calcOpticalFlowFarneback
#params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
#dense_optical_flow(method, video_path, params, to_gray=True)
#----------ROBUST LOCAL OPTICAL FLOW METHOD-----------
#method = cv2.optflow.calcOpticalFlowDenseRLOF
#dense_optical_flow(method, video_path)