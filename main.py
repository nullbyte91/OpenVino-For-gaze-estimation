import logging as log 
import cv2 
from imutils.video import FPS
import numpy as np
from collections import namedtuple

from argparse import ArgumentParser

from openvino.inference_engine import IENetwork, IECore
from src import faceDetector, headPos_Estimator, landmark_Eestimator, gaze_Estimator

CPU_DEVICE_NAME = "CPU"

FaceInferenceResults = namedtuple('Point', 'x y')

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-m_g", "--model_gaze", required=True, type=str,
                        help="Path to an .xml file with a trained Gaze Estimation model")
    parser.add_argument("-m_fd", "--mode_face_detection", required=True, type=str,
                        help="Path to an .xml file with a trained Face Detection model")
    parser.add_argument("-m_hp", "--model_head_position", required=True, type=str,
                        help="Path to an .xml file with a trained Head Pose Estimation model")                   
    parser.add_argument("-m_lm", "--model_landmark_estimation", required=True, type=str,
                        help="Path to an .xml file with a trained Facial Landmarks Estimation model")                   
    return parser
def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def main():
    # Set log to INFO
    log.basicConfig(level=log.INFO)

    # Grab command line args
    args = build_argparser().parse_args()

    # Handle the input stream
    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    # Initialize the plugin
    ie = IECore()

    # Face Detection init
    face_detection = faceDetector.FaceDetector()
    face_detection.load_model(ie, args.mode_face_detection, "CPU", num_requests=0)

    # Head Position init
    head_position  = headPos_Estimator.HeadPosEstimator()
    head_position.load_model(ie, args.model_head_position, "CPU", num_requests=0)

    # Land Mark Estimator init
    landmark_estimator = landmark_Eestimator.LandMarkEestimator()
    landmark_estimator.load_model(ie, args.model_landmark_estimation, "CPU", num_requests=0)

    # Gaze init
    gaze_estimator = gaze_Estimator.GazeEestimator()
    gaze_estimator.load_model(ie, args.model_landmark_estimation, "CPU", num_requests=0)

    # Get a Input blob shape of face detection
    _, _, in_h, in_w = face_detection.get_input_shape()

    fps = FPS().start()

    while cap.isOpened():
        #Read the next frame
        _, frame = cap.read()
        if frame is None:
            break

        fh = frame.shape[0]
        fw = frame.shape[1]
        key_pressed = cv2.waitKey(50)
    
        image_resize = cv2.resize(frame, (in_w, in_h), interpolation = cv2.INTER_AREA)
        image = np.moveaxis(image_resize, -1, 0)

        # Perform inference on the frame
        face_detection.exec_net(image, request_id=0)

        # Variable that shared across between model
        faceBoundingBox = []
        headPoseAngles = {
        "x": 0,
        "y": 0,
        "z": 0
        }

        # Get the output of inference
        if face_detection.wait(request_id=0) == 0:
            # Get Face detection
            detection = face_detection.get_output(request_id=0)
            for i in range(0, detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                # If confidence > 0.5, save it as a separate file
                if (confidence > 0.5):
                    faceBoundingBox = detection[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])
                    (startX, startY, endX, endY) = faceBoundingBox.astype("int")
                    image_fc = frame[startY:endY, startX:endX]

                    #Head position
                    # Get a Input blob shape of face detection
                    in_n, in_c, in_h, in_w = head_position.get_input_shape()
                    print("in_n:{} in_c:{} in_h:{} in_w:{}".format(in_n, in_c, in_h, in_w))
                    image_h = cv2.resize(image_fc, (in_w, in_h), interpolation = cv2.INTER_AREA)
                    image_h = np.moveaxis(image_h, -1, 0)
                    print(image_h.shape)
                    head_position.exec_net(image_h, request_id=0)
                    if head_position.wait(request_id=0) == 0:
                        head_positions = head_position.get_output(request_id=0)
                        headPoseAngles['x'] = head_positions["angle_y_fc"][0]
                        headPoseAngles['y'] = head_positions["angle_p_fc"][0]
                        headPoseAngles['z'] = head_positions["angle_r_fc"][0]
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        #Break if escape key pressed
        if key_pressed == 27:
            break
        
        fps.update()

    # Release capture node
    cap.release()

    fps.stop()
if __name__ == "__main__":
    main()