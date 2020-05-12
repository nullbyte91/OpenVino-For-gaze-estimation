import logging as log 
import cv2 
from imutils.video import FPS

from argparse import ArgumentParser

from openvino.inference_engine import IENetwork, IECore
from src import faceDetector, headPos_Estimator, landmark_Eestimator, gaze_Estimator

CPU_DEVICE_NAME = "CPU"


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

    fps = FPS().start()

    while cap.isOpened():
        #Read the next frame
        _, frame = cap.read()
        if frame is None:
            break

        fps.update()

    # Release capture node
    cap.release()

    fps.stop()
if __name__ == "__main__":
    main()