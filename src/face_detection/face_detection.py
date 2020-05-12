import logging as log
import sys
import cv2 
import numpy as np
from imutils.video import FPS

from argparse import ArgumentParser

from inference import Network

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    return parser

def inference(args):
    # Initialize the Inference Engine
    infer_network = Network()

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, num_requests=0)

    # Get a Input blob shape
    _, _, in_h, in_w = infer_network.get_input_shape()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Get a output blob name
    _ = infer_network.get_output_name()

    # Handle the input stream
    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
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
        infer_network.exec_net(image, request_id=0)
        
        # Get the output of inference
        if infer_network.wait(request_id=0) == 0:
            result = infer_network.get_output(request_id=0)
            for box in result[0][0]: # Output shape is 1x1x200x7
                conf = box[2]
                if conf >= prob_threshold:
                    xmin = int(box[3] * fw)
                    ymin = int(box[4] * fh)
                    xmax = int(box[5] * fw)
                    ymax = int(box[6] * fh)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Break if escape key pressed
        if key_pressed == 27:
            break

        fps.update()
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()

    cv2.destroyAllWindows()

    fps.stop()

    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def main():
    # Set log to INFO
    log.basicConfig(level=log.INFO)

    # Grab command line args
    args = build_argparser().parse_args()

    inference(args)
if __name__ == "__main__":
    main()    