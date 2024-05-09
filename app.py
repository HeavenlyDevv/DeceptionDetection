import cv2
import numpy as np
import skimage
from skimage import img_as_ubyte
from RetinaFace.RetinaFaceDetection import retina_face
import argparse
import os
from FaceDetection.utils import align_face
#import Emotion.lie_emotion_process as emotion
from EmotionRecognition.emotion import emotion_recognition
parser = argparse.ArgumentParser()
#Retina
parser.add_argument('--len_cut', default=30, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./RetinaFace/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
#Landmark
parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d', '3d'])
# Emotion
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',help= '0 is self-attention; 1 is self + relation-attention')
parser.add_argument('--preTrain_path', '-pret', default='./Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar', type=str, help='pre-training model path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

class InputData:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_width = 640
        self.frame_height = 480
        self.frame = None
        self.cap = None
        self.frame_number = 0
        self.num_frames = 0
        self.ubyte_frame = None

        # Retina Face attributes
        self.face_dots_rectangle = None
        self.face_rectangle = None
        self.output_points = None
        self.bbox = None
        self.userface = None

        self.emotion = "Neutral"

        self.frames_list = []

        self.initialize_frames()
        self.next_frame()

    def initialize_frames(self):
        """
        Initialize the frame properties.
        """
        ################ VIDEO TO FRAMES ################
        self.cap = cv2.VideoCapture(self.video_path)

        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def next_frame(self):
        """
        Get the next frame from the video.
        """
        ret, frame = self.cap.read()

        #################### FACE DETECTION ################
        frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
        self.frame = frame

        image = skimage.img_as_float(self.frame).astype(np.float32)
        frame = img_as_ubyte(image)

        self.face_dots_rectangle, self.face_rectangle, self.output_points,self.bbox,face_list = Retina.detect_face(frame) # face detection

        if len(self.bbox) >=2:
            print("WARNING: More than one face detected")
        if(len(self.output_points)):
            self.userface = face_list[0];

        #################### FACE ALIGNMENT ################
            self.face_cropped = align_face(self.face_rectangle, self.output_points[0], crop_size_h = 112, crop_size_w = 112)
            self.face_cropped = cv2.resize(self.face_cropped,(224, 224))

        #################### EMOTION RECOGNITION ################
            #self.emotion = emotion_recognition(self.face_cropped)


            cx = self.bbox[0][0]
            cy = self.bbox[0][1] + 50
            img = cv2.putText(self.frame, self.emotion, (cx, cy),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),2)
    
            # open face detection images and display them
            cv2.imshow('image', img)
            cv2.waitKey(1)
            #cv2.waitKey(0)

            #cv2.rectangle(self.face_rectangle, (self.bbox[0][0], self.bbox[0][1]), (self.bbox[0][2], self.bbox[0][3]), (0, 0, 255), 2)
            #cv2.imshow('image', self.face_rectangle)
            #cv2.waitKey(0)

            #print( "output_points",self.output_points[0])
            #print("bbox", self.bbox[0])
            #print("face_list", face_list[0])

            self.frames_list.append(self.face_cropped)
        
        self.frame_number += 1

    def end_frame_read(self):
        """
        Release the VideoCapture object and close all windows.
        """
        self.cap.release()

 
Retina = retina_face(crop_size = 224, args = args) # Face detection
#Emotion_class = emotion.Emotion_FAN(args = args)


# Example usage:
video_path = "./dataset-trials/Clips/Truthful/trial_truth_022.mp4"

input_data = InputData(video_path)

for i in range(input_data.num_frames-1):
    #print("Frame number:", input_data.frame_number)

    #if len(input_data.frames_list) == args.len_cut:
        #pred_score, self_embedding, relation_embedding = Emotion_class.validate(input_data.frames_list)

        #print(pred_score, self_embedding, relation_embedding)
    #print()


        #input_data.frames_list = []
    
    input_data.next_frame()


#while input_data.frame_number < input_data.num_frames:
#    print("Frame number:", input_data.frame_number)
    
#    input_data.next_frame()


input_data.end_frame_read()



