import numpy as np
import pandas as pd
from PIL import Image
import os, time, cv2, torch, argparse
from facenet_pytorch import MTCNN, InceptionResnetV1

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='./facedata/',
                    help='path to folder that contains face tensors')
args = parser.parse_args()

DATA_FOLDER = args.datapath

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')


def embed_img(path):
    img = Image.open(path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img).to(device)
    print(img_cropped.shape)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding

def embed_frame(frame):
    img = Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mode='RGB')
    # Detection
    img_cropped = mtcnn(img).to(device)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

def frame_to_img(frame):
    return Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mode='RGB')

def L2dist(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return np.dot(a-b, (a-b).T)

def tensor_to_BGR(img_cropped):
    detected_face = img_cropped.numpy()
    detected_face = np.swapaxes(detected_face, 0, 2)
    detected_face = np.swapaxes(detected_face, 0, 1)
    detected_face = (detected_face*128 + 127.5)/255
    return cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR)

def identity(faceTensor, dataset):
    for people in dataset:
        if L2dist(faceTensor, dataset[people])**2 <= 1.1:
            return people
    return "Unknown"

def load_data(path):
    data = dict()
    for faceTensor in os.listdir(path):
        data[faceTensor[:-3]] = torch.load(path + faceTensor)
    return data

# Input stream setup
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 5)

# Calls pretrained model instances
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

data = load_data(DATA_FOLDER)

FPS_LIMIT = 5
startTime = time.time()
# Frame cap
while(True):
    nowTime = time.time()
    if (nowTime - startTime) < 1/FPS_LIMIT:
        continue
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (ret == False):
        print("Cant capture any frame")
        continue
    img = frame_to_img(frame)
    # Detection (takes largest face)
    try:
        face = mtcnn(img)
    except:
        cv2.putText(frame, "No faces found", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # Move face rep. to GPU
        face = face.to(device)
        # Calculate embedding (unsqueeze to add batch dimension)
        embedding = resnet(face.unsqueeze(0))
        # Compare captured face to existing data to identify the person
        info = identity(embedding, data)
        cv2.putText(frame, info, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        '''
        Convert face tensor to BGR (preferred cv2 color mode) and display
        Minor impact on runtime resources
        cv2.imshow('detected', tensor_to_BGR(face))
        '''

    # Display the resulting frame
    cv2.imshow('frame', frame)
    startTime = time.time() # Reset time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
