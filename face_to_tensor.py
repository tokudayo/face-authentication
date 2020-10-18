import numpy as np
import pandas as pd
import cv2, os, argparse, torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', required=True,
                    help='name of the person')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Calls pretrained model instances
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def embed_img(path):
    img = Image.open(path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img).to(device)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding

def embed_frame(frame):
    img = Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mode='RGB')
    # Detection
    img_cropped = mtcnn(img).to(device)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

def L2dist(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return np.dot(a-b, (a-b).T)

def frame_to_img(frame):
    return Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mode='RGB')

def embed_capture(cap):
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (ret == False):
            print("Cant capture any frame")
            continue
        
        cv2.putText(frame, "Press space to capture", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img = frame_to_img(frame)
            # Detection (takes largest face)
            try:
                face = mtcnn(img)
            except:
                print("No faces found")
            else:
                # Move face rep. to GPU
                face = face.to(device)
                # Calculate embedding (unsqueeze to add batch dimension)
                embedding = resnet(face.unsqueeze(0))
                return embedding
                

def _main():
    embedTensor = torch.Tensor()
    if args.image:
        embedTensor = embed_img(args.image)
    else:
        cap = cv2.VideoCapture(args.src + cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        embedTensor = embed_capture(cap)
        cap.release()
        cv2.destroyAllWindows()
    if embedTensor != None:
        torch.save(embedTensor, args.name + ".pt")
        print(f"Saved face tensor to {args.name}.pt")
        

if __name__ == '__main__':
    _main()