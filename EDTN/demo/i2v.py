import cv2
import os

def get_files(path):
    list_files = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            for f in files:
                fullpath = os.path.join(root, f)
                list_files.append(fullpath)
    return list_files

fps = 20

image_files = get_files('/home/nas2_userF/dataset/anime_talk/video-preprocessing/vox/images/test/id10285#m-uILToQ9ss#001686#001943.mp4')

img = cv2.imread(image_files[0])
height, width, _ = img.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = f'/home/nas3_userM/minhopark/repos/repos4students/taewoongkang/AnimeTalkingHead/EDTN/example/vox_video/example.mp4'
writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

for file in image_files:
    img = cv2.imread(file)
    writer.write(img)

writer.release()
print(f"Both video saved as {output_file}.")
