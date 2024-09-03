import cv2
import os
import glob

def create_video_from_images(image_folder, output_video_file, frame_rate=30):
    # 이미지 파일을 가져옵니다.
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))  # 파일이 .jpg인 경우

    if not images:
        print("No images found in the folder.")
        return
    
    # 첫 번째 이미지에서 프레임의 크기를 가져옵니다.
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)
    
    # 비디오 작성자를 초기화합니다.
    out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)
    
    for image in images:
        img = cv2.imread(image)
        out.write(img)
    
    out.release()
    print(f"Video saved as {output_video_file}")

# 경로 설정
image_folder = '/home/hscho/workspace/ssd2.5d/MLPD-Multi-Label-Pedestrian-Detection/src/result(test)_visualization_R'
output_video_file = 'output_video.mp4'

# 비디오 생성 함수 호출
create_video_from_images(image_folder, output_video_file)
