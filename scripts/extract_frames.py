import os
import cv2
import argparse

def extract_frames(video_path, output_path, matching_csv=None):
    print(f"Extracting frames from {video_path} into {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)
    frame_no = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame {}".format(frame_no))
            # truncate csv to match the number of frames
            if matching_csv:
                with open(matching_csv, 'r') as f:
                    lines = f.readlines()
                with open(matching_csv, 'w') as f:
                    f.writelines(lines[:frame_no + 1]) # +1 because of header line
            break

        cv2.imwrite(os.path.join(output_path, f"{frame_no}.png"), frame)
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video into specified path')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('output_path', help='Path to the output IMAGE directory')
    parser.add_argument('--matching_csv', help='Path to the matching csv file')

    args = parser.parse_args()
    if args.matching_csv:
        extract_frames(args.video_path, args.output_path, args.matching_csv)
    else:
        extract_frames(args.video_path, args.output_path)