import cv2
import time

def capture_photos(num_photos, interval_sec, save_format='png'):
    # Open the camera
    cap = cv2.VideoCapture(0)  # Change to the appropriate camera index
    cap2 = cv2.VideoCapture(1)  # Change to the appropriate camera index
    if not cap.isOpened():
        print("Error: Camera1 not found")
        return
    
    if not cap2.isOpened():
        print("Error: Camera2 not found")
        return

    for i in range(num_photos):
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera 1 Could not capture frame")
            break
        ret2, frame2 = cap2.read()
        if not ret2:
            print("Error: Camera 2 Could not capture frame")
            break
        # Generate a filename
        filename1 = f"left/photo_{i + 1}.{save_format}"

        # Save the frame in the specified format
        cv2.imwrite(filename1, frame)
        # Generate a filename
        filename2 = f"right/photo_{i + 1}.{save_format}"

        # Save the frame in the specified format
        cv2.imwrite(filename2, frame)

        print(f"Captured {filename1}")
        print(f"Captured {filename2}")


        if i < num_photos - 1:
            time.sleep(interval_sec)

    # Release the camera
    cap.release()

# Parameters
num_photos_to_take = 50
time_interval_between_photos = 3  # in seconds
save_image_format = 'png'  # 'jpg' or 'png'

capture_photos(num_photos_to_take, time_interval_between_photos, save_image_format)