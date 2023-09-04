import cv2
import time

def capture_photos(num_photos, interval_sec, save_format='png'):
    # Open the camera
    cap = cv2.VideoCapture(1)  # Change to the appropriate camera index

    if not cap.isOpened():
        print("Error: Camera not found")
        return

    for i in range(num_photos):
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            break

        # Generate a filename
        filename = f"photo_{i + 1}.{save_format}"

        # Save the frame in the specified format
        cv2.imwrite(filename, frame)

        print(f"Captured {filename}")

        if i < num_photos - 1:
            time.sleep(interval_sec)

    # Release the camera
    cap.release()

# Parameters
num_photos_to_take = 50
time_interval_between_photos = 3  # in seconds
save_image_format = 'png'  # 'jpg' or 'png'

capture_photos(num_photos_to_take, time_interval_between_photos, save_image_format)