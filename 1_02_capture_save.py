import cv2
import os
save_dir =  "/captured_img_test"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)

img_counter = 0

print("Press 's' to save the image or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key % 256 == ord('s'):
        # Press 's' to save the image
        img_name = f"image_{img_counter:04}.png"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_name}")
        img_counter += 1
    elif key % 256 == ord('q'):
        # Press 'q' to quit the program
        break

cap.release()
cv2.destroyAllWindows()