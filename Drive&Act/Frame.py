import cv2 as cv

cap = cv.VideoCapture('../Dataset/Center mirror/vp1/run1b_2018-05-29-14-02-47.ids_1-5mincut.mp4')
i = 0

while(cap.isOpened()):
	ret, frame = cap.read()
	
	# This condition prevents from infinite looping incase video ends.
	if ret == False:
		break
	
	# Save Frame by Frame into disk using imwrite method
	cv.imwrite(str(i)+'.jpg', frame)
	i += 1

cap.release()
cv.destroyAllWindows()