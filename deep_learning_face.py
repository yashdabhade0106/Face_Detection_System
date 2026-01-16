import cv2

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

img = cv2.imread("test.jpg")
h, w = img.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(img, (300, 300)), 
    1.0, (300, 300), 
    (104.0, 177.0, 123.0)
)

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("DL Face Detection", img)
cv2.waitKey(0)
