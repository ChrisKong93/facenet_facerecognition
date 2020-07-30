# 以下是最常用的读取视频流的方法

import cv2

# url = 'rtsp://admin:password@192.168.1.104:554/11'
# url = "udp://127.0.0.1:1234"  # 此处@后的ipv4 地址需要修改为自己的地址
url = 'rtsp://admin:admin@192.168.0.14:8554/live'

cap = cv2.VideoCapture(url)
# time.sleep(60)
ret, frame = cap.read()
print(ret)
while ret:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
