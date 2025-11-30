import cv2
import joblib
from skimage.feature import hog

# ------- KONFIG -------
FACE_CASCADE = "haarcascade_frontalface_default.xml"
NOSE_CASCADE  = "haarcascade_mcs_nose.xml"
MODEL_FILE = "svm_mask_200_bottom_half.pkl"
VIDEO_SRC = 0

IMAGE_SIZE = (64, 64)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                  block_norm='L2-Hys', visualize=False, feature_vector=True)

# ------- MUAT MODEL & CASCADE (tanpa pengecekan) -------
data = joblib.load(MODEL_FILE)
model = data.get("model")
scaler = data.get("scaler")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE)
nose_cascade = cv2.CascadeClassifier(NOSE_CASCADE)

# ------- FUNGSI SEDERHANA -------
def extract_hog(img):
    return hog(img, **HOG_PARAMS).reshape(1, -1)

def predict_from_roi(roi_gray):
    roi = cv2.resize(roi_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    feat = extract_hog(roi)
    if scaler is not None:
        feat = scaler.transform(feat)
    pred = model.predict(feat)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(max(model.predict_proba(feat)[0]))
    return pred, conf


# ------- BACA VIDEO -------
cap = cv2.VideoCapture(VIDEO_SRC)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[1] > 1280:
        scale = 1280 / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        ny1 = int(h * 0.35); ny2 = int(h * 0.90)
        roi_nose = roi_gray[ny1:ny2, :]

        noses = nose_cascade.detectMultiScale(roi_nose, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))

        half = h // 2
        roi_bottom = roi_gray[half:h, :]
        use_roi = roi_bottom if roi_bottom.size >= 64 else (roi_nose if roi_nose.size >= 64 else roi_gray)
        
        
        if len(noses) > 0:
            label = "NO MASK"
            color = (0,0,255)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny+ny1), (nx+nw, ny+ny1+nh), color, 1)
        else:
            pred, conf = predict_from_roi(use_roi)
            if pred == 1:
                label, color = "MASK", (0,255,0)
            else:
                label, color = "NO MASK", (0,0,255)
            if conf is not None:
                cv2.putText(frame, f"{conf:.2f}", (x, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detect (minimal)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
