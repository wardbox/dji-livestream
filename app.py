import ultralytics
import cv2
from flask import Flask, render_template, Response, redirect, url_for
from djitellopy import Tello
from threading import Thread
from ultralytics import YOLO


app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

ultralytics.checks()
model = YOLO("yolov8x.pt")

tello = Tello()
tello.connect()

print(f"Battery: {tello.get_battery()}%")

tello.streamon()

tello.set_speed(10)


@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    height, width, _ = tello.get_frame_read().frame.shape
    while True:
        try:
            frame = tello.get_frame_read().frame
            frame = cv2.resize(frame, (width // 2, height // 2))
            results = model.predict(frame, conf=0.25, max_det=15, verbose=False)
            annotated_frame = results[0].plot()
            _, jpeg = cv2.imencode(".jpg", annotated_frame)
            frame = jpeg.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )
        except:
            pass


recorder = Thread(target=generate_frames)
recorder.start()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run()

recorder.join()
