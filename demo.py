import argparse
import math
import numpy as np

import cv2
import coin_detection as cd


def parse_args() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_CAM", help="Input camera num or video path", type=str)
    parser.add_argument("-f", "--FPS", help="Input FPS", type=int, default=None)
    args = parser.parse_args()
    return (args.IN_CAM, args.FPS)


def main() -> None:
    (in_cam, in_fps) = parse_args()
    if in_cam.isdigit():
        in_cam = int(in_cam)
    cap = cv2.VideoCapture(in_cam)
    frame_widht = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)

    if frame_widht != 640 or frame_height != 480:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("width:{}".format(frame_widht))
    print("height:{}".format(frame_height))

    if in_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, in_fps)

    print("fps:{}".format(cap.get(cv2.CAP_PROP_FPS)))

    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        src_img = frame
        if src_img is None:
            return
        height, width = src_img.shape[:2]
        if (height != 480) or (width != 640):
            src_img = cv2.resize(src_img, (640, 480))
            height, width = src_img.shape[:2]

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(gray, (5, 5), 3)
        bin_img = cv2.adaptiveThreshold(
            gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2
        )

        max_area = math.ceil((width * height) / 2)
        min_area = math.ceil((width * height) / 100)
        bin_img = cd.filter_object(
            bin_img, (0, (width / 1.5)), (0, (height / 1.5)), (min_area, max_area)
        )
        bin_img = cv2.morphologyEx(
            bin_img,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=4,
        )

        contours, hierarchy = cv2.findContours(
            bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        coin_contours = cd.find_circle_contours(contours, (min_area, max_area))

        dst_img = src_img.copy()
        out_line = np.zeros_like(bin_img)
        for cnt in coin_contours:
            cv2.drawContours(dst_img, [cnt], -1, (0, 0, 255), 2)

            (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(out_line, (int(center_x), int(center_y)), int(radius), 255, 4)

        gray_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gaus_out = cv2.cvtColor(gaus, cv2.COLOR_GRAY2BGR)
        bin_img_out = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        out_line_out = cv2.cvtColor(out_line, cv2.COLOR_GRAY2BGR)

        src_img = cv2.putText(
            src_img,
            "src_img",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        gray_out = cv2.putText(
            gray_out,
            "gray",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        gaus_out = cv2.putText(
            gaus_out,
            "gaus",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        bin_img_out = cv2.putText(
            bin_img_out,
            "bin_img",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        out_line_out = cv2.putText(
            out_line_out,
            "out_line",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        dst_img = cv2.putText(
            dst_img,
            "dst_img",
            (250, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        out_img = cd.concat_tile(
            [
                [src_img, gray_out, bin_img_out],
                [dst_img, gaus_out, out_line_out],
            ]
        )
        out_img = cv2.resize(out_img, (1920, 1080))
        frame = out_img
        cv2.imshow("output", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
