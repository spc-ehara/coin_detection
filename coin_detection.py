import argparse
import math
import numpy as np

import cv2


def find_circle_contours(contours, thresh_area):
    new_cnt = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if thresh_area[0] > area or area > thresh_area[1]:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = int(radius * radius * np.pi)
        if circle_area <= 0:
            continue
        area_diff = circle_area / area
        if 0.9 > area_diff or area_diff > 1.1:
            continue
        new_cnt.append(cnt)

    return new_cnt


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def parse_args() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_IMG", help="Input file")
    parser.add_argument("OUT_IMG", help="Output file")
    args = parser.parse_args()

    return (args.IN_IMG, args.OUT_IMG)


def main() -> None:
    (in_img, out_img) = parse_args()
    src_img = cv2.imread(in_img)
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

    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    max_area = math.ceil((width * height) / 2)
    min_area = math.ceil((width * height) / 100)
    coin_contours = find_circle_contours(contours, (min_area, max_area))

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

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    out_img = concat_tile(
        [
            [src_img, gray_out, bin_img_out],
            [dst_img, gaus_out, out_line_out],
        ]
    )
    out_img = cv2.resize(out_img, (1920, 1080))
    cv2.imshow("output", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
