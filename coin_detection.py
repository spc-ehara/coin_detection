import argparse
import math
import numpy as np
import statistics

import cv2


def binarize(src_img):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5, 5), 3)
    bin_img = cv2.adaptiveThreshold(
        gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2
    )
    return bin_img


def filter_object(bin_img, thresh_w, thresh_h, thresh_area):
    nlabels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img.astype(np.uint8)
    )
    obj_stats_idx = np.where(
        (stats[1:, cv2.CC_STAT_WIDTH] > thresh_w[0])
        & (stats[1:, cv2.CC_STAT_WIDTH] < thresh_w[1])
        & (stats[1:, cv2.CC_STAT_HEIGHT] > thresh_h[0])
        & (stats[1:, cv2.CC_STAT_HEIGHT] < thresh_h[1])
        & (stats[1:, cv2.CC_STAT_AREA] > thresh_area[0])
        & (stats[1:, cv2.CC_STAT_AREA] < thresh_area[1])
    )
    return np.where(np.isin(labels_img - 1, obj_stats_idx), 255, 0).astype(np.uint8)


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


def find_hole_contours(contours, hierarchy):
    hole_cnt = []
    for cnt_idx, cnt in enumerate(contours):
        for hier_idx, info in enumerate(hierarchy[0]):
            if info[3] != cnt_idx:
                continue
            hole_area = cv2.contourArea(contours[hier_idx])
            parent_area = cv2.contourArea(cnt)
            if hole_area < (parent_area * 0.03128) or hole_area > (
                parent_area * 0.05765
            ):
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contours[hier_idx])
            circle_area = int(radius * radius * np.pi)
            if circle_area <= 0:
                continue
            area_diff = circle_area / hole_area
            if 0.8 > area_diff or area_diff > 1.2:
                continue
            hole_cnt.append(contours[hier_idx])

    return hole_cnt


def _get_moments(cnt):
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _isinclude_cnt(cnt_1, cnt_2):
    cntM_2 = _get_moments(cnt_2)
    flag = cv2.pointPolygonTest(cnt_1, cntM_2, False)
    if flag >= 0:
        return True
    else:
        return False


def extract_feature(src_img, coin_contours, hole_contours):
    if src_img.shape[2] != 3:
        return
    bgr_img = cv2.split(src_img)
    coins_color = []
    hole_features = []
    coins_area = []
    for i, cnt in enumerate(coin_contours):
        blank_img = np.zeros_like(bgr_img[0])
        coin_img = cv2.drawContours(blank_img, [cnt], -1, 255, -1)
        hole_flag = False
        for hcnt in hole_contours:
            if _isinclude_cnt(cnt, hcnt):
                coin_img = cv2.drawContours(coin_img, [hcnt], -1, 0, -1)
                hole_flag = True

        coin_pixels = np.where(coin_img == 255)

        blue = []
        green = []
        red = []
        for p in zip(coin_pixels[0], coin_pixels[1]):
            blue.append(bgr_img[0][p[0]][p[1]])
            green.append(bgr_img[1][p[0]][p[1]])
            red.append(bgr_img[2][p[0]][p[1]])

        coins_color.append([blue, green, red])
        hole_features.append(hole_flag)
        coins_area.append(math.ceil(cv2.contourArea(cnt)))

    return (coins_color, hole_features, coins_area)


def determine_coin_type(coins_color, hole_features):
    coin_type = []
    for (cc, hf) in zip(coins_color, hole_features):
        b_ave = math.ceil(np.average(cc[0]))
        g_ave = math.ceil(np.average(cc[1]))
        r_ave = math.ceil(np.average(cc[2]))
        b_mode = math.ceil(statistics.mode(cc[0]))
        g_mode = math.ceil(statistics.mode(cc[1]))
        r_mode = math.ceil(statistics.mode(cc[2]))
        rb_ave_diff = r_ave - b_ave
        rg_ave_diff = r_ave - g_ave
        gb_ave_diff = g_ave - b_ave
        rb_mode_diff = r_mode - b_mode
        rg_mode_diff = r_mode - g_mode
        gb_mode_diff = g_mode - b_mode

        guess_type = 0
        if hf is True:
            if (b_ave / r_ave) < 0.6 and (b_ave / r_ave) > 0.4:
                guess_type = 5
            else:
                guess_type = 50
        else:
            if (b_ave / r_ave) < 0.6 and (b_ave / r_ave) > 0.4:
                guess_type = 10
            elif (rb_ave_diff + rg_ave_diff + gb_ave_diff) < 50:
                guess_type = 1
            elif (rb_mode_diff + rg_mode_diff + gb_mode_diff) < 120 or (
                rg_mode_diff - gb_mode_diff
            ) > 70:
                guess_type = 100
            else:
                guess_type = 500

        coin_type.append(guess_type)

    return coin_type


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

    bin_img = binarize(src_img)

    max_area = math.ceil((width * height) / 2)
    min_area = math.ceil((width * height) / 100)
    bin_img = filter_object(
        bin_img, (0, (width / 1.5)), (0, (height / 1.5)), (min_area, max_area)
    )
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=4,
    )

    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    coin_contours = find_circle_contours(contours, (min_area, max_area))
    hole_contours = find_hole_contours(contours, hierarchy)
    (coins_color, hole_features, coins_area) = extract_feature(
        src_img, coin_contours, hole_contours
    )
    coin_type = determine_coin_type(coins_color, hole_features)

    dst_img = src_img.copy()
    for cnt in hole_contours:
        cv2.drawContours(dst_img, [cnt], -1, (255, 0, 0), 2)
    for (cnt, type) in zip(coin_contours, coin_type):
        cv2.drawContours(dst_img, [cnt], -1, (0, 0, 255), 2)
        cv2.putText(
            dst_img,
            str(type),
            _get_moments(cnt),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
