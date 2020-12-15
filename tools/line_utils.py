import numpy as np
import requests

from person_vehicle_monitoring.config import CAMERA_LINE_URL


def get_line_config(equipment_id):
    data_json = {"cameraId": equipment_id}
    try:
        resp = requests.post(url=CAMERA_LINE_URL, json=data_json).json()
        resp = eval(resp['normalLine'])

    except Exception as e:
        return None
    return resp



def judge_empty_img(img):
    if isinstance(img, np.ndarray):
        if img.size > 0:
            return True
    return False


def is_ray_intersects_segment(poi, s_poi, e_poi):  # [x,y]
    # 输入：判断点，边起点，边终点，
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def is_poi_with_in_poly(poi, poly):
    # 输入：点，多边形二维数组
    sinsc = 0  # 交点个数
    # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    for i in range(len(poly) - 1):  # [0,len-1]
        s_poi = poly[i]
        e_poi = poly[i + 1]
        if is_ray_intersects_segment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1
    if sinsc % 2 == 1:
        return True
    else:
        return False


def count_res(p0, line_p1, line_p2):
    fl_vector = np.array([p0[0] - line_p2[0], p0[1] - line_p2[1]])
    line_vector = np.array([line_p1[0] - line_p2[0], line_p1[1] - line_p2[1]])
    return (int(np.cross(fl_vector, line_vector)))


def cross_res(p0, poly):
    rect_1 = poly[0]
    rect_2 = poly[3]
    rect_3 = poly[2]
    rect_4 = poly[1]
    n1 = count_res(p0, rect_2, rect_1)
    n2 = count_res(p0, rect_3, rect_2)
    n3 = count_res(p0, rect_4, rect_3)
    n4 = count_res(p0, rect_1, rect_4)
    if n1 < 0 and n2 < 0 and n3 < 0 and n4 < 0:
        return True
    else:
        return False
