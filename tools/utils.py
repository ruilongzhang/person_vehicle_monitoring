import threading
import time
from functools import wraps


def cut_img(img, structure):
    x1, y1, x2, y2 = structure[2]
    if x1 <0:
        x1 = 0
    if y1 <0:
        y1 = 0
    short_box = img[int(y1):int(y2), int(x1):int(x2)]
    return short_box


def cut_image(image, shortcutbox):
    x, y, w, h = shortcutbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    xmin = 0 if xmin < 0 else xmin
    ymin = 0 if ymin < 0 else ymin
    img_s = image[ymin:ymax, xmin:xmax]
    return img_s


def retry(max_retries=3, exceptions=(), time_to_sleep=0, save_result=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            _max_retries = max_retries
            while _max_retries > 0:
                try:
                    wrapper.chances_left = _max_retries - 1
                    f = func(*args, **kwargs)
                    return f
                except exceptions as e:
                    _max_retries -= 1
                    if _max_retries == 0:
                        if save_result is not None:
                            return None
                        raise e
                    time.sleep(time_to_sleep)
            return None

        return wrapper

    return decorator
