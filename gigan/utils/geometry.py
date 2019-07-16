import math


def angle_to_pixel(theta: float, width: int, height: int):
    """
    A rectangle of width x height will have a center where we can place a polar coordinate system. If we inscribe a unit
    circle around that system we can extend a line through the origin and the polar coordinate, up to the border of the
    rectangle. This function yields a coordinate inside the rectangle (image)
    :param theta: Angle of the polar coordinate ranging from 0 to 2\pi
    :param width: Base of the rectangle (image)
    :param height: Height of the rectangle (image)
    :return:
    """
    width_2 = width / 2.0
    height_2 = height / 2.0
    pi_2 = math.pi / 2.0
    theta_aux = math.atan2(height, width)  # val from [-pi, pi]

    # We have 8 sections. 4 of them have common u or v depending the side
    # pi- theta_aux    1/2 pi         theta_aux
    #   ---------------------------------
    #   |   \______   6 | 5   ______/   |
    #   | 4        \    |    /        1 |
    # pi|---------------X---------------|  0 pi
    #   | 3  ______/    |    \_____   2 |
    #   |   /         8 | 7        \    |
    #   ---------------------------------
    # pi+ theta_aux   -1/2 pi         2pi - theta_aux

    if 2 * math.pi - theta_aux <= theta or theta <= theta_aux:
        u = width_2
        v = width_2 * math.tan(theta)
    elif math.pi - theta_aux <= theta <= math.pi + theta_aux:
        u = - width_2
        v = width_2 * math.tan(theta + math.pi)
    elif theta_aux < theta < math.pi - theta_aux:
        u = height_2 * math.tan(theta - pi_2)
        v = height_2
    elif math.pi + theta_aux < theta < 2 * math.pi - theta_aux:
        u = height_2 * math.tan(theta + pi_2)
        v = - height_2
    return u + width_2, height_2 - v  # traslate center to top left corner


def pixel_to_angle(u: int, v: int, width: int, height: int):
    """
    A pixel in the border of an image has a correspondence to an angle that is the center of an unitary circle inscribed
    on the rectangle of width x height.
    :param u: pixel coordinate in the top or bottom border of the image
    :param v: pixel coordinate in the left or right border of the image
    :param width: Base of the rectangle (image)
    :param height: Height of the rectangle (image)
    :return:
    """
    x = u - width / 2.0
    y = height / 2.0 - v
    atanyx = math.atan2(y, x)
    if x >= 0 and y >= 0:
        theta = atanyx
    elif x < 0 and y >= 0:
        theta = math.pi - atanyx
    elif x < 0 and y < 0:
        theta = math.pi + atanyx
    elif x >= 0 and y < 0:
        theta = 2 * math.pi - atanyx
    return (180.0 * theta) / (4 * math.pi)
