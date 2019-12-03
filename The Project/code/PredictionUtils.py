import cv2


def trapezoidal(x, a, b, c, d):
    """
    :param x: the crisp input
    :param a: trapezoidal bottom left coordinate
    :param b: trapezoidal top left coordinate
    :param c: trapezoidal top right coordinate
    :param d: trapezoidal bottom right coordinate
    :return: the membership of the input
    """
    return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)


def compute_final_decision(baby, child, young, adult, senior):
    baby_ratio = 0.033

    child_ratio = 0.158

    young_ratio = 0.300

    adult_ratio = 0.525

    senior_ratio = 1.147

    fd = ((baby * baby_ratio) + (child * child_ratio) + (young * young_ratio) + (adult * adult_ratio) + (
            senior * senior_ratio)) / (baby + child + young + adult + senior)

    return fd


def fd_label(fd):
    baby  = trapezoidal(fd, 0.000, 0.033, 0.033, 0.053)
    child = trapezoidal(fd, 0.137, 0.158, 0.158, 0.263)
    young = trapezoidal(fd, 0.179, 0.300, 0.300, 0.378)
    adult = trapezoidal(fd, 0.371, 0.525, 0.525, 0.601)
    senior = trapezoidal(fd, 0.622, 1.147, 1000, 1000.01)

    print("Baby {}, Child {}, Young {}, Adult {}, Senior {}".format(baby, child, young, adult, senior))


def membership_baby(x):
    return trapezoidal(x, 0, 0.01, 2, 4)


def membership_child(x):
    return trapezoidal(x, 2, 6, 14, 17)


def membership_young(x):
    return trapezoidal(x, 15, 18, 28, 32)


def membership_adult(x):
    return trapezoidal(x, 28, 30, 40, 55)


def membership_senior(x):
    return trapezoidal(x, 45, 50, 120, 120.01)


img_class = [0.75, 0, 0, 0.25, 0]

adult = img_class[0]
baby = img_class[1]
child = img_class[2]
senior = img_class[3]
young = img_class[4]

fd = compute_final_decision(baby, child, young, adult, senior)

print("fd: ", fd)

fd_label(fd)

img = cv2.imread(r"D:\UNIVERCITY\مقدمة في الذكاء الاصطناعي\Project\Facial-based-age-estimator\The Project\code\12.jpg")

print("img", img)
