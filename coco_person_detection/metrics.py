'''
@Article{electronics10030279,
    AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
    TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
    JOURNAL = {Electronics},
    VOLUME = {10},
    YEAR = {2021},
    NUMBER = {3},
    ARTICLE-NUMBER = {279},
    URL = {https://www.mdpi.com/2079-9292/10/3/279},
    ISSN = {2079-9292},
    DOI = {10.3390/electronics10030279}
}
'''

class Metrics:
    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Metrics._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Metrics._getIntersectionArea(boxA, boxB)
        union = Metrics._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Metrics._getArea(boxA)
        area_B = Metrics._getArea(boxB)
        if interArea is None:
            interArea = Metrics._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
