"""
Ported from Paper.js - The Swiss Army Knife of Vector Graphics Scripting.
http://paperjs.org/

Copyright (c) 2011 - 2014, Juerg Lehni & Jonathan Puckey
http://scratchdisk.com/ & http://jonathanpuckey.com/

Distributed under the MIT license. See LICENSE file for details.
All rights reserved.

An Algorithm for Automatically Fitting Digitized Curves
by Philip J. Schneider
from "Graphics Gems", Academic Press, 1990
Modifications and optimisations of original algorithm by Juerg Lehni.

Ported by Gumble, 2015.
"""

import math
import svgwrite

TOLERANCE = 10e-6
EPSILON = 10e-12


class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y=None):
        if y is None:
            self.x, self.y = x[0], x[1]
        else:
            self.x, self.y = x, y

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)

    def __str__(self):
        return '%G,%G' % (self.x, self.y)

    def __complex__(self):
        return complex(self.x, self.y)

    def __hash__(self):
        return hash(self.__complex__())

    def __bool__(self):
        return bool(self.x or self.y)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y)
        else:
            return Point(self.x / other, self.y / other)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __len__(self):
        return math.hypot(self.x, self.y)

    def __eq__(self, other):
        try:
            return self.x == other.x and self.y == other.y
        except Exception:
            return False

    def __ne__(self, other):
        try:
            return self.x != other.x or self.y != other.y
        except Exception:
            return True

    add = __add__
    subtract = __sub__
    multiply = __mul__
    divide = __truediv__
    negate = __neg__
    getLength = __len__
    equals = __eq__

    def copy(self):
        return Point(self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def normalize(self, length=1):
        current = self.__len__()
        scale = length / current if current != 0 else 0
        return Point(self.x * scale, self.y * scale)

    def getDistance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class Segment:

    def __init__(self, *args):
        self.point = Point(0, 0)
        self.handleIn = Point(0, 0)
        self.handleOut = Point(0, 0)
        if len(args) == 1:
            if isinstance(args[0], Segment):
                self.point = args[0].point
                self.handleIn = args[0].handleIn
                self.handleOut = args[0].handleOut
            else:
                self.point = args[0]
        elif len(args) == 2 and isinstance(args[0], (int, float)):
            self.point = Point(*args)
        elif len(args) == 2:
            self.point = args[0]
            self.handleIn = args[1]
        elif len(args) == 3:
            self.point = args[0]
            self.handleIn = args[1]
            self.handleOut = args[2]
        else:
            self.point = Point(args[0], args[1])
            self.handleIn = Point(args[2], args[3])
            self.handleOut = Point(args[4], args[5])

    def __repr__(self):
        return 'Segment(%r, %r, %r)' % (self.point, self.handleIn, self.handleOut)

    def __hash__(self):
        return hash((self.point, self.handleIn, self.handleOut))

    def __bool__(self):
        return bool(self.point or self.handleIn or self.handleOut)

    def getPoint(self):
        return self.point

    def setPoint(self, other):
        self.point = other

    def getHandleIn(self):
        return self.handleIn

    def setHandleIn(self, other):
        self.handleIn = other

    def getHandleOut(self):
        return self.handleOut

    def setHandleOut(self, other):
        self.handleOut = other


class PathFitter:

    def __init__(self, segments, error=2.5):
        self.points = []
        # Copy over points from path and filter out adjacent duplicates.
        l = len(segments)
        prev = None
        for i in range(l):
            point = segments[i].point.copy()
            if prev != point:
                self.points.append(point)
                prev = point
        self.error = error

    def fit(self):
        points = self.points
        length = len(points)
        self.segments = [Segment(points[0])] if length > 0 else []
        if length > 1:
            self.fitCubic(0, length - 1,
                          # Left Tangent
                          points[1].subtract(points[0]).normalize(),
                          # Right Tangent
                          points[length - 2].subtract(points[length - 1]).normalize())
        return self.segments

    # Fit a Bezier curve to a (sub)set of digitized points
    def fitCubic(self, first, last, tan1, tan2):
        #  Use heuristic if region only has two points in it
        if last - first == 1:
            pt1 = self.points[first]
            pt2 = self.points[last]
            dist = pt1.getDistance(pt2) / 3
            self.addCurve([pt1, pt1 + tan1.normalize(dist),
                           pt2 + tan2.normalize(dist), pt2])
            return
        # Parameterize points, and attempt to fit curve
        uPrime = self.chordLengthParameterize(first, last)
        maxError = max(self.error, self.error * self.error)
        # Try 4 iterations
        for i in range(5):
            curve = self.generateBezier(first, last, uPrime, tan1, tan2)
            #  Find max deviation of points to fitted curve
            maxerr, maxind = self.findMaxError(first, last, curve, uPrime)
            if maxerr < self.error:
                self.addCurve(curve)
                return
            split = maxind
            # If error not too large, try reparameterization and iteration
            if maxerr >= maxError:
                break
            self.reparameterize(first, last, uPrime, curve)
            maxError = maxerr
        # Fitting failed -- split at max error point and fit recursively
        V1 = self.points[split - 1].subtract(self.points[split])
        V2 = self.points[split] - self.points[split + 1]
        tanCenter = V1.add(V2).divide(2).normalize()
        self.fitCubic(first, split, tan1, tanCenter)
        self.fitCubic(split, last, tanCenter.negate(), tan2)

    def addCurve(self, curve):
        prev = self.segments[len(self.segments) - 1]
        prev.setHandleOut(curve[1].subtract(curve[0]))
        self.segments.append(
            Segment(curve[3], curve[2].subtract(curve[3])))

    # Use least-squares method to find Bezier control points for region.
    def generateBezier(self, first, last, uPrime, tan1, tan2):
        epsilon = 1e-11
        pt1 = self.points[first]
        pt2 = self.points[last]
        # Create the C and X matrices
        C = [[0, 0], [0, 0]]
        X = [0, 0]

        l = last - first + 1

        for i in range(l):
            u = uPrime[i]
            t = 1 - u
            b = 3 * u * t
            b0 = t * t * t
            b1 = b * t
            b2 = b * u
            b3 = u * u * u
            a1 = tan1.normalize(b1)
            a2 = tan2.normalize(b2)
            tmp = (self.points[first + i]
                   - pt1.multiply(b0 + b1)
                   - pt2.multiply(b2 + b3))
            C[0][0] += a1.dot(a1)
            C[0][1] += a1.dot(a2)
            # C[1][0] += a1.dot(a2)
            C[1][0] = C[0][1]
            C[1][1] += a2.dot(a2)
            X[0] += a1.dot(tmp)
            X[1] += a2.dot(tmp)

        # Compute the determinants of C and X
        detC0C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
        if abs(detC0C1) > epsilon:
            # Kramer's rule
            detC0X = C[0][0] * X[1] - C[1][0] * X[0]
            detXC1 = X[0] * C[1][1] - X[1] * C[0][1]
            # Derive alpha values
            alpha1 = detXC1 / detC0C1
            alpha2 = detC0X / detC0C1
        else:
            # Matrix is under-determined, try assuming alpha1 == alpha2
            c0 = C[0][0] + C[0][1]
            c1 = C[1][0] + C[1][1]
            if abs(c0) > epsilon:
                alpha1 = alpha2 = X[0] / c0
            elif abs(c1) > epsilon:
                alpha1 = alpha2 = X[1] / c1
            else:
                # Handle below
                alpha1 = alpha2 = 0

        # If alpha negative, use the Wu/Barsky heuristic (see text)
        # (if alpha is 0, you get coincident control points that lead to
        # divide by zero in any subsequent NewtonRaphsonRootFind() call.
        segLength = pt2.getDistance(pt1)
        epsilon *= segLength
        if alpha1 < epsilon or alpha2 < epsilon:
            # fall back on standard (probably inaccurate) formula,
            # and subdivide further if needed.
            alpha1 = alpha2 = segLength / 3

        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        return [pt1, pt1.add(tan1.normalize(alpha1)),
                pt2.add(tan2.normalize(alpha2)), pt2]

    # Given set of points and their parameterization, try to find
    # a better parameterization.
    def reparameterize(self, first, last, u, curve):
        for i in range(first, last + 1):
            u[i - first] = self.findRoot(curve, self.points[i], u[i - first])

    # Use Newton-Raphson iteration to find better root.
    def findRoot(self, curve, point, u):
        # Generate control vertices for Q'
        curve1 = [
            curve[i + 1].subtract(curve[i]).multiply(3) for i in range(3)]
        # Generate control vertices for Q''
        curve2 = [
            curve1[i + 1].subtract(curve1[i]).multiply(2) for i in range(2)]
        # Compute Q(u), Q'(u) and Q''(u)
        pt = self.evaluate(3, curve, u)
        pt1 = self.evaluate(2, curve1, u)
        pt2 = self.evaluate(1, curve2, u)
        diff = pt - point
        df = pt1.dot(pt1) + diff.dot(pt2)
        # Compute f(u) / f'(u)
        if abs(df) < TOLERANCE:
            return u
        # u = u - f(u) / f'(u)
        return u - diff.dot(pt1) / df

    # Evaluate a bezier curve at a particular parameter value
    def evaluate(self, degree, curve, t):
        # Copy array
        tmp = curve[:]
        # Triangle computation
        for i in range(1, degree + 1):
            for j in range(degree - i + 1):
                tmp[j] = tmp[j].multiply(1 - t) + tmp[j + 1].multiply(t)
        return tmp[0]

    # Assign parameter values to digitized points
    # using relative distances between points.
    def chordLengthParameterize(self, first, last):
        u = {0: 0}
        print(first, last)
        for i in range(first + 1, last + 1):
            u[i - first] = u[i - first - 1] + \
                self.points[i].getDistance(self.points[i - 1])
        m = last - first
        for i in range(1, m + 1):
            u[i] /= u[m]
        return u

    # Find the maximum squared distance of digitized points to fitted curve.
    def findMaxError(self, first, last, curve, u):
        index = math.floor((last - first + 1) / 2)
        maxDist = 0
        for i in range(first + 1, last):
            P = self.evaluate(3, curve, u[i - first])
            v = P.subtract(self.points[i])
            dist = v.x * v.x + v.y * v.y  # squared
            if dist >= maxDist:
                maxDist = dist
                index = i
        return maxDist, index


def fitpath(pointlist, error=50):
    return PathFitter(list(map(Segment, map(Point, pointlist))), error).fit()


def fitpathsvg(pointlist, error=2.5):
    return pathtosvg(PathFitter(list(map(Segment, map(Point, pointlist))), error).fit())


def pathtosvg(path):
    segs = ['M', str(path[0].point)]
    last = path[0]
    for seg in path[1:]:
        segs.append('C')
        segs.append(str(last.point + last.handleOut))
        segs.append(str(seg.point + seg.handleIn))
        segs.append(str(seg.point))
        last = seg
    return ' '.join(segs)

def drawsvg(commands):
  dwg = svgwrite.Drawing('test.svg', size=(1024,1024))
  for command in commands:
    dwg.add(dwg.path(d=command,
      stroke="#000",
      fill="none",
      stroke_width=5))
  dwg.save()


if __name__ == '__main__':
    pts = [[(474, 259), (468, 258), (467, 257), (463, 257), (462, 256), (423, 256), (422, 257), (407, 257), (406, 258), (400, 258), (399, 259), (383, 259), (382, 260), (372, 261), (371, 262), (362, 262), (360, 261), (359, 262), (355, 262), (354, 263), (349, 263), (348, 264), (335, 264), (334, 265), (329, 265), (328, 266), (313, 267), (312, 268), (303, 269), (302, 270), (295, 271), (294, 272), (290, 273), (284, 279), (277, 283), (275, 285), (274, 285), (267, 292), (266, 294), (266, 296), (261, 306), (259, 314), (255, 322), (254, 331), (253, 332), (253, 337), (252, 338), (252, 354), (251, 355), (250, 375), (249, 376), (249, 414), (250, 415), (250, 427), (251, 428), (251, 449), (252, 450), (252, 471), (253, 472), (253, 485), (254, 486), (254, 492), (256, 496), (256, 498), (261, 508), (261, 511), (262, 512), (262, 517), (263, 518), (263, 525), (264, 526), (264, 533), (265, 534), (265, 538), (266, 539), (266, 541), (267, 542), (268, 546), (270, 548), (272, 552), (273, 559), (274, 560), (274, 565), (275, 566), (275, 570), (276, 571), (276, 574), (277, 575), (278, 581), (283, 588), (285, 592), (285, 594), (286, 595), (286, 597), (287, 598), (287, 600), (290, 606), (293, 609), (297, 617), (297, 619), (299, 622), (299, 624), (302, 630), (304, 632), (305, 632), (321, 648), (322, 648), (326, 651), (328, 651), (331, 653), (335, 654), (345, 661), (349, 662), (350, 663), (358, 664), (359, 665), (368, 665), (369, 666), (394, 666), (395, 665), (401, 665), (402, 664), (418, 663), (432, 656), (440, 654), (443, 652), (445, 652), (455, 647), (460, 642), (461, 642), (464, 639), (465, 639), (470, 634), (471, 634), (476, 629), (478, 625), (487, 615), (489, 611), (491, 609), (492, 606), (498, 598), (499, 595), (511, 578), (517, 566), (517, 564), (520, 558), (521, 551), (522, 550), (522, 546), (523, 545), (523, 541), (524, 540), (524, 536), (525, 535), (525, 524), (526, 523), (526, 519), (527, 518), (528, 512), (531, 506), (531, 500), (532, 499), (532, 489), (533, 488), (557, 489), (558, 490), (567, 490), (568, 489), (571, 489), (575, 485), (576, 485), (577, 481), (578, 480), (578, 474), (579, 473), (579, 460), (580, 459), (580, 454), (581, 453), (582, 440), (583, 439), (583, 422), (584, 421), (584, 408), (583, 406), (584, 405), (584, 401), (583, 400), (583, 393), (581, 389), (581, 387), (579, 383), (570, 374), (568, 373), (563, 373), (559, 371), (557, 371), (553, 374), (520, 382), (519, 377), (516, 371), (515, 364), (514, 363), (514, 359), (513, 358), (513, 353), (512, 352), (512, 344), (511, 343), (511, 335), (510, 334), (510, 325), (509, 324), (509, 319), (507, 315), (507, 312), (505, 308), (505, 306), (504, 305), (504, 303), (502, 300), (500, 292), (497, 286), (492, 281), (488, 274), (486, 272), (486, 271)], [(223, 358), (222, 359), (222, 360), (223, 361), (223, 362), (223, 363), (223, 364), (223, 365), (223, 366), (223, 367), (223, 368), (224, 369), (224, 370), (224, 371), (224, 372), (224, 373), (224, 374), (225, 375), (225, 376), (225, 377), (225, 378), (225, 379), (225, 380), (225, 381), (226, 382), (226, 383), (226, 384), (226, 385), (226, 386), (226, 387), (227, 388), (227, 389), (227, 390), (227, 391), (227, 392), (228, 393), (228, 394), (228, 395), (228, 396), (228, 397), (228, 398), (229, 399), (229, 400), (229, 401), (229, 402), (229, 403), (230, 404), (230, 405), (230, 406), (230, 407), (230, 408), (231, 409), (231, 410), (231, 411), (231, 412), (232, 413), (232, 414), (232, 415), (232, 416), (232, 417), (233, 418), (233, 419), (233, 420), (233, 421), (233, 422), (234, 423), (234, 424), (234, 425), (234, 426), (235, 427), (235, 428), (235, 429), (235, 430), (236, 431), (236, 432), (236, 433), (236, 434), (237, 435), (237, 436), (237, 437), (237, 438), (238, 439), (238, 440), (238, 441), (238, 442), (239, 443), (239, 444), (239, 445), (239, 446), (240, 447), (240, 448), (240, 449), (241, 450), (241, 451), (241, 452), (241, 453), (242, 454), (242, 455), (242, 456), (243, 457), (243, 458), (243, 459), (243, 460), (244, 461), (244, 462), (244, 463), (245, 464), (245, 465), (245, 466), (245, 467), (246, 468), (246, 469), (246, 470), (247, 471), (247, 472), (247, 473), (248, 474), (248, 475), (248, 476), (249, 477), (249, 478), (249, 479), (250, 480), (250, 481), (250, 482), (251, 483), (251, 484), (251, 485), (252, 486), (252, 487), (252, 488), (253, 489), (253, 490), (253, 491), (223, 354), (222, 353), (222, 349), (221, 348), (221, 343), (220, 342), (220, 330), (219, 329), (219, 304), (218, 303), (218, 280), (217, 279), (217, 271), (216, 270), (216, 249), (215, 248), (215, 244), (214, 243), (214, 240), (213, 239), (213, 231), (212, 230), (212, 213), (211, 212), (211, 209), (212, 208), (212, 202), (213, 201), (213, 195), (214, 194), (214, 191), (215, 190), (215, 187), (216, 186), (216, 183), (217, 182), (217, 178), (218, 177), (218, 174), (219, 173), (219, 170), (220, 169), (220, 165), (227, 158), (228, 158), (231, 155), (231, 154), (234, 151), (234, 150), (235, 149), (235, 148), (239, 144), (240, 144), (241, 143), (242, 143), (244, 141), (245, 141), (254, 132), (255, 132), (256, 131), (257, 131), (258, 130), (260, 130), (261, 129), (262, 129), (263, 128), (264, 128), (265, 127), (266, 127), (267, 126), (269, 126), (270, 125), (271, 125), (278, 118), (279, 118), (281, 116), (282, 116), (283, 115), (284, 115), (285, 114), (286, 114), (287, 113), (289, 113), (290, 112), (291, 112), (292, 111), (293, 111), (295, 109), (296, 109), (297, 108), (298, 108), (301, 105), (302, 105), (304, 103), (305, 103), (306, 102), (307, 102), (308, 101), (309, 101), (310, 100), (312, 100), (313, 99), (316, 99), (317, 98), (319, 98), (320, 97), (322, 97), (323, 96), (325, 96), (326, 95), (327, 95), (328, 94), (329, 94), (330, 93), (331, 93), (333, 91), (334, 91), (336, 89), (337, 89), (338, 88), (339, 88), (340, 87), (341, 87), (342, 86), (343, 86), (344, 85), (349, 85), (350, 84), (355, 84), (356, 83), (363, 83), (364, 82), (370, 82), (371, 81), (383, 81), (384, 80), (399, 80), (400, 81), (412, 81), (413, 82), (421, 82), (422, 83), (428, 83), (429, 84), (434, 84), (435, 85), (437, 85), (438, 86), (440, 86), (441, 87), (443, 87), (444, 88), (445, 88), (446, 89), (447, 89), (451, 93), (452, 93), (454, 95), (455, 95), (456, 96), (457, 96), (458, 97), (459, 97), (460, 98), (462, 98), (463, 99), (465, 99), (466, 100), (468, 100), (469, 101), (471, 101), (472, 102), (473, 102), (475, 104), (476, 104), (478, 106), (479, 106), (483, 110), (484, 110), (485, 111), (486, 111), (487, 112), (488, 112), (489, 113), (490, 113), (491, 114), (493, 114), (494, 115), (495, 115), (497, 117), (498, 117), (505, 124), (506, 124), (508, 126), (510, 126), (511, 127), (512, 127), (513, 128), (515, 128), (516, 129), (517, 129), (518, 130), (520, 130), (521, 131), (522, 131), (524, 133), (525, 133), (527, 135), (527, 136), (530, 139), (531, 139), (533, 141), (534, 141), (535, 142), (536, 142), (537, 143), (538, 143), (541, 146), (542, 146), (543, 147), (543, 148), (551, 156), (552, 156), (553, 157), (554, 157), (556, 159), (557, 159), (569, 171), (569, 172), (571, 174), (571, 175), (572, 176), (572, 177), (573, 178), (573, 179), (574, 180), (574, 182), (575, 183), (575, 185), (576, 186), (576, 187), (577, 188), (577, 189), (578, 190), (578, 191), (579, 192), (579, 193), (581, 195), (581, 197), (582, 198), (582, 200), (583, 201), (583, 203), (584, 204), (584, 207), (585, 208), (585, 230), (585, 231), (585, 232), (585, 233), (585, 234), (585, 235), (585, 236), (585, 237), (585, 238), (585, 239), (585, 240), (585, 241), (586, 242), (586, 243), (586, 244), (586, 245), (586, 246), (586, 247), (586, 248), (586, 249), (586, 250), (586, 251), (586, 252), (586, 253), (586, 254), (586, 255), (586, 256), (586, 257), (586, 258), (586, 259), (586, 260), (586, 261), (586, 262), (586, 263), (586, 264), (586, 265), (586, 266), (586, 267), (586, 268), (586, 269), (586, 270), (586, 271), (586, 272), (586, 273), (586, 274), (586, 275), (585, 276), (585, 277), (585, 278), (585, 279), (585, 280), (585, 281), (585, 282), (585, 283), (585, 284), (585, 285), (585, 286), (585, 287), (585, 288), (585, 289), (585, 290), (585, 291), (584, 292), (584, 293), (584, 294), (584, 295), (584, 296), (584, 297), (584, 298), (584, 299), (584, 300), (584, 301), (584, 302), (583, 303), (583, 304), (583, 305), (583, 306), (583, 307), (583, 308), (583, 309), (583, 310), (582, 311), (582, 312), (582, 313), (582, 314), (582, 315), (582, 316), (582, 317), (581, 318), (581, 319), (581, 320), (581, 321), (581, 322), (581, 323), (581, 324), (580, 325), (580, 326), (580, 327), (580, 328), (580, 329), (580, 330), (579, 331), (579, 332), (579, 333), (579, 334), (579, 335), (578, 336), (578, 337), (578, 338), (578, 339), (578, 340), (577, 341), (577, 342), (577, 343), (577, 344), (577, 345), (576, 346), (576, 347), (576, 348), (576, 349), (576, 350), (575, 351), (575, 352), (575, 353), (575, 354), (574, 355), (574, 356), (574, 357), (574, 358), (573, 359), (573, 360), (573, 361), (573, 362), (572, 363), (572, 364), (572, 365), (572, 366), (571, 367), (571, 368), (571, 369), (571, 370), (570, 371), (570, 372), (570, 373)]]

    commands = []
    for p in pts:
      pf = fitpath(p)
      sp = pathtosvg(pf)
      commands.append(sp)
    drawsvg(commands)
