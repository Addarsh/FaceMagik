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

def drawsvg(commands, fills, widths):
  dwg = svgwrite.Drawing('test.svg', size=(1024,1024))
  for i, command in enumerate(commands):
    dwg.add(dwg.path(d=command,
      stroke="#000",
      fill=fills[i],
      stroke_width=widths[i]))
  dwg.save()


if __name__ == '__main__':
    pts =  [[(514, 275), (515, 274), (515, 270), (517, 266), (517, 264), (518, 263), (518, 261), (521, 257), (522, 254), (525, 251), (526, 247), (528, 245), (529, 240), (535, 229), (535, 227), (536, 226), (536, 222), (537, 221), (537, 217), (533, 217), (530, 220), (487, 214), (486, 213), (486, 209), (485, 208), (484, 202), (483, 201), (482, 192), (481, 191), (481, 187), (480, 186), (480, 183), (479, 182), (479, 175), (478, 174), (478, 169), (473, 160), (473, 155), (472, 154), (472, 147), (471, 146), (471, 141), (470, 140), (470, 135), (469, 134), (469, 129), (468, 128), (468, 125), (464, 118), (463, 113), (460, 107), (452, 99), (451, 99), (446, 95), (442, 93), (428, 93), (427, 92), (422, 92), (421, 93), (417, 93), (416, 94), (407, 94), (406, 95), (400, 96), (399, 97), (389, 98), (385, 100), (383, 100), (380, 102), (377, 102), (374, 104), (366, 106), (365, 107), (363, 107), (362, 108), (359, 108), (358, 109), (341, 109), (340, 110), (338, 110), (337, 109), (332, 109), (331, 108), (328, 108), (327, 109), (312, 109), (300, 119), (300, 120), (296, 124), (293, 130), (293, 132), (291, 135), (290, 143), (289, 144), (289, 158), (288, 159), (287, 171), (285, 175), (285, 177), (284, 178), (283, 185), (282, 186), (282, 189), (281, 190), (281, 194), (239, 204), (232, 197), (229, 197), (228, 196), (222, 196), (221, 195), (216, 195), (214, 194), (213, 195), (210, 195), (209, 196), (208, 198), (208, 201), (207, 202), (207, 206), (208, 207), (208, 212), (210, 216), (210, 218), (212, 221), (212, 223), (217, 231), (217, 233), (219, 235), (219, 237), (222, 243), (222, 245), (224, 248), (224, 250), (227, 256), (240, 272), (242, 276), (244, 278), (245, 281), (247, 283), (247, 284), (248, 284), (250, 286), (251, 288), (255, 289), (259, 292), (265, 292), (266, 291), (277, 290), (278, 291), (278, 295), (280, 299), (280, 301), (281, 302), (281, 305), (282, 306), (282, 310), (283, 311), (283, 316), (284, 317), (284, 319), (290, 329), (290, 331), (292, 334), (292, 336), (294, 340), (303, 349), (303, 350), (310, 358), (311, 361), (314, 364), (326, 369), (333, 374), (338, 375), (339, 376), (342, 376), (346, 378), (357, 379), (358, 380), (361, 380), (362, 381), (365, 381), (369, 383), (371, 383), (372, 384), (376, 384), (377, 385), (384, 385), (385, 386), (398, 386), (399, 385), (405, 385), (411, 382), (416, 381), (419, 379), (424, 378), (432, 374), (438, 369), (452, 363), (474, 342), (475, 342), (477, 340), (479, 336), (479, 334), (481, 331), (482, 326), (484, 323), (485, 318), (486, 317), (486, 312), (487, 311), (487, 303), (488, 302), (488, 292), (489, 291), (489, 282), (490, 281), (503, 288), (505, 286), (508, 285), (511, 282), (512, 279), (514, 277)], [(209, 195), (209, 194), (209, 193), (209, 192), (209, 191), (209, 190), (209, 189), (209, 188), (209, 187), (210, 186), (210, 185), (210, 184), (210, 183), (210, 182), (210, 181), (210, 180), (210, 179), (210, 178), (211, 177), (211, 176), (211, 175), (211, 174), (211, 173), (211, 172), (212, 171), (212, 170), (212, 169), (212, 168), (213, 167), (213, 166), (213, 165), (214, 164), (214, 163), (214, 162), (214, 161), (215, 160), (215, 159), (216, 158), (216, 157), (216, 156), (217, 155), (217, 154), (217, 153), (218, 152), (218, 151), (219, 150), (219, 149), (220, 148), (220, 147), (223, 144), (223, 143), (224, 142), (224, 141), (225, 140), (225, 139), (226, 138), (226, 135), (227, 134), (227, 131), (228, 130), (228, 128), (229, 127), (229, 125), (230, 124), (230, 122), (231, 121), (231, 119), (233, 117), (233, 116), (236, 113), (236, 112), (238, 110), (238, 108), (239, 107), (239, 106), (240, 105), (240, 102), (241, 101), (241, 98), (242, 97), (242, 93), (243, 92), (243, 90), (244, 89), (244, 87), (245, 86), (245, 85), (246, 84), (246, 83), (247, 82), (247, 81), (250, 78), (250, 77), (251, 76), (251, 74), (252, 73), (252, 71), (253, 70), (253, 67), (254, 66), (254, 63), (255, 62), (255, 60), (257, 58), (257, 57), (262, 52), (262, 51), (264, 49), (264, 48), (265, 47), (265, 46), (266, 45), (266, 44), (267, 43), (267, 42), (272, 37), (273, 37), (283, 27), (284, 27), (285, 26), (286, 26), (288, 24), (289, 24), (291, 22), (293, 22), (294, 21), (295, 21), (296, 20), (301, 20), (302, 19), (308, 19), (309, 18), (320, 18), (321, 17), (328, 17), (329, 16), (334, 16), (335, 15), (350, 15), (351, 14), (359, 14), (360, 13), (393, 13), (394, 14), (403, 14), (404, 15), (410, 15), (411, 16), (422, 16), (423, 17), (432, 17), (433, 18), (455, 18), (456, 19), (468, 19), (469, 20), (480, 20), (481, 21), (484, 21), (485, 22), (487, 22), (488, 23), (490, 23), (491, 24), (493, 24), (497, 28), (498, 28), (506, 36), (506, 37), (510, 41), (510, 42), (512, 44), (512, 45), (514, 47), (514, 48), (517, 51), (517, 52), (525, 60), (526, 60), (530, 64), (530, 65), (531, 66), (531, 67), (532, 68), (532, 69), (533, 70), (533, 71), (534, 72), (534, 73), (537, 76), (537, 77), (540, 80), (541, 80), (542, 81), (542, 84), (543, 85), (543, 91), (544, 92), (544, 93), (545, 94), (545, 96), (546, 97), (546, 99), (547, 100), (547, 123), (547, 124), (547, 125), (547, 126), (547, 127), (547, 128), (547, 129), (547, 130), (547, 131), (547, 132), (547, 133), (547, 134), (547, 135), (547, 136), (547, 137), (547, 138), (547, 139), (547, 140), (547, 141), (547, 142), (547, 143), (547, 144), (547, 145), (547, 146), (547, 147), (547, 148), (547, 149), (547, 150), (547, 151), (547, 152), (547, 153), (547, 154), (546, 155), (546, 156), (546, 157), (546, 158), (546, 159), (546, 160), (546, 161), (546, 162), (546, 163), (546, 164), (546, 165), (546, 166), (546, 167), (545, 168), (545, 169), (545, 170), (545, 171), (545, 172), (545, 173), (545, 174), (545, 175), (544, 176), (544, 177), (544, 178), (544, 179), (544, 180), (544, 181), (544, 182), (543, 183), (543, 184), (543, 185), (543, 186), (543, 187), (543, 188), (543, 189), (542, 190), (542, 191), (542, 192), (542, 193), (542, 194), (541, 195), (541, 196), (541, 197), (541, 198), (541, 199), (540, 200), (540, 201), (540, 202), (540, 203), (540, 204), (539, 205), (539, 206), (539, 207), (539, 208), (538, 209), (538, 210), (538, 211), (538, 212), (537, 213), (537, 214), (537, 215), (537, 216)], [(267, 290), (268, 289), (268, 288), (269, 287), (269, 285), (270, 284), (270, 277), (269, 276), (269, 273), (268, 272), (268, 268), (267, 267), (267, 266), (266, 265), (266, 263), (265, 262), (265, 260), (264, 259), (264, 255), (263, 254), (263, 250), (262, 249), (262, 247), (261, 246), (261, 244), (260, 243), (260, 242), (259, 241), (259, 240), (258, 239), (258, 237), (256, 235), (256, 233), (254, 231), (254, 230), (253, 229), (253, 228), (251, 226), (251, 225), (250, 224), (250, 223), (249, 222), (249, 221), (248, 220), (248, 219), (247, 218), (247, 217), (246, 216), (246, 215), (245, 214), (245, 213), (244, 212), (244, 211), (242, 209), (242, 208), (239, 205)], [(530, 221), (525, 226), (523, 226), (522, 227), (518, 227), (517, 228), (516, 228), (513, 231), (513, 232), (512, 233), (512, 235), (511, 236), (511, 237), (510, 238), (510, 240), (509, 241), (509, 242), (508, 243), (508, 245), (507, 246), (507, 248), (506, 249), (506, 252), (505, 253), (505, 266), (504, 267), (504, 271), (503, 272), (503, 273), (502, 274), (502, 277), (501, 278), (501, 281), (500, 282), (500, 285), (501, 286)]]

    commands = []
    fills = ["none" for i in range(len(pts))]
    widths = [5 for i in range(len(pts))]
    for i, p in enumerate(pts):
      pf = fitpath(p)
      sp = pathtosvg(pf)
      #print ("sp: ", sp)
      if i == 0:
        sp += " Z"
        fills[i] = "#654321"
      elif i > 1:
        widths[i] = 5
      commands.append(sp)
    drawsvg(commands, fills, widths)
