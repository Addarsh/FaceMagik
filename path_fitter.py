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

def drawsvg(command):
  dwg = svgwrite.Drawing('test.svg', size=(1024,1024))
  dwg.add(dwg.path( d=command,
    stroke="#000",
    fill="none",
    stroke_width=5))
  dwg.save()


if __name__ == '__main__':
    p = [[185, 502], [188, 502], [190, 504], [223, 494], [224, 493], [224, 485], [225, 484], [225, 477], [226, 476], [226, 470], [227, 469], [228, 461], [229, 460], [230, 455], [232, 452], [232, 450], [233, 449], [233, 447], [235, 443], [235, 440], [236, 439], [236, 436], [237, 435], [238, 428], [239, 427], [240, 422], [242, 419], [242, 417], [245, 412], [245, 410], [246, 409], [246, 407], [247, 406], [247, 404], [248, 403], [248, 401], [249, 400], [249, 398], [250, 397], [252, 389], [257, 380], [257, 378], [258, 377], [260, 369], [267, 355], [269, 353], [271, 349], [289, 330], [295, 327], [297, 325], [301, 323], [304, 320], [314, 315], [317, 315], [318, 314], [326, 314], [327, 313], [342, 313], [343, 312], [439, 312], [440, 313], [452, 313], [453, 314], [456, 314], [457, 315], [459, 315], [460, 316], [464, 317], [476, 329], [479, 330], [482, 333], [483, 333], [496, 346], [497, 346], [504, 352], [507, 358], [512, 364], [512, 365], [516, 369], [519, 376], [521, 378], [523, 382], [526, 385], [529, 391], [529, 394], [530, 395], [530, 398], [531, 399], [531, 403], [532, 404], [533, 411], [536, 417], [536, 419], [540, 426], [541, 432], [542, 433], [542, 436], [543, 437], [543, 442], [544, 443], [544, 448], [545, 449], [546, 458], [547, 459], [548, 466], [551, 472], [551, 475], [552, 476], [552, 479], [553, 480], [553, 483], [554, 484], [554, 489], [555, 490], [555, 511], [556, 512], [556, 612], [555, 613], [555, 631], [554, 632], [554, 639], [553, 640], [552, 647], [550, 650], [550, 652], [548, 655], [548, 657], [546, 661], [546, 664], [545, 665], [544, 672], [540, 680], [535, 686], [534, 690], [527, 704], [523, 709], [520, 715], [518, 723], [515, 729], [510, 735], [507, 741], [506, 745], [498, 754], [495, 760], [490, 765], [482, 769], [478, 773], [477, 773], [474, 777], [473, 777], [470, 780], [464, 783], [456, 784], [455, 785], [452, 785], [451, 786], [443, 788], [439, 790], [434, 794], [428, 797], [424, 797], [423, 798], [416, 798], [415, 799], [389, 799], [388, 798], [380, 798], [379, 797], [375, 797], [374, 796], [372, 796], [369, 794], [365, 793], [363, 791], [356, 787], [354, 787], [353, 786], [351, 786], [347, 784], [344, 784], [343, 783], [338, 782], [332, 779], [330, 777], [329, 777], [328, 775], [323, 770], [306, 761], [297, 751], [290, 747], [286, 742], [284, 737], [274, 727], [271, 720], [263, 712], [263, 711], [260, 707], [259, 703], [251, 695], [251, 694], [248, 690], [248, 688], [246, 684], [239, 676], [238, 672], [237, 671], [237, 669], [236, 668], [235, 662], [231, 654], [228, 650], [228, 648], [226, 644], [226, 641], [225, 640], [225, 637], [224, 636], [215, 633], [213, 634], [212, 633], [210, 633], [206, 630], [204, 626], [204, 624], [203, 623], [203, 618], [202, 617], [202, 609], [201, 608], [201, 598], [200, 597], [200, 590], [199, 589], [198, 580], [197, 579], [197, 576], [196, 575], [196, 572], [195, 571], [194, 564], [193, 563], [193, 561], [191, 557], [191, 554], [189, 551], [189, 549], [187, 545], [187, 542], [185, 538], [185, 530], [184, 529], [184, 505], [185, 504]]


    pf = fitpath(p)
    print(pf)
    sp = pathtosvg(pf)
    print(sp)
    drawsvg(sp)
