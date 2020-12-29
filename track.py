import math
import numpy as np
import random

from scipy.spatial import ConvexHull
from scipy import interpolate

def getdataset():
    points = random_points()
    hull = ConvexHull(points)
    track_points = shape_track(get_track_points(hull,points))

    c = CatmullRomChain(track_points)
    x,y = zip(*c)
    x1,y1 = second_point(x,y)

    cx,cy = checkpoints2(x,y,x1,y1)
    return x,y,x1,y1,cx,cy

def get_checkpoints(track_points, n_checkpoints):
    checkpoint_step = len(track_points) // n_checkpoints

    checkpoints = []
    for i in range(n_checkpoints):
        index = i * checkpoint_step
        checkpoints.append(track_points[index])

    return checkpoints

def smooth_track(track_points):
    x,y = zip(*track_points)
    print(x,y)
    #x = np.array(x[i] for i in range(len(x)))
    #y = np.array(y[i] for i in range(len(y)))

    x=np.r_[x,x[0]]
    y=np.r_[y,y[0]]
    print(x,y)
    tck,u = interpolate.splprep([x,y],s=0,per=True)
    xi,yi = interpolate.splev(np.linspace(0,1,1000),tck)
    return [(int(xi[i]),int(yi[i])) for i in range(len(xi))]

def checkpoints2(x,y,x1,y1):
    cx =[]
    cy =[]
    first = True
    for i in range(len(x)):
        dist = math.sqrt((x[i] - x1[i]) ** 2 + (y[i] - y1[i]) ** 2)
        distnew = dist / 2
        pt2 = (x1[i],y1[i])
        pt1 = (x[i], y[i])
        checkpoint = intersection(pt1,distnew,pt1,pt2)
        newcx = int(checkpoint[0][0])
        newcy = int(checkpoint[0][1])
        if not first and newcx == oldcx and newcy == oldcy:
            continue
        else:
            first = False
            oldcx = newcx
            oldcy = newcy

            cx.append(newcx)
            cy.append(newcy)

    return cx,cy

def checkpoints1(track_points):
    cx =[]
    x, y = zip(*track_points)
    first = True
    for i in range(len(x)):
        dist = math.sqrt((x[i] - 49.5) ** 2 + (y[i] - 49.5) ** 2) - 4
        pt1 = (49.5, 49.5)
        pt2 = (x[i], y[i])
        checkpoint = intersection(pt1,dist,pt1,pt2)
        newcx = int(checkpoint[0][0])
        newcy = int(checkpoint[0][1])
        if not first and newcx == oldcx and newcy == oldcy:
            continue
        else:
            first = False
            oldcx = newcx
            oldcy = newcy

            cx.append((newcx,newcy))


    return np.array(cx)


def second_point(x,y):
    newx = []
    newy = []
    for i in range(len(x)):
        dist = math.sqrt((x[i]-49.5)**2 + (y[i]-49.5)**2) - 8
        pt1 = (49.5,49.5)
        pt2 = (x[i],y[i])
        new = intersection(pt1, dist, pt1, pt2)
        newx.append(new[0][0])
        newy.append(new[0][1])

    return newx,newy

def intersection(center,radius,pt1,pt2,full_line = False,tangent_tol = 1e-4):
    (p1x,p1y),(p2x,p2y),(cx,cy) = pt1,pt2,center
    (x1,y1),(x2,y2) = (p1x-cx,p1y-cy),(p2x-cx,p2y-cy)
    dx,dy = (x2-x1),(y2-y1)
    dr = (dx**2 + dy**2)**.5
    big_d = x1*y2 - x2*y1
    discriminant = radius **2 *dr**2 - big_d**2
    if discriminant <0:
        return[]
    else:
        intersections = [(cx + (big_d*dy+sign*(-1 if dy<0 else 1)*dx*discriminant**.5)/dr**2,cy + (-big_d*dx+sign*abs(dy)*discriminant**.5)/dr**2) for sign in ((1,-1) if dy <0 else (-1,1))]
        if not full_line:
            fraction_along_segment = [(xi-p1x)/dx if abs(dx) > abs(dy) else (yi-p1y)/dy for xi,yi in intersections]
            intersections = [pt for pt,frac in zip(intersections,fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
                return [intersections[0]]
            else:
                return intersections

def random_points():
    pointcount = random.randrange(10, 21, 1)
    points = []
    for i in range(pointcount):
        x = (random.randrange(5, 95, 1))
        y = (random.randrange(5, 95, 1))
        distances = list(filter(lambda x: x < 15, [math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def get_track_points(hull,points):
    return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

def make_rand_vector(dims):
    vec = [random.gauss(0,1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def shape_track(track_points):
    track_set = [[0,0] for i in range(len(track_points)*2)]
    for i in range(len(track_points)):
        displacement = math.pow(random.random(),0.1)*1
        disp = [displacement*i for i in make_rand_vector(2)]
        track_set[i*2] = track_points[i]
        track_set[i*2 + 1][0] = int((track_points[i][0] + track_points[(i+1)%len(track_points)][0])/2 + disp[0])
        track_set[i * 2 + 1][1] = int((track_points[i][1] + track_points[(i + 1) % len(track_points)][1]) / 2 + disp[1])

    for i in range(3):
        track_set = fixAngles(track_set)
        track_set = pushApart(track_set)
    final_set = []
    for point in track_set:
        if point[0] < 0:
            point[0] = 0
        elif point[0] > 100:
            point[0] = 100
        if point[1]<0:
            point[1] = 0
        elif point[1]>100:
            point[1] = 100
        final_set.append(point)
    return final_set



def CatmullRomSpline(p0,p1,p2,p3,nPoints=1000):
    p0,p1,p2,p3=map(np.array,[p0,p1,p2,p3])
    alpha = 0.5
    alpha = alpha/2
    def tj(ti,pi,pj):
        xi,yi = pi
        xj,yj = pj
        return ((xj-xi)**2 + (yj-yi)**2)**alpha + ti

    t0 = 0
    t1 = tj(t0,p0,p1)
    t2 = tj(t1,p1,p2)
    t3 = tj(t2,p2,p3)

    t = np.linspace(t1,t2,nPoints)
    t = t.reshape(len(t),1)

    A1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

    C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2

    return C

def CatmullRomChain(P):
    sz = len(P)
    C = []
    for i in range(sz-3):
        c = CatmullRomSpline(P[i], P[i+1],P[i+2],P[i+3])
        C.extend(c)
    c = CatmullRomSpline(P[sz-3], P[sz-2], P[sz-1], P[0])
    C.extend(c)
    c = CatmullRomSpline(P[sz - 2], P[sz - 1], P[0], P[1])
    C.extend(c)
    c = CatmullRomSpline(P[sz - 1], P[0], P[1], P[2])
    C.extend(c)

    return C


def pushApart(dataset):
    dst = 6
    dst2 = dst * dst
    for i in range(len(dataset)):
        for j in range(i+1,len(dataset)):
            p_distance = math.sqrt((dataset[i][0]  - dataset[j][0])**2 + (
                    dataset[i][1]- dataset[j][1])**2)
            if p_distance < dst:
                hx = (dataset[j][0] - dataset[i][0])
                hy = (dataset[j][1] - dataset[i][1])
                hl = (math.sqrt(hx * hx + hy * hy))
                hx /= hl
                hy /= hl
                diff = dst - hl
                hx *= diff
                hy *= diff
                dataset[j][0] = int(dataset[j][0] + hx)
                dataset[j][1] = int(dataset[j][1] + hy)
                dataset[i][0] = int(dataset[i][0] - hx)
                dataset[i][1] = int(dataset[i][1] - hy)
    return dataset

def fixAngles(dataset):

    for i in range(len(dataset)):
        if i>0:
            previous = i-1
        else:
            previous = len(dataset) -1
        next = (i+1)%len(dataset)
        px = dataset[i][0] - dataset[previous][0]
        py = dataset[i][1] - dataset[previous][1]
        pl = (math.sqrt(px*px+py*py))

        px /= pl
        py /= pl

        nx = -(dataset[i][0] - dataset[next][0])
        ny = -(dataset[i][1] - dataset[next][1])

        nl = (math.sqrt(nx*nx+ny*ny))

        nx /=nl
        ny /=nl

        a = (math.atan2(px*ny-py*nx,px*nx+py*ny))

        if (abs(math.degrees(a))<=100):
            continue

        nA = math.radians(100*math.copysign(1,a))
        diff = nA - a
        cos = (math.cos(diff))
        sin = (math.sin(diff))
        newX = (nx*cos-ny*sin)*nl
        newY = (nx*sin+ny*cos)*nl

        dataset[next][0] = int(dataset[i][0]+newX)
        dataset[next][1] = int(dataset[i][1]+newY)
    return dataset
