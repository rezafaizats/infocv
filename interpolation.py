import numpy as np

def order_points(pts):
    """
    Orders the 4 points as top-left(TL), top-right(TR), bottom-right(BR), bottom-left(BL).
    
    Returns: a list of 4 points in the order of TL, TR, BR, BL.
    """
    # Sort by Y (from top to bottom)
    pts = sorted(pts, key=lambda p: p[1])

    # The 2 in the top are sorted by X (from left to right)
    top = sorted(pts[:2], key=lambda p: p[0])

    # The 2 in the bottom are sorted by X (from left to right)
    bottom = sorted(pts[2:], key=lambda p: p[0])

    tl, tr = top
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype="float32")



def linear_interpolation(points, size):
    """
    Linear interpolation of all the internal corners from 4 corner points.
    
    Returns: a list of the internal corners with format similar to the OpenCV function
    """
    cols, rows = size 
    tl, tr, br, bl = order_points(points) # order points as TL, TR, BR, BL

    corners = []
    for i in range(rows):
        # Edge interpolation: interpolate left and right edge points for each row
        v = i / (rows - 1) 
        left = (1 - v) * tl + v * bl #Interpolate left edge: L(v)
        right = (1 - v) * tr + v * br #Interpolate right edge: R(v)

        for j in range(cols):
            # Interpolate between left and right edge points for each column
            u = j / (cols - 1) 
            p = (1 - u) * left + u * right # Interpolate the point between left and right edge: P(u,v)
            corners.append(p)
    
    return np.array(corners, dtype="float32").reshape(-1, 1, 2)