# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    is_circle = alien.is_circle()
    radius = alien.get_width()
    position = alien.get_centroid() if is_circle else alien.get_head_and_tail()
    
    for wall in walls:
        wall_seg = ((wall[0], wall[1]), (wall[2], wall[3]))

        # Check if the circle (alien) intersects with the wall segment
        if is_circle and point_segment_distance(position, wall_seg) <= radius:
            return True
        # Check if the oblong (alien) intersects with the wall segment
        if not is_circle and segment_distance(position, wall_seg) <= radius:
            return True

    return False

def within_window_helper(vertices, head_and_tail, radius, window: Tuple[int]):
    edges = (((0, 0),(window[0], 0)), 
             ((window[0], 0),(window[0], window[1])), 
             ((window[0], window[1]),(0, window[1])), 
             ((0, window[1]),(0, 0)))
    
    (x_head,y_head),(x_tail,y_tail) = head_and_tail

    # Check if all vertices are in the window
    for p in vertices:
        if not (0 <= p[0] <= window[0] and 0 <= p[1] <= window[1]):
            return False
    
    # Check if all vertices are far enough from the window
    for edge in edges:
        if point_segment_distance((x_head,y_head), edge) <= radius or point_segment_distance((x_tail,y_tail), edge) <= radius:
            return False

    return True

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """

    radius = alien.get_width()
    (x_head,y_head),(x_tail,y_tail) = alien.get_head_and_tail()

    if x_head == x_tail and y_head == y_tail:
        # Ball
        pos_x, pos_y = alien.get_centroid()
        return (radius <= pos_x <= window[0] - radius
        and radius <= pos_y <= window[1] - radius)
    
    if x_head == x_tail:
        # Vertical
        alien_vertices = ((x_head-radius,y_head), 
                (x_head+radius,y_head), 
                (x_tail-radius,y_tail), 
                (x_tail+radius,y_tail))
    else:
        # Horizontal
        alien_vertices = ((x_head,y_head-radius), 
                (x_head,y_head+radius), 
                (x_tail,y_tail-radius), 
                (x_tail,y_tail+radius))
        
    return within_window_helper(alien_vertices, alien.get_head_and_tail(), radius, window)


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    num_vertices = len(polygon)

    # Check if the point is inside the parallelogram by seeing if the Euclidean vectors is always clockwise or counter-clockwise
    all_in_a_line, clockWise, counterclockWise = True, True, True

    for i in range(num_vertices):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % num_vertices]
        cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

        all_in_a_line &= (cross == 0)
        clockWise &= (cross < 0)
        counterclockWise &= (cross >= 0)

        if not clockWise and not counterclockWise:
            return False

    if not all_in_a_line:
        return True
    
    # If all the vertices and the point are in the same line
    maxX, minX = polygon[0][0], polygon[0][0]
    maxY, minY = polygon[0][1], polygon[0][1]
    for vertex in polygon:
        maxX, minX = max(vertex[0], maxX), min(vertex[0], minX)
        maxY, minY = max(vertex[1], maxY), min(vertex[1], minY)
    return minX <= point[0] <= maxX and minY <= point[1] <= maxY


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    radius = alien.get_width()
    if alien.is_circle():
        # Check if the circle (alien) intersects with any walls on its path
        path_segment = (alien.get_centroid(), waypoint)

        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            if segment_distance(path_segment, wall_segment) <= radius:
                return True
    else:
        # Check if the oblong (alien) intersects with any walls on its path
        path_head, path_tail = alien.get_centroid(), waypoint
        vector = (path_tail[0] - path_head[0], path_tail[1] - path_head[1])
        head, tail = alien.get_head_and_tail()
        goalHead = (head[0] + vector[0], head[1] + vector[1])
        goalTail = (tail[0] + vector[0], tail[1] + vector[1])

        Parallelogram_vertices = (head, tail, goalTail, goalHead)
        Parallelogram_edges = ((head,goalHead), (tail,goalTail), (head,tail), (goalHead,goalTail))
        
        for wall in walls:
            # Check if the whole segment is inside the polygon
            if is_point_in_polygon((wall[0], wall[1]), Parallelogram_vertices):
                return True
            
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            for edge in Parallelogram_edges:
                if segment_distance(edge, wall_segment) <= radius:
                    return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    # Calculate the Euclidean distance from a point to a line segment
    x, y = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    dx = x2 - x1
    dy = y2 - y1

    if dx == dy == 0:  # Line segment is just a point
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5

    t = ((x - x1) * dx + (y - y1) * dy) / (dx ** 2 + dy ** 2)

    if t < 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    elif t > 1:
        return ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
    else:
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    ((x1, y1), (x2, y2)) = s1
    ((x3, y3), (x4, y4)) = s2

    # Calculate the directions of the line segments
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    if dx1 == 0 and dy1 == 0:
        return point_segment_distance(s1[0], s2) == 0
    if dx2 == 0 and dy2 == 0:
        return point_segment_distance(s2[0], s1) == 0

    # Calculate the determinant
    det = dx1 * dy2 - dx2 * dy1

    # Check if the line segments are parallel (det == 0)
    if det == 0:
        # Check if the line segments are in the same line
        if dx1 == 0:
            if x1 == x3:
                return min(y1, y2) <= max(y3, y4) and max(y1, y2) >= min(y3, y4)
            else:
                return False
        if dy1 == 0:
            if y1 == y3:
                return min(x1, x2) <= max(x3, x4) and max(x1, x2) >= min(x3, x4)
            else:
                return False
        if y1 - dy1/float(dx1) * x1 == y3 - dy2/float(dx2) * x3:
            # Return if the line segments overlap
            return min(x1, x2) <= max(x3, x4) and max(x1, x2) >= min(x3, x4)
        return False
    

    # Calculate the parameters for intersection
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det

    # Check if the intersection is within both line segments
    return 0 <= t1 <= 1 and 0 <= t2 <= 1


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0.0  # The segments intersect, so distance is zero

    # Calculate the distances between each endpoint of s1 and s2
    distances = [
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1)
    ]

    # Return the minimum distance among all pairs of endpoints
    return min(distances)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                    f'{b} is expected to be {result[i][j][k]}, but your' \
                    f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
