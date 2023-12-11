def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x, y = p
    ((x1, y1), (x2, y2)) = s
    dx, dy = x2 - x1, y2 - y1

    if dy == 0:
        # For x1 <= x2
        # max(x1 - x, x - x2) < 0 means x1 <= x <= x2
        # max(x1 - x, x - x2) >= 0 means the x-distance to the closest x
        dx = max(0, max(min(x1, x2) - x, x - max(x1, x2)))
        return (dx ** 2 + (y-y1) ** 2) ** 0.5

    m = dx / dy

    c1, c2, c = m * x1 + y1, m * x2 + y2, m * x + y
    if c1 > c2:
        x1, x2, y1, y2, c1, c2 = x2, x1, y2, y1, c2, c1

    if c < c1:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    elif c > c2:
        return ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
    else:
        # a×b = a1b2 − a2b1 = |a||b|sinα
        return abs((x-x1)*dy - (y-y1)*dx) / (dx ** 2 + dy ** 2) ** 0.5
    

def point_segment_distance2(p, s):
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

# Example usage:
# test_cases = [
#     # ((2, 2), (0, 0), (4, 4)),
#     # ((-1, -1), (0, 0), (4, 4)),
#     # ((5, 5), (0, 0), (-4, 4)),
#     # ((3, 1), (0, 0), (4, 4)),
#     # ((1, 3), (0, 0), (-4, -4)),
#     # ((2, 5), (0, 0), (4, -4)),
#     # ((0, 2), (0, 0), (-4, 4)),
#     # ((0, 4), (0, 0), (4, 4)),
#     # ((4, 0), (0, 0), (4, 4)),
#     # ((-2, -2), (-4, -4), (0, 0)),
#     # ((50, 120), (50, 134), (50, 174)),
#     ((100.0, 40), (20, 10), (100, 10)),
# ]

# for point, start, end in test_cases:
#     distance = point_segment_distance(point, (start, end))
#     # distance2 = point_segment_distance2(point, (start, end))
#     print(f"Point: {point}, Start: {start}, End: {end}, Distance: {distance}")
#     # print(f"Point: {point}, Start: {start}, End: {end}, Distance2: {distance2}")

from geometry import do_segments_intersect

def test_do_segments_intersect():
    # Define the line segments as tuples of endpoint coordinates
    segment1 = ((40, 100), (-40, 100))
    segment2 = ((100, 100), (100, 100))
    segment3 = ((1.7, 2.1), (5, 1))
    segment4 = ((2, 2.9), (-7, 2))
    segment5 = ((8, 8), (10, 10))

    # Example usage of the do_segments_intersect function
    print(do_segments_intersect(segment1, segment2))
    print(do_segments_intersect(segment1, segment3))
    print(do_segments_intersect(segment1, segment4))
    print(do_segments_intersect(segment1, segment5))


from geometry import does_alien_touch_wall
def test_does_alien_touch_wall():
    test_cases = [
        [
                # (20, 10, 100, 10),
                # (20, 70, 100, 70),
                # (20, 10, 20, 70),
                # (100, 10, 120, 20),
                # (100, 70, 120, 55),
                # (120, 20, 220, 20),
                # (120, 55, 170, 55),
                # (220, 20, 260, 110),
                # (170, 55, 200, 110),
                # (200, 110, 50, 110),
                # (200, 120, 50, 120),
                (50, 110, 50, 120),
                # (200, 120, 200, 150),
                # (260, 110, 260, 130),
                # (260, 130, 235, 160),
                # (235, 160, 235, 220),
                # (200, 150, 100, 150),
                # (100, 150, 50, 200),
                # (100, 150, 100, 200),
                # (200, 200, 50, 200),
                # (235, 220, 300, 220),
                # (200, 280, 300, 280),
                # (300, 220, 300, 280),
                # (200, 200, 200, 280)

            ],
        [
                (20, 10, 100, 10),
                (20, 55, 100, 55),
                (20, 10, 20, 55),
                (100, 10, 120, 30),
                (100, 55, 120, 50),
                (120, 30, 220, 30),
                (120, 50, 170, 50),
                (220, 30, 250, 110),
                (170, 50, 200, 110),
                (200, 110, 50, 110),
                (200, 120, 50, 120),
                (50, 110, 50, 120),
                (200, 120, 200, 150),
                (250, 110, 250, 130),
                (250, 130, 220, 160),
                (220, 160, 220, 200),
                (200, 150, 100, 150),
                (100, 150, 50, 200),
                (125, 150, 125, 200),
                (220, 200, 50, 200)
            ]
    ]
    # A = Alien((50,154), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', (310, 300))
    A = Alien((100,40), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', (260, 210))
    result = does_alien_touch_wall(A, test_cases[1])
    assert result == False

    # for point, polygon, expected_result in test_cases:
    #     result = does_alien_touch_wall(A, polygon)
    #     assert result == expected_result, f'does_alien_touch_wall with point {point} ' \
    #                                      f'and polygon {polygon} returns {result}, expected: {expected_result}'

from geometry import is_point_in_polygon

def test_is_point_in_polygon():
    test_cases = [
        ((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)], True),
        ((10, 5), [(0, 0), (10, 0), (10, 10), (0, 10)], True),
        ((3, 2), [(1, 1), (6, 1), (4, 3), (-1, 3)], True),
        ((10, 10), [(0, 0), (10, 0), (10, 10), (0, 10)], True),
        ((2, 0), [(0, 0), (1, 0), (3, 0), (4, 0)], True),
        ((0, 2), [(0, 0), (0, 1), (0, 3), (0, 4)], True),
        ((5, 0), [(0, 0), (1, 0), (3, 0), (4, 0)], False),
        ((0, 5), [(0, 0), (0, 1), (0, 3), (0, 4)], False),
        # ((50, 119), [(30, 110.0), (30, 130.0), (70, 130.0), (70, 110.0)], True),
    ]

    for point, polygon, expected_result in test_cases:
        result = is_point_in_polygon(point, polygon)
        assert result == expected_result, f'is_point_in_polygon with point {point} ' \
                                         f'and polygon {polygon} returns {result}, expected: {expected_result}'
from alien import Alien
from geometry import is_alien_within_window
from geometry import does_alien_path_touch_wall
def test_is_alien_within_window():
    window = (220, 200)  # Window size

    # Test cases with expected results
    test_cases = [
        (Alien([0, 100], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window), False),
        (Alien([25.5, 25.5], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window), False),
        (Alien([25.5, 25.5], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window), False),
        (Alien([194.5, 174.5], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window), False),
        (Alien([194.5, 174.5], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window), False),
        (Alien([205.5, 100], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window), False),
        (Alien([30, 120], [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window), True),
    ]

    for i, (alien, expected_result) in enumerate(test_cases):
        result = is_alien_within_window(alien, window)
        if result == expected_result:
            print(f"Test case {i + 1}: PASSED")
        else:
            print(f"Test case {i + 1}: FAILED (Expected: {expected_result}, Got: {result})")

def test_does_alien_path_touch_wall():
    # Define the alien with the specified configuration
    alien = Alien((100, 40), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', (260, 210))

    # Define the list of walls
    walls = [
                (20, 10, 100, 10),
                (20, 55, 100, 55),
                (20, 10, 20, 55),
                (100, 10, 120, 30),
                (100, 55, 120, 50),
                (120, 30, 220, 30),
                (120, 50, 170, 50),
                (220, 30, 250, 110),
                (170, 50, 200, 110),
                (200, 110, 50, 110),
                (200, 120, 50, 120),
                (50, 110, 50, 120),
                (200, 120, 200, 150),
                (250, 110, 250, 130),
                (250, 130, 220, 160),
                (220, 160, 220, 200),
                (200, 150, 100, 150),
                (100, 150, 50, 200),
                (125, 150, 125, 200),
                (220, 200, 50, 200)
            ]

    # Define the waypoint
    waypoint = (120, 40)

    # Check if the function returns the expected result
    result = does_alien_path_touch_wall(alien, walls, waypoint)
    expected_result = False
    assert result == expected_result, f"Result: {result}, Expected: {expected_result}"


if __name__ == '__main__':
    # test_is_point_in_polygon()
    # test_does_alien_path_touch_wall()
    test_do_segments_intersect()