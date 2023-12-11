from state import euclidean_distance
# def euclidean_distance(a, b):
#     # Calculate the Euclidean distance between two points (x1, y1) and (x2, y2)
#     x1, y1, _ = a
#     x2, y2, _ = b
#     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# (30, 120, 1), (49, 77, 1) 47.01
# (110, 40, 1), (49, 77, 1) 71.34
# (49, 77, 1) 118.35
# (66, 58, 1) 119.23
# (30, 120, 0) 113.137
# (55, 125, 1) 126.73
s, g = (30, 120, 1), (110, 40, 1)
b = (55, 125, 1)
print(euclidean_distance(s, b)+euclidean_distance(g, b))