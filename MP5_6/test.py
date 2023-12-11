# geometry.py
# ---------------
# Licensing Information: You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018
"""
This file contains geometry functions related to Part 1 in MP2.
"""
import math
import pdb
# import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end coordinate based on the given start position, length, and angle.
    
    Args:
        start (tuple): base of the arm link. (x-coordinate, y-coordinate)
        length (int): length of the arm link
        angle (int): degree of the arm link from the x-axis counterclockwise
    
    Returns:
        End position (int, int): of the arm link, (x-coordinate, y-coordinate)
    """
    x = math.cos(angle * math.pi / 180) * length
    y = math.sin(angle * math.pi / 180) * length
    if x >= 0:
        x = math.floor(x)
    else:
        x = math.ceil(x)
    if y >= 0:
        y = math.floor(y)
    else:
        y = math.ceil(y)
    x += start[0]
    y = start[1] - y
    return (x, y)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal.
    
    Args:
        armPosDist (list): start and end position and padding distance of all
            arm links [(start, end, distance)]
        objects (list): x-, y- coordinates and radius of objects (obstacles or
            goals) [(x, y, r)]
        isGoal (bool): True if the object is a goal and False if the object is
            an obstacle. When the object is an obstacle, consider padding
            distance. When the object is a goal, no need to consider padding
            distance.
    
    Returns:
        True if touched. False if not.
    """
    x = 0
    y = 0
    # pdb.set_trace()
    for arm in armPosDist:
        armLength = math.sqrt((arm[0][0] - arm[1][0]) ** 2 + (arm[0][1] - arm[1][1]) ** 2)
        for o in objects:
            # Check distance to arm start
            distance = math.sqrt((arm[0][0] - o[0]) ** 2 + (arm[0][1] - o[1]) ** 2)
            padding = o[2]
            if not isGoal:
                padding += arm[2]
            if padding >= distance:
                return True
            # Check distance to arm end
            distance = math.sqrt((arm[1][0] - o[0]) ** 2 + (arm[1][1] - o[1]) ** 2)
            padding = o[2]
            if not isGoal:
                padding += arm[2]
            if padding >= distance:
                return True
            # Find intersection of the two orthogonal lines
            if arm[1][0] == arm[0][0]:
                x = arm[0][0]
                y = o[1]
            elif arm[0][1] == arm[1][1]:
                x = o[0]
                y = arm[0][1]
            else:
                slope = 1.0 * (arm[1][1] - arm[0][1]) / (arm[1][0] - arm[0][0])
                perpendicular = -1 / slope
                x = (slope * arm[0][0] - arm[0][1] - perpendicular * o[0] + o[1]) / (slope - perpendicular)
                y = slope * (x - arm[0][0]) + arm[0][1]
            # Check if intersection point is actually part of the arm
            if arm[0][0] >= arm[1][0]:
                if x > arm[0][0] or x < arm[1][0]:
                    continue
            else:
                if x < arm[0][0] or x > arm[1][0]:
                    continue
            if arm[0][1] >= arm[1][1]:
                if y > arm[0][1] or y < arm[1][1]:
                    continue
            else:
                if y < arm[0][1] or y > arm[1][1]:
                    continue
            distance = math.sqrt((x - o[0]) ** 2 + (y - o[1]) ** 2)
            padding = o[2]
            if not isGoal:
                padding += arm[2]
            if padding >= distance:
                return True
    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touches goals.
    
    Args:
        armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
        goals (list): x-, y- coordinates and radius of goals [(x, y, r)]. There
            can be more than one goal.
    
    Returns:
        True if arm tip touches any goal. False if not.
    """
    for g in goals:
        distance = math.sqrt((armEnd[0] - g[0]) ** 2 + (armEnd[1] - g[1]) ** 2)
        if distance <= g[2]:
            return True
    return False

def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window.
    
    Args:
        armPos (list): start and end positions of all arm links [(start, end)]
        window (tuple): (width, height) of the window
    
    Returns:
        True if all parts are in the window. False if not.
    """
    for a in armPos:
        if a[0][0] < 0 or a[0][0] > window[0]:
            return False
        if a[1][0] < 0 or a[1][0] > window[0]:
            return False
        if a[0][1] < 0 or a[0][1] > window[1]:
            return False
        if a[1][1] < 0 or a[1][1] > window[1]:
            return False
    return True

if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190), 100, 20), ((150, 190), 100, 40), ((150, 190), 100, 60), ((150, 190), 100, 160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testResults = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testResults == resultComputeCoordinate

    testArmPosDists = [((100, 100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [True, True, False, True, False, True, False, True, False, True, False, True, False, False, False, True]
    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
    
    print("\n")
    
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
    
    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]
    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100, 100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    
    assert resultIsArmWithinWindow == testResults

    print("Test passed")
