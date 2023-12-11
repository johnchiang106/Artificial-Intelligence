# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""

import heapq
from state import euclidean_distance


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP3/4 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    # Initialize the start state
    start_state = maze.get_start()

    # Initialize data structures for visited states and the priority queue
    visited_states = {start_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, start_state)
        
    while frontier:
        current_state = heapq.heappop(frontier)
        if current_state.dist_from_start != visited_states[current_state][1]:
            continue
            # current_state.set_dist_from_start(visited_states[current_state][1])

        # path = backtrack(visited_states, current_state)
        # if any(path_state in changed for path_state in path[:-2]):
        #     print("no")


        if current_state.is_goal(): # Find goal, retrieve the path
            return backtrack(visited_states, current_state)

        for neighbor in current_state.get_neighbors():
            neighbor_distance = neighbor.dist_from_start

            if neighbor not in visited_states:
                # If neighbor is not visited
                visited_states[neighbor] = (current_state, neighbor_distance)
                heapq.heappush(frontier, neighbor)
            elif neighbor_distance < visited_states[neighbor][1]:
                # If neighbor has a shorter distance from the start
                visited_states.pop(neighbor)
                # Recreate neighbor object, since its dist_from_start needs to be updated
                visited_states[neighbor] = (current_state, neighbor_distance)
                heapq.heappush(frontier, neighbor)

    # If the goal is not found, return None
    return None

# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, goal_state):
    path = []
    current_state = goal_state
    while current_state is not None:
        path.append(current_state)
        current_state = visited_states[current_state][0]
    path.reverse()  # Reverse the list to maintain the correct order
    return path

# def recalculateDistance(visited_states, path, changed):
#     new_distance = 0
#     start = False
#     last_state = None
#     for state in path:
#         if start:
#             new_distance += euclidean_distance(last_state, state)
#         elif last_state in changed:
#             # new_distance = visited_states[state][1]
#             new_distance = visited_states[last_state][1] + euclidean_distance(last_state, state)
#             start = True
#         last_state = state
#     return new_distance
