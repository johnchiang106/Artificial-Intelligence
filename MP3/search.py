import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    #   - you can reuse the search code from mp3...
    # Your code here ---------------
    while frontier:
        current_state = heapq.heappop(frontier)

        if current_state.is_goal():
            # If find goal, call backtrack to retrieve the path
            return backtrack(visited_states, current_state)

        for neighbor in current_state.get_neighbors():
            neighbor_distance = neighbor.dist_from_start

            if neighbor not in visited_states or neighbor_distance < visited_states[neighbor][1]:
                # If neighbor is not visited or has a shorter distance from the start
                visited_states[neighbor] = (current_state, neighbor_distance)
                heapq.heappush(frontier, neighbor)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# You can reuse the backtracking code from MP3
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    current_state = goal_state

    while current_state is not None:
        path.append(current_state)
        current_state = visited_states[current_state][0]

    path.reverse()  # Reverse the list to maintain the correct order
    # ------------------------------
    return path