import heapq
from common.move import Move

class Node:
    """
    Node class to represent positions in the pathfinding graph, enabling efficient path cost tracking and comparison.
    
    Attributes:
        position: The (x,y) position of the node
        parent: Reference to the parent node in the path
        g: Cost from start to current node
        h: Heuristic (estimated cost from current to goal)
        f: Total cost (g + h)
    """
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __eq__(self, other):
        return self.position == other.position
        
    def __lt__(self, other):
        return self.f < other.f
        
    def __hash__(self):
        return hash(self.position)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance to provide a consistent and admissible heuristic for A* pathfinding"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_search(start_pos, goal_pos, is_position_safe_func, cell_size, current_direction=None):
    """
    A* pathfinding implementation to find optimal paths while respecting train movement constraints.
    Uses current_direction to prevent reversing, which trains cannot do.
    
    The algorithm:
    1. Creates start and end nodes
    2. Initializes open and closed lists
    3. Defines possible moves (UP, RIGHT, DOWN, LEFT)
    4. Maps directions to Move enums
    5. Uses a maximum iteration limit to prevent infinite loops
    6. Reconstructs the path when goal is reached
    7. Generates children nodes and calculates their costs
    8. Skips invalid moves and positions
    """
    start_node = Node(start_pos)
    start_node.g = 0
    start_node.h = manhattan_distance(start_pos, goal_pos) / cell_size
    start_node.f = start_node.h
    
    goal_node = Node(goal_pos)
    
    open_list = []
    closed_set = set()
    
    heapq.heappush(open_list, start_node)
    
    move_directions = [
        (0, -cell_size),  # UP
        (cell_size, 0),   # RIGHT
        (0, cell_size),   # DOWN
        (-cell_size, 0),  # LEFT
    ]
    
    dir_to_move = {
        (0, -cell_size): Move.UP,
        (cell_size, 0): Move.RIGHT,
        (0, cell_size): Move.DOWN,
        (-cell_size, 0): Move.LEFT
    }
    
    max_iterations = 1000
    iterations = 0
    
    while open_list and iterations < max_iterations:
        iterations += 1
        
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)
        
        if manhattan_distance(current_node.position, goal_node.position) <= cell_size:
            path = []
            node = current_node
            while node.parent:
                dx = node.position[0] - node.parent.position[0]
                dy = node.position[1] - node.parent.position[1]
                dir_tuple = (dx, dy)
                
                for direction, move in dir_to_move.items():
                    if direction == dir_tuple:
                        path.append(move)
                        break
                    
                node = node.parent
                
            path.reverse()
            return path
        
        for dx, dy in move_directions:
            new_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            
            if new_pos in closed_set or not is_position_safe_func(new_pos):
                continue
                
            if current_node.parent is None and current_direction:
                move_direction = (dx // cell_size, dy // cell_size)
                opposite_direction = (-current_direction[0], -current_direction[1])
                if move_direction == opposite_direction:
                    continue
            
            new_node = Node(new_pos, current_node)
            
            new_node.g = current_node.g + 1
            new_node.h = manhattan_distance(new_pos, goal_node.position) / cell_size
            new_node.f = new_node.g + new_node.h
            
            skip = False
            for open_node in open_list:
                if new_node == open_node and new_node.g >= open_node.g:
                    skip = True
                    break
            
            if not skip:
                heapq.heappush(open_list, new_node)
    
    return None

def get_next_move_to_target(current_pos, current_direction, target_pos, 
                           is_position_safe_func, cell_size):
    """
    Determine the next move to reach a target position using A* pathfinding.
    Falls back to simple navigation if A* fails, ensuring the train always has a move to make.
    
    The algorithm:
    1. Tries to find a path using A*
    2. If a path is found, returns the first move
    3. Otherwise falls back to simple navigation:
       - Calculates dx and dy to target
       - Determines primary and secondary directions
       - Prioritizes horizontal or vertical movement based on larger distance
       - Checks if moves are valid (not opposite to current direction)
       - Falls back to current direction if no valid moves
    """
    path = a_star_search(current_pos, target_pos, is_position_safe_func, 
                        cell_size, current_direction)
    
    if path and len(path) > 0:
        return path[0]
    
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    if abs(dx) > abs(dy):
        primary_move = Move.RIGHT if dx > 0 else Move.LEFT
        secondary_move = Move.DOWN if dy > 0 else Move.UP
    else:
        primary_move = Move.DOWN if dy > 0 else Move.UP
        secondary_move = Move.RIGHT if dx > 0 else Move.LEFT
    
    if (primary_move.value[0] == -current_direction[0] and 
        primary_move.value[1] == -current_direction[1]):
        if (secondary_move.value[0] == -current_direction[0] and 
            secondary_move.value[1] == -current_direction[1]):
            for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
                if move.value == current_direction:
                    return move
            return Move.RIGHT
        return secondary_move
    
    return primary_move

def find_safest_path(current_pos, current_direction, target_pos, 
                    is_position_safe_func, cell_size, all_trains, 
                    nickname, opponent_prediction_steps=3):
    """
    Find the safest path considering opponent train movements.
    Uses prediction to avoid potential collisions with other trains.
    
    The algorithm:
    1. Creates a modified safety check function that:
       - Checks basic safety first
       - Predicts opponent positions for future steps
       - Applies a certainty discount for further predictions
       - Avoids positions that match predicted opponent positions
    2. Tries to find a path using A* with the enhanced safety function
    3. Falls back to regular pathfinding if no safe path is found
    """
    def is_position_safe_with_predictions(position):
        if not is_position_safe_func(position):
            return False
        
        for step in range(1, opponent_prediction_steps + 1):
            for train_name, train in all_trains.items():
                if train_name == nickname or not train["alive"]:
                    continue
                
                opponent_pos = train["position"]
                opponent_dir = train["direction"]
                
                pred_x = opponent_pos[0] + (opponent_dir[0] * cell_size * step)
                pred_y = opponent_pos[1] + (opponent_dir[1] * cell_size * step)
                predicted_pos = (pred_x, pred_y)
                
                certainty = 1.0 - (step - 1) * 0.2
                
                if certainty > 0.2 and position == predicted_pos:
                    return False
        
        return True
    
    path = a_star_search(current_pos, target_pos, is_position_safe_with_predictions, 
                        cell_size, current_direction)
    
    if path and len(path) > 0:
        return path[0]
    
    return get_next_move_to_target(current_pos, current_direction, target_pos, 
                                  is_position_safe_func, cell_size) 