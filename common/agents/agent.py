import random
from common.base_agent import BaseAgent
from common.move import Move
from common.agents.astar_algo import get_next_move_to_target

# Student scipers - replace with your actual SCIPER numbers
SCIPERS = ["380451", "445566"]

class Agent(BaseAgent):
    """
    Agent implementation that balances passenger collection with delivery, using A* for pathfinding and tactical decision making.
    
    The agent uses several key strategies:
    1. State tracking for position, path, and stuck detection
    2. Fixed parameters for wagon management and opponent avoidance
    3. High-value zone tracking for optimized collection
    4. Blocked zone tracking to avoid problematic areas
    """
    def __init__(self, nickname, network, logger="client.agent", timeout=None):
        super().__init__(nickname, network, logger)
        self.target = None
        self.path = []
        self.stuck_counter = 0
        self.last_position = None
        self.delivery_mode = False
        self.timeout = timeout
        
        # Fixed parameters for optimal performance
        self.max_preferred_wagons = 3
        self.min_drop_wagons = 1
        self.opponent_avoidance_distance = 3
        self.opportunistic_threshold = 1
        self.high_value_threshold = 7
        
        self.high_value_zones = []
        self.blocked_zones = set()
        
    def get_move(self):
        """
        Main decision function that balances passenger collection with delivery, using tactical pathfinding and opponent avoidance.
        
        The decision process:
        1. Waits for game initialization
        2. Handles dead train state
        3. Updates position history and stuck detection
        4. Updates high-value zones
        5. Handles stuck situations
        6. Manages delivery mode transitions
        7. Implements wagon dropping strategy
        8. Handles opportunistic collection
        9. Avoids opponents
        10. Makes mode-specific decisions (delivery vs collection)
        """
        if not hasattr(self, 'all_trains') or not self.all_trains or self.nickname not in self.all_trains:
            return Move.RIGHT
            
        my_train = self.all_trains[self.nickname]
        
        if not my_train["alive"]:
            return Move.RIGHT
            
        position = my_train["position"]
        direction = my_train["direction"]
        wagons = my_train["wagons"]
        wagons_count = len(wagons)
        
        try:
            if self.last_position is None:
                self.last_position = position
                
            if position == self.last_position:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
            self.last_position = position
            
            self.update_high_value_zones()
            
            if self.stuck_counter > 5:
                self.logger.debug("Detected stuck situation based on counter")
                return self.simple_unstuck_move(position, direction)
            
            delivery_zone = self.get_delivery_zone_rect()
            distance_to_delivery = float('inf')
            if delivery_zone:
                delivery_target = self.find_delivery_zone_target(delivery_zone)
                if delivery_target:
                    distance_to_delivery = self.manhattan_distance(position, delivery_target)
            
            if not self.delivery_mode and wagons_count >= self.max_preferred_wagons:
                self.delivery_mode = True
                self.target = None
                self.logger.debug("Switching to delivery mode")
                
            if self.delivery_mode and wagons_count == 0:
                self.delivery_mode = False
                self.target = None
                self.logger.debug("Switching to collection mode")
            
            if wagons_count >= self.min_drop_wagons:
                if self.should_drop_wagons(wagons_count, distance_to_delivery):
                    self.logger.debug(f"Tactical wagon drop, wagons: {wagons_count}")
                    return Move.DROP
                
            high_value_passenger = self.find_opportunistic_target(position, wagons_count)
            if high_value_passenger and wagons_count < 8:
                self.logger.debug("Opportunistically targeting high-value passenger")
                next_move = self.get_tactical_move(position, direction, high_value_passenger, True)
                if next_move:
                    return next_move
            
            avoid_move = self.avoid_opponents(position, direction)
            if avoid_move:
                self.logger.debug(f"Avoiding opponent train with move: {avoid_move}")
                return avoid_move
                
            if self.delivery_mode and wagons_count > 0:
                target = self.find_delivery_zone_target(delivery_zone)
                if not target:
                    move = self.avoid_obstacles(position, direction)
                    return self.ensure_move_enum(move)
                
                next_move = self.get_tactical_move(position, direction, target, False)
                if next_move:
                    return next_move
                    
                move = self.navigate_to_target(position, direction, target)
                return self.ensure_move_enum(move)
            else:
                target = self.find_optimal_passenger(position)
                if target:
                    self.logger.debug("Targeting passenger")
                    next_move = self.get_tactical_move(position, direction, target, True)
                    if next_move:
                        return next_move
                    move = self.navigate_to_target(position, direction, target)
                    return self.ensure_move_enum(move)
                    
            move = self.avoid_obstacles(position, direction)
            return self.ensure_move_enum(move)
            
        except Exception as e:
            self.logger.error(f"Error in get_move: {e}")
            return Move.RIGHT
    
    def ensure_move_enum(self, move):
        """
        Ensure we always return a valid Move enum, preventing invalid moves from being executed.
        
        The process:
        1. Checks if move is already a valid Move enum
        2. Tries to convert based on value if not
        3. Falls back to RIGHT as a safe default
        """
        if move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT, Move.DROP]:
            return move
        for move_enum in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            if move_enum.value == move:
                return move_enum
        return Move.RIGHT
        
    def avoid_opponents(self, position, current_direction):
        """
        Simple opponent avoidance that scores moves based on distance from other trains.
        
        The process:
        1. Gets available moves from current position
        2. Scores each move based on distance from opponents
        3. Only considers relatively close opponents
        4. Returns the move with highest safety score
        """
        if not hasattr(self, 'all_trains') or not self.all_trains:
            return None
            
        cell_size = self.cell_size or 20
        if cell_size == 0:
            cell_size = 20
            
        available_moves = self.get_available_moves(position, current_direction)
        if not available_moves:
            return None
            
        move_safety_scores = {move: 0 for move in available_moves}
        nearby_opponent_count = 0
        
        for train_name, train in self.all_trains.items():
            if train_name == self.nickname or not train["alive"]:
                continue
                
            opponent_pos = train["position"]
            distance = self.manhattan_distance(position, opponent_pos) / cell_size
            
            if distance <= self.opponent_avoidance_distance:
                nearby_opponent_count += 1
                
                for move in available_moves:
                    next_pos = self.get_next_position(position, move.value)
                    if next_pos is None:
                        continue
                    
                    move_distance = self.manhattan_distance(next_pos, opponent_pos) / cell_size
                    move_safety_scores[move] += move_distance
        
        if nearby_opponent_count > 0:
            best_move = max(move_safety_scores.items(), key=lambda x: x[1])[0]
            return best_move
        
        return None
        
    def update_high_value_zones(self):
        """
        Track areas with frequent high-value passengers to optimize collection strategy.
        
        The process:
        1. Validates passenger data
        2. Adds high-value passengers to zones
        3. Cleans up old zones to maintain recent data
        """
        if not hasattr(self, 'passengers') or not self.passengers:
            return
            
        current_time = self.game_time if hasattr(self, 'game_time') else 0
        
        for passenger in self.passengers:
            if "position" in passenger and "value" in passenger:
                pos = passenger["position"]
                
                if not pos or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                    continue
                
                pos = tuple(pos)
                value = passenger["value"]
                
                if value >= self.high_value_threshold:
                    self.high_value_zones.append({
                        "position": pos,
                        "value": value,
                        "time": current_time
                    })
        
        if hasattr(self, 'game_time'):
            self.high_value_zones = [zone for zone in self.high_value_zones 
                                  if self.game_time - zone["time"] < 15]
        
    def get_delivery_zone_rect(self):
        """
        Extract delivery zone coordinates, ensuring valid format for pathfinding.
        
        The process:
        1. Validates delivery zone existence
        2. Checks position format
        3. Validates dimensions
        4. Returns tuple of coordinates
        """
        if not hasattr(self, 'delivery_zone') or not self.delivery_zone:
            return None
            
        position_data = self.delivery_zone.get("position")
        if not position_data or not isinstance(position_data, (list, tuple)) or len(position_data) != 2:
            self.logger.warning(f"Invalid delivery zone position format: {position_data}")
            return None
            
        x, y = tuple(position_data)
        
        width = self.delivery_zone.get("width", 0)
        height = self.delivery_zone.get("height", 0)
        
        if width <= 0 or height <= 0:
            self.logger.warning(f"Invalid delivery zone dimensions: width={width}, height={height}")
            return None
            
        return (x, y, width, height)
        
    def find_delivery_zone_target(self, delivery_zone):
        """
        Find optimal target point in delivery zone, preferring center for efficient delivery.
        
        The process:
        1. Calculates center point
        2. Snaps to grid
        3. Returns target coordinates
        """
        if not delivery_zone:
            return None
            
        x, y, width, height = delivery_zone
        
        target_x = x + width // 2
        target_y = y + height // 2
        
        cell_size = self.cell_size or 20
        if cell_size == 0:
            cell_size = 20
            
        target_x = (target_x // cell_size) * cell_size
        target_y = (target_y // cell_size) * cell_size
        
        return (target_x, target_y)
        
    def navigate_to_target(self, position, current_direction, target):
        """
        Simple navigation that prioritizes the axis with larger distance to target.
        
        The process:
        1. Calculates direction vector
        2. Determines primary and secondary moves
        3. Tries primary move first
        4. Falls back to secondary move
        5. Uses random valid move as last resort
        """
        if not target:
            return self.avoid_obstacles(position, current_direction)
            
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        
        primary_move = None
        secondary_move = None
        
        if abs(dx) > abs(dy):
            primary_move = Move.RIGHT if dx > 0 else Move.LEFT
            secondary_move = Move.DOWN if dy > 0 else Move.UP
        else:
            primary_move = Move.DOWN if dy > 0 else Move.UP
            secondary_move = Move.RIGHT if dx > 0 else Move.LEFT
            
        if self.is_valid_move(position, primary_move.value, current_direction):
            return primary_move
            
        if self.is_valid_move(position, secondary_move.value, current_direction):
            return secondary_move
            
        available_moves = self.get_available_moves(position, current_direction)
        if available_moves:
            return random.choice(available_moves)
            
        return self.direction_value_to_move(current_direction)
        
    def avoid_obstacles(self, position, current_direction):
        """
        Choose a direction that avoids obstacles, with random choice if multiple options.
        
        The process:
        1. Projects forward to check for obstacles
        2. Finds available moves if next position is unsafe
        3. Chooses randomly from available moves
        4. Continues in current direction if safe
        """
        next_pos = self.get_next_position(position, current_direction)
        
        if next_pos is None or not self.is_position_safe(next_pos):
            available_moves = self.get_available_moves(position, current_direction)
            
            if available_moves:
                return random.choice(available_moves)
        else:
            return self.direction_value_to_move(current_direction)
            
        return self.direction_value_to_move(current_direction)
    
    def get_available_moves(self, position, current_direction):
        """
        Get list of valid moves from current position, excluding reversing.
        
        The process:
        1. Checks each possible direction
        2. Excludes opposite direction
        3. Validates moves
        """
        available_moves = []
        
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            if self.is_valid_move(position, move.value, current_direction):
                available_moves.append(move)
                
        return available_moves
        
    def is_valid_move(self, position, new_direction, current_direction):
        """
        Check if a move is valid by ensuring it's not reversing and doesn't hit obstacles.
        
        The process:
        1. Checks for reversing
        2. Validates new position
        3. Checks for obstacles
        """
        if (new_direction[0] == -current_direction[0] and 
            new_direction[1] == -current_direction[1]):
            return False
            
        new_pos = self.get_next_position(position, new_direction)
        if new_pos is None:
            return False
            
        return self.is_position_safe(new_pos)
        
    def get_next_position(self, position, direction):
        """
        Calculate next position based on current position and direction, respecting cell size.
        
        The process:
        1. Validates inputs
        2. Applies direction vector
        3. Respects cell size
        """
        if not position or not direction:
            return None
            
        cell_size = self.cell_size or 20
        if cell_size == 0:
            cell_size = 20
            
        x = position[0] + direction[0] * cell_size
        y = position[1] + direction[1] * cell_size
        return (x, y)
        
    def is_position_safe(self, position):
        """
        Check if a position is safe by validating boundaries and checking for collisions.
        
        The process:
        1. Validates position format
        2. Checks game boundaries
        3. Checks for train collisions
        4. Checks for wagon collisions
        """
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return False
            
        position = tuple(position)
            
        if not hasattr(self, 'game_width') or not hasattr(self, 'game_height') or not hasattr(self, 'all_trains'):
            return True
            
        if (position[0] < 0 or position[0] >= self.game_width or
            position[1] < 0 or position[1] >= self.game_height):
            return False
            
        for train_name, train in self.all_trains.items():
            if not train["alive"]:
                continue
                
            train_pos = train["position"]
            if isinstance(train_pos, (list, tuple)) and len(train_pos) == 2:
                train_pos = tuple(train_pos)
                if position == train_pos:
                    return False
                
            for wagon_pos in train["wagons"]:
                if isinstance(wagon_pos, (list, tuple)) and len(wagon_pos) == 2:
                    wagon_pos = tuple(wagon_pos)
                    if position == wagon_pos:
                        return False
                    
        return True
        
    def direction_value_to_move(self, direction_value):
        """
        Convert direction tuple to Move enum for consistent move representation.
        
        The process:
        1. Validates input
        2. Maps direction to Move enum
        3. Falls back to RIGHT if invalid
        """
        if not direction_value:
            return Move.RIGHT
            
        if direction_value == Move.UP.value:
            return Move.UP
        elif direction_value == Move.RIGHT.value:
            return Move.RIGHT
        elif direction_value == Move.DOWN.value:
            return Move.DOWN
        elif direction_value == Move.LEFT.value:
            return Move.LEFT
        else:
            return Move.RIGHT
    
    def manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between positions for path cost estimation.
        
        The process:
        1. Validates both positions
        2. Converts to tuples
        3. Calculates distance
        """
        if (not pos1 or not isinstance(pos1, (list, tuple)) or len(pos1) != 2 or
            not pos2 or not isinstance(pos2, (list, tuple)) or len(pos2) != 2):
            return float('inf')
        
        pos1 = tuple(pos1)
        pos2 = tuple(pos2)
            
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_opportunistic_target(self, position, wagon_count):
        """
        Find high-value passengers worth targeting even during delivery.
        
        The process:
        1. Validates passenger data
        2. Checks position format
        3. Considers distance and wagon count
        4. Returns best target
        """
        if not hasattr(self, 'passengers') or not self.passengers:
            return None
            
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
            
        position = tuple(position)
            
        for passenger in self.passengers:
            if "position" not in passenger or "value" not in passenger:
                continue
                
            p_pos = passenger["position"]
            if not p_pos or not isinstance(p_pos, (list, tuple)) or len(p_pos) != 2:
                continue
                
            p_pos = tuple(p_pos)
                
            distance = self.manhattan_distance(position, p_pos)
            
            if distance < 100 and wagon_count < 4:
                return p_pos
                    
        return None
        
    def get_tactical_move(self, position, direction, target, is_passenger_collection):
        """
        Get next move using A* pathfinding, falling back to simple navigation if needed.
        
        The process:
        1. Validates all inputs
        2. Converts to tuples
        3. Tries A* pathfinding
        4. Falls back to simple navigation
        """
        if not target:
            return None
            
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
            
        if not direction or not isinstance(direction, (list, tuple)) or len(direction) != 2:
            return None
            
        if not target or not isinstance(target, (list, tuple)) or len(target) != 2:
            return None
            
        position = tuple(position)
        direction = tuple(direction)
        target = tuple(target)
            
        next_move = get_next_move_to_target(
            position,
            direction,
            target,
            self.is_position_safe,
            self.cell_size
        )
        
        if next_move:
            return next_move
            
        return self.navigate_to_target(position, direction, target)
        
    def simple_unstuck_move(self, position, direction):
        """
        Simple unstuck strategy that tries turning or dropping wagons.
        
        The process:
        1. Tries turning left
        2. Tries turning right
        3. Tries dropping wagon
        4. Continues forward as last resort
        """
        left_move = Move.turn_left(self.direction_value_to_move(direction))
        right_move = Move.turn_right(self.direction_value_to_move(direction))
        
        left_pos = self.get_next_position(position, left_move.value)
        right_pos = self.get_next_position(position, right_move.value)
        
        if left_pos and self.is_position_safe(left_pos):
            return left_move
            
        if right_pos and self.is_position_safe(right_pos):
            return right_move
            
        my_train = self.all_trains[self.nickname]
        if len(my_train["wagons"]) > 0:
            return Move.DROP
            
        return self.direction_value_to_move(direction)
        
    def add_to_blocked_zones(self, position):
        """
        Add position to blocked zones to avoid returning to problematic areas.
        
        The process:
        1. Validates position
        2. Adds to blocked zones
        3. Maintains size limit
        """
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return
            
        position = tuple(position)
        
        self.blocked_zones.add(position)
        
        if len(self.blocked_zones) > 20:
            if self.blocked_zones:
                self.blocked_zones.pop()
                
    def find_optimal_passenger(self, position):
        """
        Find optimal passenger using value/distance ratio and high-value zone bonuses.
        
        The process:
        1. Validates passenger data
        2. Calculates scores based on value/distance
        3. Applies high-value zone bonuses
        4. Returns best target
        """
        if not hasattr(self, 'passengers') or not self.passengers:
            return None
            
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            self.logger.warning(f"Invalid agent position format: {position}")
            return None
            
        position = tuple(position)
            
        passenger_scores = []
        cell_size = self.cell_size or 20
        
        for passenger in self.passengers:
            if "position" not in passenger or "value" not in passenger:
                continue
                
            p_pos = passenger["position"]
            p_value = passenger["value"]
            
            if not p_pos or not isinstance(p_pos, (list, tuple)) or len(p_pos) != 2:
                continue
                
            p_pos = tuple(p_pos)
            
            distance = self.manhattan_distance(position, p_pos)
            
            if distance == 0:
                score = p_value * 1000
            else:
                score = p_value / max(1, distance / cell_size)
                
            for zone in self.high_value_zones:
                zone_pos = zone.get("position")
                if not zone_pos or not isinstance(zone_pos, (list, tuple)) or len(zone_pos) != 2:
                    continue
                zone_pos = tuple(zone_pos)
                if self.manhattan_distance(p_pos, zone_pos) < cell_size*2:
                    score *= 1.2
                    break
                
            passenger_scores.append((p_pos, score))
            
        passenger_scores.sort(key=lambda x: x[1], reverse=True)
        if passenger_scores:
            best_pos = passenger_scores[0][0]
            return best_pos
            
        return None
        
    def should_drop_wagons(self, wagon_count, distance_to_delivery):
        """
        Determine if wagons should be dropped based on count and stuck status.
        
        The process:
        1. Checks for excessive wagons
        2. Checks stuck status
        3. Returns decision
        """
        if wagon_count > 8:
            return True
            
        if self.stuck_counter > 3 and wagon_count > 2:
            return True
            
        return False 