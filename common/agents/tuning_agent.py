import random
from common.base_agent import BaseAgent
from common.move import Move
from common.agents.astar_algo import get_next_move_to_target, find_safest_path

# Student scipers - replace with your actual SCIPER numbers
SCIPERS = ["380451", "445566"]

class Agent(BaseAgent):
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        # Initialize state tracking
        self.target = None
        self.path = []
        self.previous_positions = []
        self.stuck_counter = 0
        self.last_position = None
        self.delivery_mode = False
        self.stuck_detection_window = 10
        
        # Enhanced tracking for tactical play
        self.passenger_memory = {}  # Remember passenger values at positions
        self.high_value_zones = []  # Areas with frequent high-value passengers
        self.successful_paths = []  # Store paths that worked well
        self.passenger_spawn_points = set()  # Track where passengers tend to spawn
        self.opponent_history = {}  # Track opponent behaviors
        self.delivery_visits = 0  # Count successful deliveries
        self.last_score = 0  # Track score changes
        self.passenger_values_collected = []  # Track values of passengers we collect
        self.blocked_zones = set()  # Areas to temporarily avoid
        self.last_strategy_change = 0  # Time tracker for strategy shifts
        
        # Fixed parameter values
        self.max_preferred_wagons = 3  # Balance between speed and score potential
        self.min_drop_wagons = 1  # Min wagons to consider dropping
        self.opponent_avoidance_distance = 3  # Cell distance to consider avoiding other trains
        self.use_astar = True
        self.opponent_prediction_steps = 2
        
        # Advanced strategic parameters
        self.opportunistic_threshold = 1  # Value threshold for opportunistic collection
        self.defensive_mode = False  # Engage defensive routing when ahead
        self.aggressive_mode = False  # Engage aggressive routing when behind
        self.passenger_value_memory_time = 200  # How long to remember passenger values
        self.high_value_threshold = 7  # Minimum value to be considered high-value
        
    def get_move(self):
        """Main decision function for the agent"""
        # Wait until game is properly initialized
        if not hasattr(self, 'all_trains') or not self.all_trains or self.nickname not in self.all_trains:
            return Move.RIGHT
            
        # Get my train
        my_train = self.all_trains[self.nickname]
        
        # If train is dead, just continue (will be respawned automatically)
        if not my_train["alive"]:
            return Move.RIGHT
            
        # Get current train position and status
        position = my_train["position"]
        direction = my_train["direction"]
        wagons = my_train["wagons"]
        wagons_count = len(wagons)
        
        try:
            # Track score changes for strategy adjustment
            current_score = self.scores.get(self.nickname, 0) if hasattr(self, 'scores') else 0
            current_score - self.last_score
            self.last_score = current_score
            
            # Update passenger memory and spawn points
            self.update_passenger_memory()
            
            # Update opponent behavior analysis
            self.analyze_opponents()
            
            # Initialize position history on first run
            if self.last_position is None:
                self.last_position = position
                
            # Check if we're stuck by monitoring position changes
            if position == self.last_position:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
            self.last_position = position
            
            # Update position history for pattern detection
            self.previous_positions.append(position)
            if len(self.previous_positions) > self.stuck_detection_window:
                self.previous_positions.pop(0)
            
            # Enhanced stuck detection and recovery
            if self.is_stuck_in_pattern() or self.stuck_counter > 5:
                # Apply pattern-breaking strategy
                self.logger.debug("Breaking detected loop or stuck situation")
                return self.get_unstuck_move(position, direction)
            
            # Get delivery zone for distance calculation
            delivery_zone = self.get_delivery_zone_rect()
            distance_to_delivery = float('inf')
            if delivery_zone:
                delivery_target = self.find_delivery_zone_target(delivery_zone)
                if delivery_target:
                    distance_to_delivery = self.manhattan_distance(position, delivery_target)
            
            # Check if we're in delivery mode (have enough wagons or near max capacity)
            if not self.delivery_mode and wagons_count >= self.max_preferred_wagons:
                self.delivery_mode = True
                self.target = None  # Reset target to find delivery zone
                self.logger.debug("Switching to delivery mode")
                
            # If in delivery mode and we have no wagons, switch back to collection
            if self.delivery_mode and wagons_count == 0:
                self.delivery_mode = False
                self.target = None  # Reset target for passenger collection
                self.logger.debug("Switching to collection mode")
                
            # Update aggressive/defensive mode based on scores
            if hasattr(self, 'scores') and len(self.scores) > 1:
                my_score = self.scores.get(self.nickname, 0)
                max_opponent_score = max([s for name, s in self.scores.items() if name != self.nickname], default=0)
                
                # If we're significantly behind, become more aggressive
                if my_score < max_opponent_score * 0.7 and hasattr(self, 'game_time') and self.game_time > 60:
                    self.aggressive_mode = True
                    self.defensive_mode = False
                    self.logger.debug("Switching to aggressive mode")
                # If we're significantly ahead, play more defensively
                elif my_score > max_opponent_score * 1.3 and hasattr(self, 'game_time') and self.game_time > 60:
                    self.aggressive_mode = False
                    self.defensive_mode = True
                    self.logger.debug("Switching to defensive mode")
                else:
                    # Otherwise, balanced mode
                    self.aggressive_mode = False
                    self.defensive_mode = False
            
            # Enhanced wagon dropping strategy - more tactical and situation-aware
            if wagons_count >= self.min_drop_wagons:
                if self.should_drop_wagons(wagons_count, distance_to_delivery):
                    self.logger.debug(f"Tactical wagon drop, wagons: {wagons_count}, " +
                                    f"distance to delivery: {distance_to_delivery}")
                    return Move.DROP
                
            # Main decision logic with enhanced strategic awareness
            
            # Opportunistic collection - grab high-value passengers even in delivery mode
            high_value_passenger = self.find_opportunistic_target(position, wagons_count)
            if high_value_passenger and wagons_count < 8:
                self.logger.debug("Opportunistically targeting high-value passenger")
                # Use advanced tactical pathfinding to the high-value target
                next_move = self.get_tactical_move(position, direction, high_value_passenger, True)
                if next_move:
                    return next_move
            
            # Check for immediate collision dangers and avoid if necessary
            avoid_move = self.avoid_opponents(position, direction, wagons_count)
            if avoid_move:
                self.logger.debug(f"Avoiding opponent train with move: {avoid_move}")
                return avoid_move
                
            # Decision logic based on current mode, with additional strategic considerations
            if self.delivery_mode and wagons_count > 0:
                # In delivery mode - head to delivery zone using tactical navigation
                target = self.find_delivery_zone_target(delivery_zone)
                if not target:
                    # If we can't find a delivery target, fallback to avoiding obstacles
                    move = self.avoid_obstacles(position, direction)
                    # Ensure we return a Move enum member
                    return self.ensure_move_enum(move)
                
                # Use enhanced pathfinding with opponent prediction for safer navigation
                self.logger.debug("Using enhanced navigation to delivery zone")
                next_move = self.get_tactical_move(position, direction, target, False)
                if next_move:
                    return next_move
                    
                # If tactical navigation fails, fall back to simple navigation
                move = self.navigate_to_target(position, direction, target)
                # Ensure we return a Move enum member
                return self.ensure_move_enum(move)
            else:
                # In collection mode - find optimal passenger using enhanced selection
                target = self.find_optimal_passenger(position, direction)
                if target:
                    self.logger.debug("Targeting optimal passenger")
                    # Use enhanced tactical pathfinding to the passenger
                    next_move = self.get_tactical_move(position, direction, target, True)
                    if next_move:
                        return next_move
                    move = self.navigate_to_target(position, direction, target)
                    # Ensure we return a Move enum member
                    return self.ensure_move_enum(move)
                    
            # If no clear decision, avoid obstacles and continue
            move = self.avoid_obstacles(position, direction)
            # Ensure we return a Move enum member
            return self.ensure_move_enum(move)
            
        except Exception as e:
            self.logger.error(f"Error in get_move: {e}")
            # If any error occurs, return a safe default
            return Move.RIGHT
    
    def ensure_move_enum(self, move):
        """Make sure we always return a Move enum value, not a tuple or list"""
        if move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT, Move.DROP]:
            return move
        # If not a Move enum, try to convert based on value
        for move_enum in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            if move_enum.value == move:
                return move_enum
        # If all else fails, just go RIGHT as a safe default
        return Move.RIGHT
        
    def avoid_opponents(self, position, current_direction, wagons_count):
        """Enhanced opponent avoidance with tactical considerations"""
        if not hasattr(self, 'all_trains') or not self.all_trains:
            return None
            
        cell_size = self.cell_size or 20
        if cell_size == 0:
            cell_size = 20
            
        # Get all available moves from current position
        available_moves = self.get_available_moves(position, current_direction)
        if not available_moves:
            return None
            
        # Score each move based on safety from all opponents with enhanced criteria
        move_safety_scores = {move: 0 for move in available_moves}
        
        # Count nearby opponents to normalize scores
        nearby_opponent_count = 0
        
        # Look for nearby opponent trains with enhanced awareness
        for train_name, train in self.all_trains.items():
            # Skip our own train or dead trains
            if train_name == self.nickname or not train["alive"]:
                continue
                
            opponent_pos = train["position"]
            opponent_dir = train["direction"]
            opponent_wagons = len(train["wagons"])
            
            # Calculate distance to opponent
            distance = self.manhattan_distance(position, opponent_pos) / cell_size
            
            # Adjust avoidance distance based on train lengths
            effective_avoidance_distance = self.opponent_avoidance_distance
            
                
            # If opponent has many wagons, they're less maneuverable - we can be less cautious
            if opponent_wagons > 7:
                effective_avoidance_distance -= 1
                
            # Only consider relatively close opponents
            if distance <= effective_avoidance_distance:
                nearby_opponent_count += 1
                
                # More sophisticated opponent position prediction
                predicted_positions = self.predict_opponent_positions(opponent_pos, opponent_dir, opponent_wagons)
                
                # Score each move based on distance from this opponent's predicted positions
                for move in available_moves:
                    next_pos = self.get_next_position(position, move.value)
                    if next_pos is None:
                        continue
                    
                    # Calculate minimum distance to any predicted opponent position
                    min_distance = float('inf')
                    for pred_pos in predicted_positions:
                        d = self.manhattan_distance(next_pos, pred_pos) / cell_size
                        min_distance = min(min_distance, d)
                    
                    # Score this move (higher distance is better)
                    # Add extra points if it increases distance from opponent
                    if min_distance > distance:
                        move_safety_scores[move] += min_distance + 2.0
                    else:
                        move_safety_scores[move] += min_distance
        
        # If we found nearby opponents, choose the move with the highest safety score
        if nearby_opponent_count > 0:
            # Find the move with the highest safety score
            best_move = max(move_safety_scores.items(), key=lambda x: x[1])[0]
            return best_move
        
        # No immediate threats detected
        return None
        
    def get_delivery_zone_rect(self):
        """Extract the delivery zone rectangle coordinates"""
        if not hasattr(self, 'delivery_zone') or not self.delivery_zone:
            return None
            
        # Add defensive check for position format
        position_data = self.delivery_zone.get("position")
        if not position_data or not isinstance(position_data, (list, tuple)) or len(position_data) != 2:
            self.logger.warning(f"Invalid delivery zone position format: {position_data}")
            return None
            
        # Convert position to tuple to ensure consistent format
        x, y = tuple(position_data)
        
        # Add defensive checks for width and height
        width = self.delivery_zone.get("width", 0)
        height = self.delivery_zone.get("height", 0)
        
        if width <= 0 or height <= 0:
            self.logger.warning(f"Invalid delivery zone dimensions: width={width}, height={height}")
            return None
            
        return (x, y, width, height)
        
    def find_delivery_zone_target(self, delivery_zone):
        """Find a good target point in the delivery zone"""
        if not delivery_zone:
            return None
            
        x, y, width, height = delivery_zone
        
        # Aim for the center of the delivery zone
        target_x = x + width // 2
        target_y = y + height // 2
        
        # Snap to grid
        cell_size = self.cell_size or 20
        # Ensure cell_size is never zero
        if cell_size == 0:
            cell_size = 20
            
        target_x = (target_x // cell_size) * cell_size
        target_y = (target_y // cell_size) * cell_size
        
        return (target_x, target_y)
        
    def navigate_to_target(self, position, current_direction, target):
        """Navigate towards the target position, avoiding obstacles"""
        if not target:
            return self.avoid_obstacles(position, current_direction)
            
        # Calculate direction vector to target
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        
        # Determine primary and secondary movement directions
        primary_move = None
        secondary_move = None
        
        # Prioritize the axis with the larger distance
        if abs(dx) > abs(dy):
            # Prioritize horizontal movement
            primary_move = Move.RIGHT if dx > 0 else Move.LEFT
            secondary_move = Move.DOWN if dy > 0 else Move.UP
        else:
            # Prioritize vertical movement
            primary_move = Move.DOWN if dy > 0 else Move.UP
            secondary_move = Move.RIGHT if dx > 0 else Move.LEFT
            
        # Check if primary move is valid
        if self.is_valid_move(position, primary_move.value, current_direction):
            return primary_move
            
        # If primary move is not valid, try secondary
        if self.is_valid_move(position, secondary_move.value, current_direction):
            return secondary_move
            
        # If neither primary nor secondary are valid, find any valid move
        available_moves = self.get_available_moves(position, current_direction)
        if available_moves:
            return random.choice(available_moves)
            
        # If no valid moves, continue in current direction
        return self.direction_value_to_move(current_direction)
        
    def avoid_obstacles(self, position, current_direction):
        """Choose a direction that avoids obstacles"""
        # Project forward to see if we would hit something
        next_pos = self.get_next_position(position, current_direction)
        
        # If can't determine next position or it's not safe, find another direction
        if next_pos is None or not self.is_position_safe(next_pos):
            # Find available moves
            available_moves = self.get_available_moves(position, current_direction)
            
            # If we have available moves, choose one randomly
            if available_moves:
                return random.choice(available_moves)
        else:
            # If the next position is safe, keep going
            return self.direction_value_to_move(current_direction)
            
        # If no available moves, just continue (we'll probably crash)
        return self.direction_value_to_move(current_direction)
    
    def get_available_moves(self, position, current_direction):
        """Get list of available moves from current position"""
        available_moves = []
        
        # Check each possible direction
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            # Don't consider the opposite direction (trains can't reverse)
            if self.is_valid_move(position, move.value, current_direction):
                available_moves.append(move)
                
        return available_moves
        
    def is_valid_move(self, position, new_direction, current_direction):
        """Check if the move is valid (not reversing and not hitting obstacle)"""
        # Can't reverse direction
        if (new_direction[0] == -current_direction[0] and 
            new_direction[1] == -current_direction[1]):
            return False
            
        # Check if the new position would be safe
        new_pos = self.get_next_position(position, new_direction)
        if new_pos is None:
            return False
            
        return self.is_position_safe(new_pos)
        
    def get_next_position(self, position, direction):
        """Calculate the next position given current position and direction"""
        if not position or not direction:
            return None
            
        cell_size = self.cell_size or 20  # Default to 20 if not set
        # Ensure cell_size is never zero
        if cell_size == 0:
            cell_size = 20
            
        x = position[0] + direction[0] * cell_size
        y = position[1] + direction[1] * cell_size
        return (x, y)
        
    def is_position_safe(self, position):
        """Check if a position is safe (no collision with walls, trains, or wagons)"""
        # Validate position format
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return False
            
        # Convert to tuple for consistency
        position = tuple(position)
            
        # Check if required properties are available
        if not hasattr(self, 'game_width') or not hasattr(self, 'game_height') or not hasattr(self, 'all_trains'):
            return True  # Can't check properly, assume safe
            
        # Check game boundaries
        if (position[0] < 0 or position[0] >= self.game_width or
            position[1] < 0 or position[1] >= self.game_height):
            return False
            
        # Check collision with other trains and wagons
        for train_name, train in self.all_trains.items():
            # Skip if the train is dead
            if not train["alive"]:
                continue
                
            # Check collision with train head
            train_pos = train["position"]
            if isinstance(train_pos, (list, tuple)) and len(train_pos) == 2:
                train_pos = tuple(train_pos)
                if position == train_pos:
                    return False
                
            # Check collision with wagons
            for wagon_pos in train["wagons"]:
                if isinstance(wagon_pos, (list, tuple)) and len(wagon_pos) == 2:
                    wagon_pos = tuple(wagon_pos)
                    if position == wagon_pos:
                        return False
                    
        return True
        
    def direction_value_to_move(self, direction_value):
        """Convert a direction value tuple to a Move enum"""
        if not direction_value:
            return Move.RIGHT  # Default if no direction
            
        if direction_value == Move.UP.value:
            return Move.UP
        elif direction_value == Move.RIGHT.value:
            return Move.RIGHT
        elif direction_value == Move.DOWN.value:
            return Move.DOWN
        elif direction_value == Move.LEFT.value:
            return Move.LEFT
        else:
            return Move.RIGHT  # Default
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        # Validate both positions
        if (not pos1 or not isinstance(pos1, (list, tuple)) or len(pos1) != 2 or
            not pos2 or not isinstance(pos2, (list, tuple)) or len(pos2) != 2):
            return float('inf')  # If either position is invalid, return infinite distance
        
        # Convert both to tuples for consistency
        pos1 = tuple(pos1)
        pos2 = tuple(pos2)
            
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_stuck_in_pattern(self):
        """Detect if we're stuck in a movement pattern"""
        # Need enough position history
        if len(self.previous_positions) < self.stuck_detection_window:
            return False
            
        # Check for repeating patterns in our position history
        # Look for patterns of length 2, 3, and 4
        for pattern_length in [2, 3, 4]:
            if len(self.previous_positions) >= pattern_length * 2:
                # Get the latest positions equal to pattern length
                latest = self.previous_positions[-pattern_length:]
                previous = self.previous_positions[-(pattern_length*2):-pattern_length]
                
                # If latest matches previous, we're in a loop
                if latest == previous:
                    return True
                    
        return False 

    def update_passenger_memory(self):
        """Update memory of passenger locations and values"""
        if not hasattr(self, 'passengers') or not self.passengers:
            return
            
        # Update passenger spawn points and values
        current_time = self.game_time if hasattr(self, 'game_time') else 0
        
        # Track passenger positions and values
        for passenger in self.passengers:
            if "position" in passenger and "value" in passenger:
                pos = passenger["position"]
                
                # Validate position format
                if not pos or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                    continue
                
                # Convert to tuple for hashability
                pos = tuple(pos)
                value = passenger["value"]
                
                # Add to passenger spawn points set for future reference
                self.passenger_spawn_points.add(pos)
                
                # Remember this position and its value with a timestamp
                self.passenger_memory[pos] = {
                    "value": value,
                    "last_seen": current_time
                }
                
                # If high value, add to high value zones
                if value >= self.high_value_threshold:
                    self.high_value_zones.append({
                        "position": pos,  # Already converted to tuple above
                        "value": value,
                        "time": current_time
                    })
        
        # Clean up old high value zones (keep only recent ones)
        if hasattr(self, 'game_time'):
            self.high_value_zones = [zone for zone in self.high_value_zones 
                                  if self.game_time - zone["time"] < 15]
                                  
    def analyze_opponents(self):
        """Analyze opponent behavior to inform strategic decisions"""
        if not hasattr(self, 'all_trains') or not self.all_trains:
            return
            
        for name, train in self.all_trains.items():
            # Skip our own train or dead trains
            if name == self.nickname or not train["alive"]:
                continue
                
            # Initialize tracking for this opponent if needed
            if name not in self.opponent_history:
                self.opponent_history[name] = {
                    "positions": [],
                    "wagon_counts": [],
                    "targeting_delivery": False,
                    "last_delivery_time": 0
                }
                
            # Validate and convert position
            train_pos = train["position"]
            if not train_pos or not isinstance(train_pos, (list, tuple)) or len(train_pos) != 2:
                continue
                
            # Convert to tuple for consistency
            train_pos = tuple(train_pos)
                
            # Track position and wagon count
            self.opponent_history[name]["positions"].append(train_pos)
            self.opponent_history[name]["wagon_counts"].append(len(train["wagons"]))
            
            # Limit history size
            max_history = 50
            if len(self.opponent_history[name]["positions"]) > max_history:
                self.opponent_history[name]["positions"].pop(0)
                self.opponent_history[name]["wagon_counts"].pop(0)
                
            # Detect if opponent is targeting delivery zone
            if hasattr(self, 'delivery_zone') and self.delivery_zone:
                delivery_target = self.find_delivery_zone_target(self.get_delivery_zone_rect())
                if delivery_target:
                    distance = self.manhattan_distance(train_pos, delivery_target)
                    # If opponent is close to delivery zone with wagons, they're probably targeting it
                    if distance < 100 and len(train["wagons"]) > 0:
                        self.opponent_history[name]["targeting_delivery"] = True
                    else:
                        self.opponent_history[name]["targeting_delivery"] = False

    def find_opportunistic_target(self, position, wagon_count):
        """Find high-value passengers that are worth targeting opportunistically"""
        if not hasattr(self, 'passengers') or not self.passengers:
            return None
            
        # Validate position format
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
            
        # Convert position to tuple for consistency
        position = tuple(position)
            
        # Only look for truly high-value opportunities
        for passenger in self.passengers:
            if "position" not in passenger or "value" not in passenger:
                continue
                
            # Validate passenger position
            p_pos = passenger["position"]
            if not p_pos or not isinstance(p_pos, (list, tuple)) or len(p_pos) != 2:
                continue
                
            # Convert to tuple for consistency
            p_pos = tuple(p_pos)
                
            # Check value and wagon count conditions
            distance = self.manhattan_distance(position, p_pos)
            
            # Only consider if reasonably close and less than 4 wagons
            if distance < 100 and wagon_count < 4:
                return p_pos
                    
        return None
        
    def get_tactical_move(self, position, direction, target, is_passenger_collection):
        """Get next move using advanced tactical pathfinding"""
        if not target:
            return None
            
        # Validate position, direction and target formats
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
            
        if not direction or not isinstance(direction, (list, tuple)) or len(direction) != 2:
            return None
            
        if not target or not isinstance(target, (list, tuple)) or len(target) != 2:
            return None
            
        # Convert all to tuples for consistency
        position = tuple(position)
        direction = tuple(direction)
        target = tuple(target)
            
        # Adjust pathfinding approach based on game situation
        # For passenger collection, we can be more aggressive
        if is_passenger_collection:
            prediction_steps = max(1, self.opponent_prediction_steps - 1)
        else:
            # When delivering, be more cautious
            prediction_steps = self.opponent_prediction_steps + 1
            
        # Use enhanced safety function that considers more factors
        enhanced_next_move = find_safest_path(
            position, 
            direction, 
            target, 
            self.is_tactically_safe,  # Enhanced safety function
            self.cell_size,
            self.all_trains,
            self.nickname,
            prediction_steps
        )
        
        if enhanced_next_move:
            return enhanced_next_move
            
        # Fallback to simpler pathfinding if enhanced fails
        return get_next_move_to_target(
            position,
            direction,
            target,
            self.is_position_safe,
            self.cell_size
        )
        
    def is_tactically_safe(self, position):
        """Enhanced safety check that considers strategic factors beyond simple collisions"""
        # Validate position format
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return False
            
        # Convert to tuple for hashability and consistent comparison
        position = tuple(position)
            
        # First do basic safety check
        if not self.is_position_safe(position):
            return False
            
        # Is this position in a temporarily blocked zone?
        if position in self.blocked_zones:
            return False
            
        # If we're in aggressive mode, take more risks
        if self.aggressive_mode:
            return True
            
        # Check for opponent trains that might be targeting the same area
        if hasattr(self, 'all_trains'):
            for name, train in self.all_trains.items():
                if name == self.nickname or not train["alive"]:
                    continue
                    
                # Check if opponent is likely to move to this position
                opponent_pos = train["position"]
                train["direction"]
                
                # Calculate distance to opponent train
                self.manhattan_distance(position, opponent_pos)
                
        return True
        
    def get_unstuck_move(self, position, direction):
        # Consider the full train length when planning turns
        my_train = self.all_trains[self.nickname]
        train_length = len(my_train["wagons"]) + 1  # +1 for the head
        
        # Try turns with more lookahead
        left_move = Move.turn_left(self.direction_value_to_move(direction))
        right_move = Move.turn_right(self.direction_value_to_move(direction))
        
        # Check if turns are safe with more lookahead
        left_safe = self.is_turn_safe(position, direction, left_move.value, train_length)
        right_safe = self.is_turn_safe(position, direction, right_move.value, train_length)
        
        if left_safe:
            return left_move
        if right_safe:
            return right_move
        
        # If neither turn works, drop a wagon
        if len(my_train["wagons"]) > 0:
            return Move.DROP
        
        return self.direction_value_to_move(direction)
        
    def add_to_blocked_zones(self, position):
        """Add a position to the blocked zones to avoid returning to it"""
        # Validate position format
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            return
            
        # Convert to tuple for hashability
        position = tuple(position)
        
        self.blocked_zones.add(position)
        
        # Ensure we don't accumulate too many blocked zones
        if len(self.blocked_zones) > 20:
            # Remove oldest blocked zone (approximately)
            if self.blocked_zones:
                self.blocked_zones.pop()
                
    def find_optimal_passenger(self, position, direction):
        """Find the optimal passenger using enhanced selection criteria"""
        if not hasattr(self, 'passengers') or not self.passengers:
            return None
            
        # Ensure position is in the right format
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            self.logger.warning(f"Invalid agent position format: {position}")
            return None
            
        # Convert position to tuple for consistency
        position = tuple(position)
            
        # Advanced passenger scoring that considers:
        # 1. Value of passenger
        # 2. Distance to passenger
        # 3. Whether passenger is in a known high-value zone
        # 4. Proximity to other valuable passengers (clustering)
        # 5. Whether other trains are heading toward the passenger
        
        passenger_scores = []
        cell_size = self.cell_size or 20
        
        for passenger in self.passengers:
            if "position" not in passenger or "value" not in passenger:
                continue
                
            p_pos = passenger["position"]
            p_value = passenger["value"]
            
            # Validate passenger position format
            if not p_pos or not isinstance(p_pos, (list, tuple)) or len(p_pos) != 2:
                continue
                
            # Convert to tuple for consistency
            p_pos = tuple(p_pos)
            
            # Basic score starts with value/distance ratio
            distance = self.manhattan_distance(position, p_pos)
            
            if distance == 0:
                base_score = p_value * 1000  # Very high score for passengers at our position
            else:
                base_score = p_value / max(1, distance / cell_size)
                
            # Apply modifiers
            score = base_score
            
            # Bonus if in known high-value zone
            for zone in self.high_value_zones:
                zone_pos = zone.get("position")
                if not zone_pos or not isinstance(zone_pos, (list, tuple)) or len(zone_pos) != 2:
                    continue
                zone_pos = tuple(zone_pos)
                if self.manhattan_distance(p_pos, zone_pos) < cell_size*2:
                    score *= 1.3
                    break
            
            # Look for passenger clusters (multiple passengers close together)
            cluster_value = 0
            for other in self.passengers:
                if other != passenger and "position" in other and "value" in other:
                    other_pos = other["position"]
                    if not other_pos or not isinstance(other_pos, (list, tuple)) or len(other_pos) != 2:
                        continue
                    other_pos = tuple(other_pos)
                    other_distance = self.manhattan_distance(p_pos, other_pos)
                    if other_distance < cell_size * 5:  # Within cluster range
                        cluster_value += other["value"] / max(1, other_distance / cell_size)
            
            # Add cluster bonus
            score += cluster_value * 0.2
            
            # Penalty if opponents are likely targeting this passenger
            for name, train in self.all_trains.items():
                if name != self.nickname and train["alive"]:
                    opponent_pos = train["position"]
                    if not opponent_pos or not isinstance(opponent_pos, (list, tuple)) or len(opponent_pos) != 2:
                        continue
                    opponent_pos = tuple(opponent_pos)
                    opp_to_passenger = self.manhattan_distance(opponent_pos, p_pos)
                    my_to_passenger = self.manhattan_distance(position, p_pos)
                    
                    # If opponent is closer and likely heading there
                    if opp_to_passenger < my_to_passenger * 0.7:
                        score *= 0.7  # Apply penalty
                
            passenger_scores.append((p_pos, score))
            
        # Get the best target
        passenger_scores.sort(key=lambda x: x[1], reverse=True)
        if passenger_scores:
            # Return the position of the highest-scoring passenger (already a tuple from above)
            best_pos = passenger_scores[0][0]
            self.logger.debug(f"Best passenger at position {best_pos} with score {passenger_scores[0][1]}")
            return best_pos
            
        return None
        
    def should_drop_wagons(self, wagon_count, distance_to_delivery):
        """Enhanced strategic wagon dropping decision"""
        # If we have massive number of wagons, consider dropping
        if wagon_count > 8:
            return True
            
        # If in danger of collision with long train, consider dropping to improve maneuverability
        in_danger = False
        if hasattr(self, 'all_trains') and self.all_trains and self.nickname in self.all_trains:
            my_pos = self.all_trains[self.nickname]["position"]
            for name, train in self.all_trains.items():
                if name != self.nickname and train["alive"]:
                    distance = self.manhattan_distance(my_pos, train["position"])
                    if distance < 80 and wagon_count > 10:
                        in_danger = True
                        
        if in_danger:
            return random.random() < 0.3
            
        # Base drop probability
        base_prob = 0.05
        
        # If we're close to delivery zone, increase probability
        if distance_to_delivery < 50:
            base_prob *= 4
        
        # If we have lots of wagons, increase probability
        wagon_factor = (wagon_count / 10)
        adjusted_prob = base_prob * wagon_factor
        
        # Make the decision
        return random.random() < adjusted_prob
        
    def predict_opponent_positions(self, position, direction, wagon_count):
        """Predict possible future positions of opponent with smart heuristics"""
        # Validate position and direction
        if not position or not isinstance(position, (list, tuple)) or len(position) != 2:
            self.logger.warning(f"Invalid opponent position format: {position}")
            return []
            
        if not direction or not isinstance(direction, (list, tuple)) or len(direction) != 2:
            self.logger.warning(f"Invalid opponent direction format: {direction}")
            return []
            
        # Convert to tuples for consistency
        position = tuple(position)
        direction = tuple(direction)
    
        predicted_positions = []
        cell_size = self.cell_size or 20
        
        # Predict straight line movement first (most likely)
        for i in range(1, self.opponent_prediction_steps + 1):
            pred_x = position[0] + direction[0] * cell_size * i
            pred_y = position[1] + direction[1] * cell_size * i
            predicted_positions.append((pred_x, pred_y))
            
        # For long trains, they're less likely to turn, so straight prediction is enough
        if wagon_count > 10:
            return predicted_positions
            
        # For shorter trains, consider possible turns as well
        current_direction_move = self.direction_value_to_move(direction)
        left_turn = Move.turn_left(current_direction_move).value
        right_turn = Move.turn_right(current_direction_move).value
        
        # Add potential positions after turns (with less weight for further steps)
        turn_pos_left = (position[0] + left_turn[0] * cell_size, 
                         position[1] + left_turn[1] * cell_size)
        turn_pos_right = (position[0] + right_turn[0] * cell_size, 
                          position[1] + right_turn[1] * cell_size)
                          
        # Add these potential turn positions
        predicted_positions.append(turn_pos_left)
        predicted_positions.append(turn_pos_right)
        
        return predicted_positions 