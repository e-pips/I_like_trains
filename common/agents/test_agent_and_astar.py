import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.agents.agent import Agent
from common.agents.astar_algo import (
    Node,
    manhattan_distance,
    a_star_search,
    get_next_move_to_target,
    find_safest_path
)
from common.move import Move

class TestAStarAlgorithm(unittest.TestCase):
    def setUp(self):
        """
        Setup test environment with a simple grid world to test pathfinding.
        
        The setup:
        1. Defines cell size for consistent testing
        2. Creates a bounded grid world (100x100)
        3. Implements position safety check for boundaries
        """
        self.cell_size = 20
        
        def is_position_safe(position):
            x, y = position
            return 0 <= x <= 100 and 0 <= y <= 100
            
        self.is_position_safe = is_position_safe
        
    def test_manhattan_distance(self):
        """
        Test distance calculation to ensure proper path cost estimation.
        
        The test cases:
        1. Simple diagonal distance (0,0) to (3,4)
        2. Equal distance in both axes (10,10) to (20,20)
        """
        pos1 = (0, 0)
        pos2 = (3, 4)
        self.assertEqual(manhattan_distance(pos1, pos2), 7)
        
        pos3 = (10, 10)
        pos4 = (20, 20)
        self.assertEqual(manhattan_distance(pos3, pos4), 20)
        
    def test_node_creation(self):
        """
        Test node comparison to ensure proper path graph construction.
        
        The test cases:
        1. Equal nodes at same position
        2. Different nodes at different positions
        """
        node1 = Node((0, 0))
        node2 = Node((0, 0))
        node3 = Node((1, 1))
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_a_star_search_simple_path(self):
        """
        Test basic pathfinding to ensure algorithm can find valid paths.
        
        The test case:
        1. Start at (0,0)
        2. Goal at (40,40) - 2 cells away in both directions
        3. Verifies path exists and has moves
        """
        start_pos = (0, 0)
        goal_pos = (40, 40)
        
        path = a_star_search(
            start_pos,
            goal_pos,
            self.is_position_safe,
            self.cell_size
        )
        
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        
    def test_a_star_search_no_path(self):
        """
        Test pathfinding failure to ensure algorithm handles blocked paths correctly.
        
        The test case:
        1. Creates a blocked area around goal
        2. Verifies no path is found when goal is unreachable
        """
        def blocked_safety(position):
            x, y = position
            return not (abs(x - 40) <= 20 and abs(y - 40) <= 20)
            
        start_pos = (0, 0)
        goal_pos = (40, 40)
        
        path = a_star_search(
            start_pos,
            goal_pos,
            blocked_safety,
            self.cell_size
        )
        
        self.assertIsNone(path)
        
    def test_get_next_move_to_target(self):
        """
        Test move selection to ensure proper direction choices.
        
        The test case:
        1. Current position at (0,0)
        2. Moving right (1,0)
        3. Target at (40,40)
        4. Verifies move is either right or down
        """
        current_pos = (0, 0)
        current_direction = (1, 0)
        target_pos = (40, 40)
        
        next_move = get_next_move_to_target(
            current_pos,
            current_direction,
            target_pos,
            self.is_position_safe,
            self.cell_size
        )
        
        self.assertIsNotNone(next_move)
        self.assertIn(next_move, [Move.RIGHT, Move.DOWN])
        
    def test_find_safest_path(self):
        """
        Test opponent avoidance to ensure safe path selection.
        
        The test case:
        1. Current position at (0,0)
        2. Moving right (1,0)
        3. Target at (40,40)
        4. Opponent at (20,20) moving right
        5. Verifies move avoids opponent
        """
        current_pos = (0, 0)
        current_direction = (1, 0)
        target_pos = (40, 40)
        
        all_trains = {
            "opponent1": {
                "position": (20, 20),
                "direction": (1, 0),
                "alive": True
            }
        }
        
        next_move = find_safest_path(
            current_pos,
            current_direction,
            target_pos,
            self.is_position_safe,
            self.cell_size,
            all_trains,
            "our_train"
        )
        
        self.assertIsNotNone(next_move)
        self.assertIn(next_move, [Move.RIGHT, Move.DOWN, Move.UP])

class TestSimplifiedAgent(unittest.TestCase):
    def setUp(self):
        """
        Setup test environment with minimal required game state.
        
        The setup:
        1. Creates mock network for game environment
        2. Initializes agent with test configuration
        3. Sets up basic game state (grid size, cell size)
        4. Creates mock train data for testing
        """
        class MockNetwork:
            def __init__(self):
                pass
                
        self.agent = Agent("test_agent", MockNetwork(), "test_agent")
        
        self.agent.cell_size = 20
        self.agent.game_width = 100
        self.agent.game_height = 100
        
        self.agent.all_trains = {
            "test_agent": {
                "position": (0, 0),
                "direction": (1, 0),
                "alive": True,
                "wagons": []
            }
        }
        
    def test_agent_initialization(self):
        """
        Test agent setup to ensure proper state initialization.
        
        The test cases:
        1. Verifies nickname is set
        2. Checks initial target is None
        3. Confirms stuck counter starts at 0
        4. Verifies last position is None
        5. Checks delivery mode starts as False
        """
        self.assertEqual(self.agent.nickname, "test_agent")
        self.assertIsNone(self.agent.target)
        self.assertEqual(self.agent.stuck_counter, 0)
        self.assertIsNone(self.agent.last_position)
        self.assertFalse(self.agent.delivery_mode)
        
    def test_manhattan_distance(self):
        """
        Test distance calculation to ensure proper target selection.
        
        The test case:
        1. Calculates distance between (0,0) and (3,4)
        2. Verifies result is 7 (3 + 4)
        """
        pos1 = (0, 0)
        pos2 = (3, 4)
        self.assertEqual(self.agent.manhattan_distance(pos1, pos2), 7)
        
    def test_is_position_safe(self):
        """
        Test boundary checking to ensure proper collision avoidance.
        
        The test cases:
        1. Valid position within bounds
        2. Invalid position below bounds
        3. Invalid position above bounds
        """
        self.assertTrue(self.agent.is_position_safe((20, 20)))
        
        self.assertFalse(self.agent.is_position_safe((-1, -1)))
        self.assertFalse(self.agent.is_position_safe((101, 101)))
        
    def test_get_available_moves(self):
        """
        Test move validation to ensure proper movement constraints.
        
        The test case:
        1. Position at (20,20)
        2. Moving right (1,0)
        3. Verifies available moves exclude reversing
        """
        position = (20, 20)
        current_direction = (1, 0)
        
        moves = self.agent.get_available_moves(position, current_direction)
        self.assertIsInstance(moves, list)
        self.assertTrue(len(moves) > 0)
        self.assertNotIn(Move.LEFT, moves)
        
    def test_should_drop_wagons(self):
        """
        Test wagon management to ensure proper capacity control.
        
        The test cases:
        1. Too many wagons (9)
        2. Stuck with moderate wagons (3)
        3. Normal case with few wagons (2)
        """
        self.assertTrue(self.agent.should_drop_wagons(9, 100))
        
        self.agent.stuck_counter = 4
        self.assertTrue(self.agent.should_drop_wagons(3, 100))
        
        self.assertFalse(self.agent.should_drop_wagons(2, 100))
        
    def test_get_move(self):
        """
        Test decision making to ensure proper behavior in different situations.
        
        The test cases:
        1. Initial move when game not initialized
        2. Move when train is dead
        3. Normal operation with passenger nearby
        """
        move = self.agent.get_move()
        self.assertEqual(move, Move.RIGHT)
        
        self.agent.all_trains["test_agent"]["alive"] = False
        move = self.agent.get_move()
        self.assertEqual(move, Move.RIGHT)
        
        self.agent.all_trains["test_agent"]["alive"] = True
        self.agent.passengers = [
            {"position": (40, 40), "value": 5}
        ]
        move = self.agent.get_move()
        self.assertIsNotNone(move)
        self.assertIn(move, [Move.RIGHT, Move.DOWN])

if __name__ == '__main__':
    unittest.main() 