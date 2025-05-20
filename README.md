# Train Agent Implementation

## Overview
This repository contains an intelligent train agent implementation for a train-based collection and delivery game. The agent uses sophisticated strategies to efficiently collect passengers and deliver them to designated zones while avoiding obstacles and optimizing performance.

## Key Features

### 1. Strategic Decision Making
The agent employs several key strategies:
- State tracking for position, path, and stuck detection
- Fixed parameters for wagon management
- High-value zone tracking for optimized collection
- Blocked zone tracking to avoid problematic areas
- Tactical pathfinding using A* algorithm

### 2. Core Components

#### Wagon Management
- Maintains optimal wagon count (max 3 preferred wagons)
- Implements tactical wagon dropping when:
  - Wagon count exceeds 8
  - Train is stuck for more than 3 moves with more than 2 wagons
  - Strategic dropping near delivery zones

#### Passenger Collection Strategy
- Prioritizes high-value passengers
- Uses value/distance ratio for target selection
- Maintains memory of high-value zones
- Implements opportunistic collection during delivery

#### Delivery Optimization
- Switches to delivery mode when wagon threshold is reached
- Centers delivery zone targeting for efficient unloading
- Balances collection and delivery priorities

#### Obstacle Avoidance
- Implements opponent avoidance with safety scoring
- Maintains safe distance from other trains
- Uses tactical pathfinding to navigate around obstacles
- Handles stuck situations with multi-strategy approach

### 3. Navigation System

#### Pathfinding
- Primary: A* algorithm for optimal path calculation
- Fallback: Manhattan distance-based simple navigation
- Tactical movement with obstacle consideration

#### Movement Decision Process
1. Validates game state and train status
2. Updates position history and stuck detection
3. Processes high-value zones
4. Handles stuck situations
5. Manages delivery mode transitions
6. Implements wagon dropping strategy
7. Processes opportunistic collection
8. Applies opponent avoidance
9. Executes mode-specific decisions

### 4. Safety Features
- Comprehensive position validation
- Boundary checking
- Collision avoidance with other trains and wagons
- Stuck detection and recovery
- Invalid move prevention

## Technical Details

### Parameters
```python
max_preferred_wagons = 3
min_drop_wagons = 1
opponent_avoidance_distance = 3
opportunistic_threshold = 1
high_value_threshold = 7
```

### Key Methods
- `get_move()`: Main decision function
- `navigate_to_target()`: Path planning
- `find_optimal_passenger()`: Target selection
- `avoid_opponents()`: Collision avoidance
- `get_tactical_move()`: A* pathfinding integration
- `update_high_value_zones()`: Zone tracking

## Performance Optimization
- Efficient pathfinding with A* algorithm
- Smart caching of high-value zones
- Tactical decision making based on current state
- Balanced resource management

## Error Handling
- Comprehensive input validation
- Fallback strategies for failed pathfinding
- Safe move verification
- Graceful degradation of functionality

## Usage
The agent is designed to work within the train game framework and requires initialization with:
- Nickname
- Network configuration
- Logger (optional)
- Timeout settings (optional)

## Dependencies
- Base game framework
- A* algorithm implementation
- Random module for tactical decisions

## Authors
SCIPER: 380451, 445566
