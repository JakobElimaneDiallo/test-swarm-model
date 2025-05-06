import random
import numpy as np
from enum import Enum, auto
from scipy.stats import beta as beta_dist
import pandas as pd
import os

class RobotState(Enum):
    RANDOM_WALK = auto()
    COLLISION_AVOIDANCE = auto()
    GET_OBSERVATION = auto()
    SEND_SAMPLE = auto()
    RESET = auto()

class FeedbackStrategy(Enum):
    NONE = auto()
    POSITIVE = auto()
    SOFT = auto()

class SimpleRobot:
    def __init__(self, x, y, env=None, robot_id=None, message_queue=None):
        # Position
        self.x = x
        self.y = y
        self.env = env
        
        # Movement parameters
        self.speed = 0.05  # Fixed speed
        self.direction = random.uniform(0, 2 * np.pi)  # Random initial direction
        # Lévy flight scaling factor (median step length)
        self.levy_scale = 0.05
        
        # State machine parameters
        self.state = RobotState.RANDOM_WALK
        self.timer = 0
        self.tau = 0.2  # Time between observations (faster)
        self.random_walk_duration = 5.0  # Duration before random turn
        self.random_walk_timer = 0
        
        # Observation parameters
        self.last_observation = None
        self.observations = []
        
        # Beta distribution parameters for inference
        self.alpha = 1  # Prior parameter
        self.beta = 1   # Prior parameter
        self.belief_threshold = 0.5  # Threshold for CDF check
        
        # Decision parameters
        self.p_c = 0.95  # Confidence threshold
        self.decision_flag = -1  # -1: no decision, 0: decided vibrating, 1: decided non-vibrating
        self.min_swarm_count = 100  # Minimum number of observations before decision
        
        # Communication counters
        self.sends = 0
        self.recvs = 0
        
        # Feedback strategy
        self.feedback_strategy = FeedbackStrategy.POSITIVE
        self.eta = 1250  # Parameter for soft feedback
        self.us_exponent = 2.0
        
        # Random generators for soft feedback
        self.sf_rd = random.Random()
        self.sf_rd.seed(42)  # Set a fixed seed for reproducibility
        
        # Measurement logging
        self.robot_id = robot_id
        
        # Reference to global message queue
        self.message_queue = message_queue if message_queue is not None else []
        self.last_msg_index = 0  # Track processed messages
        
    def turnRandomAngle(self):
        self.direction = random.uniform(0, 2 * np.pi)
        
    def performMovement(self, dt):
        # Lévy flight: draw step length from a Cauchy (Lorentzian) distribution
        step_size = np.abs(np.random.standard_cauchy()) * self.levy_scale  # heavy-tailed
        # Limit extreme jumps to keep movement within the arena
        step_size = np.clip(step_size, 0.005, 0.3)
        
        # Calculate movement based on step size and direction
        dx = step_size * np.cos(self.direction)
        dy = step_size * np.sin(self.direction)
        
        # Using step_size as delta per update, no extra dt-scaling needed

        # Try changing direction up to 8 times if we would go outside the arena
        attempts = 0
        while attempts < 8:
            next_x = self.x + dx
            next_y = self.y + dy
            if 0 <= next_x <= 1 and 0 <= next_y <= 1:
                break  # valid step
            # Choose a new random direction and try again
            self.turnRandomAngle()
            dx = step_size * np.cos(self.direction)
            dy = step_size * np.sin(self.direction)
            attempts += 1
        else:
            # After 8 attempts still outside -> stay in place but rotate
            self.turnRandomAngle()
            return
            
        # First, update position
        self.x = next_x
        self.y = next_y
        
        # Update timers **after** movement so movement happens every call
        self.timer += dt
        self.random_walk_timer += dt
        
        # Check if it's time to take an observation
        if self.timer >= self.tau:
            self.state = RobotState.GET_OBSERVATION
            self.timer = 0
            return
            
        # Randomly change walking direction after some time
        if self.random_walk_timer >= self.random_walk_duration:
            self.turnRandomAngle()
            self.random_walk_timer = 0
            
    def getObservation(self):
        if self.env is not None:
            # Get current grid position
            grid_x = min(int(self.x * self.env.grid_size), self.env.grid_size - 1)
            grid_y = min(int(self.y * self.env.grid_size), self.env.grid_size - 1)
            return self.env.grid[grid_x, grid_y]
        return random.choice([True, False])
        
    def calculateMessage(self, sample):
        if self.feedback_strategy == FeedbackStrategy.NONE:
            return sample
            
        if self.feedback_strategy == FeedbackStrategy.POSITIVE:
            if self.decision_flag == -1:
                return sample
            # Map decision to message consistent with observation encoding:
            #   decision_flag == 0  -> majority vibrating  -> send 1
            #   decision_flag == 1  -> majority non-vibrating -> send 0
            return 1 if self.decision_flag == 0 else 0
            
        if self.feedback_strategy == FeedbackStrategy.SOFT:
            # Calculate delta based on belief variance and distance from threshold
            variance = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
            belief = self.alpha / (self.alpha + self.beta)
            delta = np.exp(-self.eta * variance) * abs(0.5 - belief)**self.us_exponent
            
            # Soft feedback probability
            p = delta * (1.0 - belief) + (1 - delta) * sample
            return int(random.random() < p)
            
        return sample
        
    def sendSample(self, message):
        # Update Beta distribution parameters based on observation
        if message == 1:
            self.alpha += 1
        else:
            self.beta += 1
        self.sends += 1
        
        # Record measurement when sending sample
        # Confidence measure: max(belief, 1 - belief)
        belief = beta_dist.cdf(0.5, self.alpha, self.beta)
        belief_conf = max(belief, 1.0 - belief)
        self.last_measurement = {
            'robot_id': int(self.robot_id) if self.robot_id is not None else None,
            'x': float(self.x),
            'y': float(self.y),
            'sample_value': int(message),
            'belief': float(belief_conf),
            'belief_conf': float(belief_conf)
        }
        
        # Broadcast message to global queue
        if self.message_queue is not None:
            self.message_queue.append((self.robot_id, message))
        
    def get_last_measurement(self):
        """Get and clear the last measurement"""
        if hasattr(self, 'last_measurement'):
            measurement = self.last_measurement
            del self.last_measurement
            return measurement
        return None
        
    def recvSample(self, message):
        # Update Beta distribution based on received message
        if message == 1:
            self.alpha += 1
        else:
            self.beta += 1
        self.recvs += 1
        
    def update(self, dt):
        # Process incoming messages from queue
        while self.last_msg_index < len(self.message_queue):
            sender_id, msg = self.message_queue[self.last_msg_index]
            if sender_id != self.robot_id:
                self.recvSample(msg)
            self.last_msg_index += 1
        
        # Re-evaluate decision after processing new messages
        self.check_decision()
        
        # Increment timer
        self.timer += dt
        
        if self.state == RobotState.RANDOM_WALK:
            self.performMovement(dt)
            
            # Check if it's time to get an observation
            if self.timer >= self.tau:
                self.state = RobotState.GET_OBSERVATION
            
        elif self.state == RobotState.COLLISION_AVOIDANCE:
            self.turnRandomAngle()
            self.state = RobotState.RANDOM_WALK
            
        elif self.state == RobotState.GET_OBSERVATION:
            self.last_observation = self.getObservation()
            self.observations.append(self.last_observation)
            self.state = RobotState.SEND_SAMPLE
            
        elif self.state == RobotState.SEND_SAMPLE:
            # Convert observation to binary message and apply feedback strategy
            raw_message = 1 if self.last_observation else 0
            message = self.calculateMessage(raw_message)
            
            # Send sample and update Beta distribution
            self.sendSample(message)
            
            # Check if decision can be made
            self.check_decision()
            
            # Return to random walk
            self.state = RobotState.RANDOM_WALK
            self.timer = 0
            
        elif self.state == RobotState.RESET:
            # Reset state and return to random walk
            self.state = RobotState.RANDOM_WALK
            self.timer = 0
            
    def check_decision(self):
        # Re-evaluate decision whenever we have enough (possibly correlated) samples.
        #   belief  = P(f < 0.5)
        #   high belief  -> confident that f < 0.5   (non-vibrating majority)
        #   low  belief  -> confident that f > 0.5   (vibrating majority)
        if (self.sends + self.recvs) >= self.min_swarm_count:
            belief = beta_dist.cdf(0.5, self.alpha, self.beta)
            if belief > self.p_c:
                self.decision_flag = 1  # confident: non-vibrating majority
            elif belief < (1 - self.p_c):
                self.decision_flag = 0  # confident: vibrating majority
            else:
                # Uncertainty back above (1-p_c): withdraw decision.
                self.decision_flag = -1
                
    def get_belief_confidence(self):
        # Calculate belief as cumulative probability P(f < 0.5)
        belief = beta_dist.cdf(0.5, self.alpha, self.beta)
        return max(belief, 1.0 - belief)
