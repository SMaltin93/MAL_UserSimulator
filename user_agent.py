
import random
import logging
import json
import numpy as np
import re
from libexec.userAgent.timestamp_generator import TimestampGenerator
from datetime import datetime
from collections import defaultdict

from maltoolbox.attackgraph import AttackGraphNode
from malsim.mal_simulator import MalSimAgentStateView

logger = logging.getLogger(__name__)

class UserAgent:
    name = ' '.join(re.findall(r'[A-Z][^A-Z]*', __qualname__))

    def __init__(self, agent_config: dict) -> None:
        """Initialize agent that follows a transition matrix for state selection."""
        self.attack_graph = agent_config.pop('attack_graph')
        self.logs = []
        self.state_names = {}
        # Initialize states with empty list (will be populated from attack graph)
        self.states = []
        self.current_state_idx = 0
        self.agent_path = [self.current_state_idx]
        # Map state IDs to their available step IDs
        self.mapping = defaultdict(list)
        # Track allowed steps from action surface
        self.allowed_steps = []
        # Map step IDs to their names
        self.steps_name = {}
        # Current step being executed
        self.current_step = 0
       
        
        # Transition matrix defining probability of moving between states
        # this values can be changed to reflect the actual transition probabilities
        # Each row corresponds to a state, each column to a possible next state
        # m = number of states, n = number of possible next states - matrix is m x n = 11 x 11
        # the values in this matrix are as example and not that exakt based on any real data , read the report for more details
        # changing values mean different behavior of the agent
        self.transition_matrix = np.array([
            [0.00, 0.80, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Start -> [PublicContent, LoginProcess]
            [0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.60],  # PublicContent -> [Blog, Search]
            [0.30, 0.00, 0.00, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # LoginProcess -> [Start, Overview]
            [0.00, 0.20, 0.00, 0.00, 0.30, 0.30, 0.20, 0.00, 0.00, 0.00, 0.00],  # Overview -> [PublicContent, WatchList, TradingRelated, Account]
            [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # WatchList -> [Overview]
            [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # TradingRelated -> [Overview]
            [0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00],  # Account -> [Overview, Messages, PrivateData]
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],  # Messages -> [Account]
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],  # PrivateData -> [Account]
            [0.00, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00],  # Blog -> [PublicContent, Blog]
            [0.00, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40]   # Search -> [PublicContent, Search]
        ])
        
        #Process attack graph to populate mappings
        self.utilize_states_steps(self.attack_graph)
        
        # Set up timestamp generator
        target_date = agent_config.get('target_date', datetime(2025, 5, 12))  # this date can be changed
        self.timestamp_generator = TimestampGenerator(target_date) 
        
        print(f"Agent initialized with timestamps for {target_date.strftime('%Y-%m-%d')}")
        print(f"States mapping: {dict(self.mapping)}")
    
    def utilize_states_steps(self, attack_graph: AttackGraphNode):
        """Utilize states and steps from the attack graph."""
        for node, state in self.attack_graph.nodes.items():
            #print(f"step_id: {node}, State Index: {state.model_asset.id}")
            if state.model_asset.id not in self.state_names:
                self.state_names[state.model_asset.id] = state.model_asset.name
            # state.lg_attack_step.name with the attack step name
            if node not in self.steps_name:
                self.steps_name[node] = state.lg_attack_step.name
            # state.model_asset.id with the attack step ids
            if not state.model_asset.id in self.mapping:
                if node not in self.mapping[state.model_asset.id]:
                    self.mapping[state.model_asset.id].append(node)
            else:
                # check if the node is not already in the mapping
                if node not in self.mapping[state.model_asset.id]:
                    self.mapping[state.model_asset.id].append(node)
       #print(f"states {self.state_names} " )

    def get_state_from_step(self, step_id):
        """Get the state corresponding to a given step ID."""
        for state_id, steps in self.mapping.items():
            if step_id in steps:
                return state_id
        return None
    
            
    def get_next_state_id_based_transition_matrix(self, current_state):
        """Return the next state ID based on the transition matrix."""
        row = self.transition_matrix[current_state]
        # Check if row has any non-zero values / shouldnt has
        if np.sum(row) == 0:
            next_state_idx = random.randint(0, len(row) - 1)
            print(f"No valid transitions from current state, selecting random state: {next_state_idx}")
            return next_state_idx
        else:
            row = row / np.sum(row)
            next_state_idx = np.random.choice(len(row), p=row)
            print(f"Next state index based on transition matrix: {next_state_idx}")
            return next_state_idx
    
    def check_if_step_is_allowed(self, step_id):
        """Check if step ID is in allowed steps."""
        return step_id in self.allowed_steps
        
    def get_step_name_by_id(self, step_id):
        """Return step name for given step ID."""
        return self.steps_name.get(step_id, f"Unknown_Step_{step_id}")
    
    
    def policy_step_allowed(self, step_id):
        """Check if step is preferred according to policy."""
        step_name = self.get_step_name_by_id(step_id)
        if "stay" not in step_name.lower():
            return True
        else:
            print(f"Step {step_name} is not preferred")
            return False
        
    def get_state_name_by_id(self, state_id):
        return N
  
    def get_next_action(self, agent_state: MalSimAgentStateView, **kwargs):
        """Choose an action based on transition matrix priorities."""
        try:
            # Get available attack steps from action surface
            attack_surface = list(agent_state.action_surface)
            
            if not attack_surface:
                print("No attack surface available")
                return None
            print(f"Attack surface contains {len(attack_surface)} items")
            # Clear previous allowed steps and update with current ones
            self.allowed_steps = []
            
            # Process attack surface to get allowed steps
            for node in attack_surface:
                if node.id not in self.allowed_steps:
                    self.allowed_steps.append(node.id)
        
            print(f"Allowed steps: {self.allowed_steps}")
            
            # Choose next state based on transition matrix
            next_state_idx = self.get_next_state_id_based_transition_matrix(self.current_state_idx)
            
            # Get steps for the chosen state
            if self.mapping:
                state_steps = self.mapping[next_state_idx]
                print(f"State_steps {state_steps}")
            else: 
                print("Mapping is empty")
                
            # Find an allowed step from the chosen state
            chosen_step_id = None
            for step_id in state_steps:
                if step_id in self.allowed_steps and self.policy_step_allowed(step_id):
                    chosen_step_id = step_id
                    print(f"Selected step {chosen_step_id} ({self.get_step_name_by_id(chosen_step_id)})")
                    break

            # If no step from chosen state is allowed, try backtracking
            if chosen_step_id is None:
                print("No allowed steps in chosen state, trying backtracking")
                # Remove current state from path if it's not empty
                if len(self.agent_path) > 1:
                    self.agent_path.pop()
                    previous_state = self.agent_path[-1]
                    print(f"Backtracking to state {previous_state}")
                    self.current_state_idx = previous_state
                # Try to find any allowed step
                for step_id in self.allowed_steps:
                    if self.policy_step_allowed(step_id):
                        chosen_step_id = step_id
                        print(f"Backtracking selected step {chosen_step_id}")
                        break
                
                # If still no step, just pick any allowed step
                if chosen_step_id is None and self.allowed_steps:
                    chosen_step_id = random.choice(self.allowed_steps)
                    print(f"Go back to random step {chosen_step_id}")
            
            # Find the node in attack surface that corresponds to chosen step
            if chosen_step_id is not None:
                # Update current state and path
                state_of_step = self.get_state_from_step(chosen_step_id)
                if state_of_step is not None:
                    self.current_state_idx = state_of_step
                    self.agent_path.append(self.current_state_idx)
                    # Limit history to last 20 states
                    if len(self.agent_path) > 20:
                        self.agent_path = self.agent_path[-20:]
                # Generate log
                self._collect_logs()
                
                # Find and return the appropriate node
                for node in attack_surface:
                    if isinstance(node, tuple) and len(node) >= 2:
                        node_obj = node[1]
                        if hasattr(node_obj, 'id') and node_obj.id == chosen_step_id:
                            print(f"Returning action: {node}")
                            return node
                    elif hasattr(node, 'id') and node.id == chosen_step_id:
                        print(f"Returning action: {node}")
                        return node
            
            print("No valid action found")
            return None
            
        except Exception as e:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>> Error in get_next_action: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>< {e}")
            return None

    def _collect_logs(self):
        """Generate timestamped logs for the current state."""
        try:
            # Get timestamp from generator
            timestamp = self.timestamp_generator.get_next_timestamp()
            
            # Get current state name or index if name is not available
            if 0 <= self.current_state_idx < len(self.state_names):
                state_name = self.state_names[self.current_state_idx]
            else:
                state_name = f"State_{self.current_state_idx}"
            
            # Create log entry
            log_entry = {
                "timestamp": timestamp,
                "request_url": f"/{state_name}",
                "agent": self.__class__.__name__,
            }
            self.logs.append(log_entry)
            print(f"Log generated for {state_name} at {timestamp}")
        except Exception as e:
            print(f"Error collecting logs: {e}")

    def terminate(self):
        """Write logs to file."""
        print(f"Writing {len(self.logs)} logs to file")
        with open('user_logs.json', 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)
        print("Logs written successfully")
        self.logs = []
