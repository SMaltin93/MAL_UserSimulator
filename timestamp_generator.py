import random
from datetime import datetime, timedelta

class TimestampGenerator:
    """Generate timestamps for simulation logs based on horizon parameter"""
    
    def __init__(self, target_date=None, horizon=100):
        """
        Initialize the timestamp generator
        
        Args:
            target_date (datetime): Starting date (default: today)
            horizon (int): Number of steps in the simulation
        """
        # Set target date (default to today)
        self.target_date = target_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Store simulation parameters - ensure horizon is an integer
        try:
            self.horizon = int(horizon)
        except (ValueError, TypeError):
            # If horizon is not convertible to int, use default
            self.horizon = 100
            print(f"Warning: Invalid horizon value. Using default: {self.horizon}")
        
        # Determine simulation days based on horizon
        if self.horizon <= 100:
            self.simulation_days = 1  #  simulation: single day
        elif self.horizon <= 500:
            self.simulation_days = 3  # simulation: spread over 3 days
        else:
            self.simulation_days = 14  # Long simulation
        
        # Calculate total logs to generate (horizon × logs per step)
        self.logs_per_step = 1
        self.total_logs = self.horizon * self.logs_per_step
        
        # distribution from analysis 
        self.hour_distribution = {
            7: 57,    # 7 AM
            9: 190,   # 9 AM
            10: 635,  # 10 AM
            11: 108,  # 11 AM
            13: 63,  # 1 PM 
            14: 351,  # 2 PM
            15: 693,  # 3 PM
            16: 77,  # 4 PM 
            17: 232   # 5 PM 
        }
        
        # Pre-generate all timestamps
        self.timestamps = self._generate_all_timestamps()
        self.current_index = 0
        
        # Create mapping of step to timestamp
        self.step_to_timestamp = {}
        self._map_steps_to_timestamps()
    
    def _generate_all_timestamps(self):
        """Generate all timestamps based on distribution"""
        all_timestamps = []
        total_distribution = sum(self.hour_distribution.values())
        
        # Calculate logs per day
        logs_per_day = self.total_logs // self.simulation_days
        
        # For each day in the simulation
        for day_offset in range(self.simulation_days):
            target_date = self.target_date + timedelta(days=day_offset)
            
            # Calculate logs for this day (last day gets any remainder)
            if day_offset == self.simulation_days - 1:
                day_logs = self.total_logs - (logs_per_day * (self.simulation_days - 1))
            else:
                day_logs = logs_per_day
            
            # For each hour in the distribution
            for hour, count in self.hour_distribution.items():
                # Scale by total_logs / total_distribution
                scaled_count = max(1, int((count / total_distribution) * day_logs))
                
                # Generate timestamps for this hour
                for _ in range(scaled_count):
                    # Random minute and second
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    microsecond = random.randint(0, 999999)
                    
                    # Create timestamp
                    timestamp = datetime(
                        target_date.year,
                        target_date.month,
                        target_date.day,
                        hour, minute, second, microsecond
                    )
                    
                    all_timestamps.append(timestamp)
        
        # Sort all timestamps
        all_timestamps.sort()
        
        # Make sure we have enough timestamps (at least one per step)
        while len(all_timestamps) < self.horizon:
            # Add more timestamps if needed
            last_date = self.target_date + timedelta(days=self.simulation_days-1)
            hour = random.choice(list(self.hour_distribution.keys()))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            microsecond = random.randint(0, 999999)
            
            timestamp = datetime(
                last_date.year,
                last_date.month,
                last_date.day,
                hour, minute, second, microsecond
            )
            all_timestamps.append(timestamp)
            
        # Re-sort after adding any extras
        all_timestamps.sort()
        
        return all_timestamps
    
    def _map_steps_to_timestamps(self):
        """Map each simulation step to a timestamp"""
        # Distribute timestamps evenly across steps
        if len(self.timestamps) < self.horizon:
            # Safety check to avoid index errors
            print(f"Warning: Not enough timestamps ({len(self.timestamps)}) for horizon ({self.horizon})")
            # Use modulo to cycle through available timestamps
            for step in range(self.horizon):
                self.step_to_timestamp[step] = self.timestamps[step % len(self.timestamps)]
        else:
            # Select timestamps at regular intervals to cover all steps
            indices = [int(i * len(self.timestamps) / self.horizon) for i in range(self.horizon)]
            for step, idx in enumerate(indices):
                self.step_to_timestamp[step] = self.timestamps[min(idx, len(self.timestamps)-1)]
    
    def get_timestamp_for_step(self, step):
        """Get timestamp for a specific simulation step"""
        if step in self.step_to_timestamp:
            return self.step_to_timestamp[step]
        
        # Fallback if step is not mapped
        if self.timestamps:
            # Use modulo to map any step to an existing timestamp
            return self.timestamps[step % len(self.timestamps)]
        else:
            # If no timestamps available, return start date + offset
            return self.target_date + timedelta(minutes=step)
    
    def get_next_timestamp(self):
        """Get next timestamp from pre-generated list"""
        if not self.timestamps:
            return self.target_date
            
        if self.current_index >= len(self.timestamps):
            # Reset if we've used all timestamps
            self.current_index = 0
            
        timestamp = self.timestamps[self.current_index]
        self.current_index += 1
        return timestamp
    
  
