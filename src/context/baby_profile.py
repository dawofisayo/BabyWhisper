"""Baby profile management for personalized cry interpretation."""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import os


class BabyProfile:
    """Store and manage baby-specific information for context-aware predictions."""
    
    def __init__(self, 
                 baby_name: str = "Baby",
                 birth_date: Optional[datetime] = None,
                 age_months: Optional[int] = None):
        """
        Initialize baby profile.
        
        Args:
            baby_name: Name of the baby
            birth_date: Birth date of the baby
            age_months: Current age in months (alternative to birth_date)
        """
        self.baby_name = baby_name
        self.birth_date = birth_date
        self.age_months = age_months or self._calculate_age_months()
        
        # Feeding information
        self.last_feeding_time: Optional[datetime] = None
        self.typical_feeding_interval_hours: float = 3.0
        self.feeding_history: List[datetime] = []
        
        # Sleep information
        self.last_nap_time: Optional[datetime] = None
        self.last_wake_time: Optional[datetime] = None
        self.typical_nap_duration_hours: float = 2.0
        self.sleep_history: List[Dict] = []
        
        # Diaper/comfort information
        self.last_diaper_change: Optional[datetime] = None
        self.diaper_change_history: List[datetime] = []
        
        # Health and comfort patterns
        self.comfort_preferences: Dict = {
            'responds_to_rocking': True,
            'responds_to_music': True,
            'responds_to_white_noise': False,
            'prefers_swaddling': True
        }
        
        # Cry pattern learning
        self.cry_history: List[Dict] = []
        self.pattern_preferences: Dict = {}
        
        # Environmental factors
        self.current_environment: Dict = {
            'temperature': None,
            'noise_level': 'normal',
            'lighting': 'normal',
            'people_present': 1
        }
    
    def _calculate_age_months(self) -> int:
        """Calculate age in months from birth date."""
        if self.birth_date:
            today = datetime.now()
            age_days = (today - self.birth_date).days
            return max(0, age_days // 30)  # Approximate months
        return 0
    
    def update_feeding(self, feeding_time: Optional[datetime] = None):
        """
        Update feeding information.
        
        Args:
            feeding_time: Time of feeding (defaults to now)
        """
        if feeding_time is None:
            feeding_time = datetime.now()
        
        self.last_feeding_time = feeding_time
        self.feeding_history.append(feeding_time)
        
        # Keep only recent history (last 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        self.feeding_history = [
            t for t in self.feeding_history if t > cutoff_time
        ]
    
    def update_sleep(self, sleep_start: Optional[datetime] = None,
                    sleep_end: Optional[datetime] = None,
                    is_nap: bool = True):
        """
        Update sleep information.
        
        Args:
            sleep_start: Start time of sleep
            sleep_end: End time of sleep
            is_nap: Whether this was a nap or nighttime sleep
        """
        now = datetime.now()
        
        if sleep_start and sleep_end:
            # Complete sleep record
            sleep_record = {
                'start': sleep_start,
                'end': sleep_end,
                'duration_hours': (sleep_end - sleep_start).total_seconds() / 3600,
                'is_nap': is_nap
            }
            self.sleep_history.append(sleep_record)
            
            if is_nap:
                self.last_nap_time = sleep_start
            self.last_wake_time = sleep_end
            
        elif sleep_start:
            # Just went to sleep
            if is_nap:
                self.last_nap_time = sleep_start
                
        elif sleep_end:
            # Just woke up
            self.last_wake_time = sleep_end
        
        # Keep only recent history (last 7 days)
        cutoff_time = now - timedelta(days=7)
        self.sleep_history = [
            record for record in self.sleep_history 
            if record['end'] > cutoff_time
        ]
    
    def update_diaper_change(self, change_time: Optional[datetime] = None):
        """
        Update diaper change information.
        
        Args:
            change_time: Time of diaper change (defaults to now)
        """
        if change_time is None:
            change_time = datetime.now()
        
        self.last_diaper_change = change_time
        self.diaper_change_history.append(change_time)
        
        # Keep only recent history (last 3 days)
        cutoff_time = datetime.now() - timedelta(days=3)
        self.diaper_change_history = [
            t for t in self.diaper_change_history if t > cutoff_time
        ]
    
    def add_cry_record(self, cry_type: str, confidence: float,
                      resolution: Optional[str] = None,
                      resolution_time_minutes: Optional[float] = None):
        """
        Add a cry classification record for learning.
        
        Args:
            cry_type: Predicted cry type
            confidence: Model confidence
            resolution: How the cry was resolved
            resolution_time_minutes: Time to resolution in minutes
        """
        cry_record = {
            'timestamp': datetime.now(),
            'type': cry_type,
            'confidence': confidence,
            'resolution': resolution,
            'resolution_time': resolution_time_minutes,
            'context': self._get_current_context()
        }
        
        self.cry_history.append(cry_record)
        
        # Keep only recent history (last 30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        self.cry_history = [
            record for record in self.cry_history 
            if record['timestamp'] > cutoff_time
        ]
    
    def _get_current_context(self) -> Dict:
        """Get current contextual information."""
        return {
            'time_since_feeding': self.get_time_since_feeding(),
            'time_since_nap': self.get_time_since_nap(),
            'time_since_diaper_change': self.get_time_since_diaper_change(),
            'time_awake': self.get_time_awake(),
            'environment': self.current_environment.copy()
        }
    
    def get_time_since_feeding(self) -> Optional[float]:
        """Get hours since last feeding."""
        if self.last_feeding_time:
            return (datetime.now() - self.last_feeding_time).total_seconds() / 3600
        return None
    
    def get_time_since_nap(self) -> Optional[float]:
        """Get hours since last nap."""
        if self.last_nap_time:
            return (datetime.now() - self.last_nap_time).total_seconds() / 3600
        return None
    
    def get_time_since_diaper_change(self) -> Optional[float]:
        """Get hours since last diaper change."""
        if self.last_diaper_change:
            return (datetime.now() - self.last_diaper_change).total_seconds() / 3600
        return None
    
    def get_time_awake(self) -> Optional[float]:
        """Get hours since last wake up."""
        if self.last_wake_time:
            return (datetime.now() - self.last_wake_time).total_seconds() / 3600
        return None
    
    def is_likely_hungry(self) -> bool:
        """Determine if baby is likely hungry based on feeding patterns."""
        time_since_feeding = self.get_time_since_feeding()
        if time_since_feeding is None:
            return True  # Unknown, assume possible
        
        # Adjust feeding interval based on age
        expected_interval = self.typical_feeding_interval_hours
        if self.age_months < 2:
            expected_interval = 2.0  # More frequent for newborns
        elif self.age_months < 6:
            expected_interval = 3.0
        else:
            expected_interval = 4.0
        
        return time_since_feeding >= expected_interval * 0.8
    
    def is_likely_tired(self) -> bool:
        """Determine if baby is likely tired based on sleep patterns."""
        time_awake = self.get_time_awake()
        if time_awake is None:
            return False
        
        # Age-appropriate wake windows
        if self.age_months < 2:
            max_awake_time = 1.5
        elif self.age_months < 6:
            max_awake_time = 2.5
        elif self.age_months < 12:
            max_awake_time = 3.5
        else:
            max_awake_time = 4.0
        
        return time_awake >= max_awake_time
    
    def is_likely_uncomfortable(self) -> bool:
        """Determine if baby is likely uncomfortable (diaper, etc.)."""
        time_since_change = self.get_time_since_diaper_change()
        if time_since_change is None:
            return True  # Unknown, assume possible
        
        return time_since_change >= 2.0  # 2+ hours since last change
    
    def get_context_probabilities(self) -> Dict[str, float]:
        """
        Get probability adjustments based on current context.
        
        Returns:
            Dictionary with probability adjustments for each cry type
        """
        adjustments = {
            'hunger': 0.0,
            'tiredness': 0.0,
            'discomfort': 0.0,
            'pain': 0.0,
            'normal': 0.0
        }
        
        # Hunger adjustments
        if self.is_likely_hungry():
            adjustments['hunger'] += 0.3
        
        # Tiredness adjustments
        if self.is_likely_tired():
            adjustments['tiredness'] += 0.25
        
        # Discomfort adjustments
        if self.is_likely_uncomfortable():
            adjustments['discomfort'] += 0.2
        
        # Time-of-day adjustments
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:  # Night time
            adjustments['tiredness'] += 0.1
            adjustments['hunger'] += 0.1  # Night feedings common
        
        return adjustments
    
    def get_historical_patterns(self) -> Dict:
        """Analyze historical cry patterns for insights."""
        if len(self.cry_history) < 5:
            return {}
        
        patterns = {}
        
        # Most common cry types
        cry_types = [record['type'] for record in self.cry_history]
        from collections import Counter
        type_counts = Counter(cry_types)
        patterns['most_common_types'] = type_counts.most_common(3)
        
        # Most effective resolutions
        resolutions = [
            record['resolution'] for record in self.cry_history 
            if record['resolution']
        ]
        if resolutions:
            resolution_counts = Counter(resolutions)
            patterns['effective_resolutions'] = resolution_counts.most_common(3)
        
        # Time patterns
        hours = [record['timestamp'].hour for record in self.cry_history]
        hour_counts = Counter(hours)
        patterns['common_cry_times'] = hour_counts.most_common(3)
        
        return patterns
    
    def save_profile(self, filepath: str):
        """Save baby profile to JSON file."""
        profile_data = {
            'baby_name': self.baby_name,
            'birth_date': self.birth_date.isoformat() if self.birth_date else None,
            'age_months': self.age_months,
            'last_feeding_time': self.last_feeding_time.isoformat() if self.last_feeding_time else None,
            'typical_feeding_interval_hours': self.typical_feeding_interval_hours,
            'last_nap_time': self.last_nap_time.isoformat() if self.last_nap_time else None,
            'last_wake_time': self.last_wake_time.isoformat() if self.last_wake_time else None,
            'typical_nap_duration_hours': self.typical_nap_duration_hours,
            'last_diaper_change': self.last_diaper_change.isoformat() if self.last_diaper_change else None,
            'comfort_preferences': self.comfort_preferences,
            'current_environment': self.current_environment,
            # Note: For privacy, we might not want to save detailed history
            'feeding_history_count': len(self.feeding_history),
            'sleep_history_count': len(self.sleep_history),
            'cry_history_count': len(self.cry_history)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    @classmethod
    def load_profile(cls, filepath: str) -> 'BabyProfile':
        """Load baby profile from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        profile = cls(
            baby_name=data['baby_name'],
            age_months=data['age_months']
        )
        
        if data['birth_date']:
            profile.birth_date = datetime.fromisoformat(data['birth_date'])
        
        if data['last_feeding_time']:
            profile.last_feeding_time = datetime.fromisoformat(data['last_feeding_time'])
        
        if data['last_nap_time']:
            profile.last_nap_time = datetime.fromisoformat(data['last_nap_time'])
        
        if data['last_wake_time']:
            profile.last_wake_time = datetime.fromisoformat(data['last_wake_time'])
        
        if data['last_diaper_change']:
            profile.last_diaper_change = datetime.fromisoformat(data['last_diaper_change'])
        
        profile.typical_feeding_interval_hours = data['typical_feeding_interval_hours']
        profile.typical_nap_duration_hours = data['typical_nap_duration_hours']
        profile.comfort_preferences = data['comfort_preferences']
        profile.current_environment = data['current_environment']
        
        return profile 