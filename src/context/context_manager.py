"""Context-aware cry classification management."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

from .baby_profile import BabyProfile


class ContextManager:
    """Manage context-aware baby cry classification."""
    
    def __init__(self):
        """Initialize the context manager."""
        self.active_profiles: Dict[str, BabyProfile] = {}
        self.context_weights = {
            'hunger': 0.4,
            'tiredness': 0.3,
            'discomfort': 0.2,
            'pain': 0.1,
            'normal': 0.0
        }
    
    def add_baby_profile(self, profile: BabyProfile, profile_id: str = None) -> str:
        """
        Add a baby profile to the context manager.
        
        Args:
            profile: BabyProfile instance
            profile_id: Optional custom ID (defaults to baby name)
            
        Returns:
            Profile ID for future reference
        """
        if profile_id is None:
            profile_id = profile.baby_name.lower().replace(' ', '_')
        
        self.active_profiles[profile_id] = profile
        return profile_id
    
    def get_baby_profile(self, profile_id: str) -> Optional[BabyProfile]:
        """Get a baby profile by ID."""
        return self.active_profiles.get(profile_id)
    
    def apply_context_to_prediction(self, 
                                   base_probabilities: np.ndarray,
                                   baby_profile: BabyProfile,
                                   classes: List[str]) -> Dict:
        """
        Apply contextual information to modify base model predictions.
        
        Args:
            base_probabilities: Raw model probabilities
            baby_profile: Baby profile with contextual information
            classes: List of class names
            
        Returns:
            Dictionary with context-adjusted predictions and explanations
        """
        # Get context probability adjustments
        context_adjustments = baby_profile.get_context_probabilities()
        
        # Apply adjustments to base probabilities
        adjusted_probs = base_probabilities.copy()
        
        for i, class_name in enumerate(classes):
            if class_name in context_adjustments:
                adjustment = context_adjustments[class_name] * self.context_weights.get(class_name, 0.1)
                adjusted_probs[i] += adjustment
        
        # Normalize probabilities
        adjusted_probs = np.maximum(adjusted_probs, 0.01)  # Prevent negative probabilities
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        # Get predictions
        base_prediction_idx = np.argmax(base_probabilities)
        adjusted_prediction_idx = np.argmax(adjusted_probs)
        
        base_prediction = classes[base_prediction_idx]
        adjusted_prediction = classes[adjusted_prediction_idx]
        
        # Generate explanation
        explanation = self._generate_context_explanation(
            baby_profile, base_prediction, adjusted_prediction, 
            base_probabilities[base_prediction_idx],
            adjusted_probs[adjusted_prediction_idx],
            context_adjustments
        )
        
        return {
            'base_prediction': base_prediction,
            'base_confidence': float(base_probabilities[base_prediction_idx]),
            'base_probabilities': {
                classes[i]: float(prob) for i, prob in enumerate(base_probabilities)
            },
            'context_adjusted_prediction': adjusted_prediction,
            'context_adjusted_confidence': float(adjusted_probs[adjusted_prediction_idx]),
            'context_adjusted_probabilities': {
                classes[i]: float(prob) for i, prob in enumerate(adjusted_probs)
            },
            'explanation': explanation,
            'context_factors': context_adjustments,
            'prediction_changed': base_prediction != adjusted_prediction
        }
    
    def _generate_context_explanation(self, 
                                    profile: BabyProfile,
                                    base_prediction: str,
                                    adjusted_prediction: str,
                                    base_confidence: float,
                                    adjusted_confidence: float,
                                    context_factors: Dict[str, float]) -> str:
        """Generate human-readable explanation for the prediction."""
        explanations = []
        
        # Base prediction
        explanations.append(f"Initial AI prediction: {base_prediction} ({base_confidence:.2f} confidence)")
        
        # Context factors
        active_factors = []
        
        # Check feeding status
        time_since_feeding = profile.get_time_since_feeding()
        if time_since_feeding is not None:
            if time_since_feeding == 0.0:
                active_factors.append("Just fed")
            elif profile.is_likely_hungry():
                active_factors.append(f"Last fed {time_since_feeding:.1f} hours ago (typical interval: {profile.typical_feeding_interval_hours:.1f}h)")
            else:
                active_factors.append(f"Recently fed {time_since_feeding:.1f} hours ago")
        
        # Check sleep status
        time_awake = profile.get_time_awake()
        if time_awake is not None:
            if time_awake == 0.0:
                active_factors.append("Just woke up")
            elif profile.is_likely_tired():
                active_factors.append(f"Awake for {time_awake:.1f} hours (may be tired)")
            else:
                active_factors.append(f"Awake for {time_awake:.1f} hours")
        
        # Check diaper status
        time_since_diaper = profile.get_time_since_diaper_change()
        if time_since_diaper is not None:
            if time_since_diaper == 0.0:
                active_factors.append("Diaper just changed")
            elif profile.is_likely_uncomfortable():
                active_factors.append(f"Diaper changed {time_since_diaper:.1f} hours ago (may need changing)")
            else:
                active_factors.append(f"Diaper recently changed ({time_since_diaper:.1f}h ago)")
        
        # Time of day context
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:
            active_factors.append("It's nighttime (common for feeding or sleep issues)")
        elif 6 <= current_hour <= 12:
            active_factors.append("It's morning")
        elif 12 <= current_hour <= 18:
            active_factors.append("It's afternoon")
        else:
            active_factors.append("It's evening")
        
        if active_factors:
            explanations.append("Context factors: " + "; ".join(active_factors))
        
        # Final prediction with reasoning
        if base_prediction != adjusted_prediction:
            explanations.append(
                f"Context-adjusted prediction: {adjusted_prediction} "
                f"({adjusted_confidence:.2f} confidence) - "
                f"prediction changed based on baby's current needs"
            )
        else:
            explanations.append(
                f"Final prediction: {adjusted_prediction} "
                f"({adjusted_confidence:.2f} confidence) - "
                f"context supports AI prediction"
            )
        
        # Add specific recommendations
        recommendations = self._get_recommendations(adjusted_prediction, profile)
        if recommendations:
            explanations.append(f"Recommendations: {recommendations}")
        
        return " | ".join(explanations)
    
    def _get_recommendations(self, prediction: str, profile: BabyProfile) -> str:
        """Generate specific recommendations based on prediction and context."""
        recommendations = []
        
        if prediction == 'hunger':
            if profile.is_likely_hungry():
                recommendations.append("Try feeding")
            else:
                recommendations.append("Check if baby is showing hunger cues")
        
        elif prediction == 'tiredness':
            if profile.is_likely_tired():
                recommendations.append("Create calm environment for sleep")
                if profile.comfort_preferences.get('responds_to_rocking'):
                    recommendations.append("try gentle rocking")
                if profile.comfort_preferences.get('prefers_swaddling'):
                    recommendations.append("consider swaddling")
            else:
                recommendations.append("Look for other sleep cues")
        
        elif prediction == 'discomfort':
            if profile.is_likely_uncomfortable():
                recommendations.append("Check diaper")
            recommendations.append("check temperature, clothing, or position")
        
        elif prediction == 'pain':
            recommendations.append("Check for signs of illness or injury")
            recommendations.append("consider consulting healthcare provider if persistent")
        
        return ", ".join(recommendations)
    
    def learn_from_feedback(self, 
                           profile_id: str,
                           prediction: str,
                           confidence: float,
                           actual_cause: str,
                           resolution_method: str,
                           resolution_time_minutes: float):
        """
        Learn from user feedback to improve future predictions.
        
        Args:
            profile_id: Baby profile ID
            prediction: What was predicted
            confidence: Model confidence
            actual_cause: What actually resolved the crying
            resolution_method: How it was resolved
            resolution_time_minutes: Time to resolution
        """
        profile = self.get_baby_profile(profile_id)
        if profile:
            profile.add_cry_record(
                cry_type=actual_cause,
                confidence=confidence,
                resolution=resolution_method,
                resolution_time_minutes=resolution_time_minutes
            )
            
            # Update context weights based on accuracy
            if prediction == actual_cause:
                # Correct prediction - slightly increase weight
                if actual_cause in self.context_weights:
                    self.context_weights[actual_cause] = min(
                        1.0, self.context_weights[actual_cause] * 1.02
                    )
            else:
                # Incorrect prediction - adjust weights
                if prediction in self.context_weights:
                    self.context_weights[prediction] = max(
                        0.05, self.context_weights[prediction] * 0.98
                    )
                if actual_cause in self.context_weights:
                    self.context_weights[actual_cause] = min(
                        1.0, self.context_weights[actual_cause] * 1.01
                    )
    
    def get_personalized_insights(self, profile_id: str) -> Dict:
        """
        Get personalized insights for a specific baby.
        
        Args:
            profile_id: Baby profile ID
            
        Returns:
            Dictionary with insights and patterns
        """
        profile = self.get_baby_profile(profile_id)
        if not profile:
            return {}
        
        insights = {
            'baby_name': profile.baby_name,
            'age_months': profile.age_months,
            'current_status': self._get_current_status(profile),
            'patterns': profile.get_historical_patterns(),
            'recommendations': self._get_general_recommendations(profile)
        }
        
        return insights
    
    def _get_current_status(self, profile: BabyProfile) -> Dict:
        """Get current status summary for the baby."""
        return {
            'time_since_feeding': profile.get_time_since_feeding(),
            'time_since_nap': profile.get_time_since_nap(),
            'time_since_diaper_change': profile.get_time_since_diaper_change(),
            'time_awake': profile.get_time_awake(),
            'likely_hungry': profile.is_likely_hungry(),
            'likely_tired': profile.is_likely_tired(),
            'likely_uncomfortable': profile.is_likely_uncomfortable()
        }
    
    def _get_general_recommendations(self, profile: BabyProfile) -> List[str]:
        """Get general care recommendations based on current context."""
        recommendations = []
        
        if profile.is_likely_hungry():
            recommendations.append("Consider feeding soon")
        
        if profile.is_likely_tired():
            recommendations.append("Prepare for nap time")
        
        if profile.is_likely_uncomfortable():
            recommendations.append("Check diaper and comfort")
        
        # Age-specific recommendations
        if profile.age_months < 3:
            recommendations.append("Remember: newborns cry 2-3 hours daily on average")
        elif profile.age_months < 6:
            recommendations.append("Watch for developmental changes in cry patterns")
        
        return recommendations
    
    def export_baby_data(self, profile_id: str, include_history: bool = False) -> Dict:
        """
        Export baby data for analysis or backup.
        
        Args:
            profile_id: Baby profile ID
            include_history: Whether to include detailed history
            
        Returns:
            Dictionary with baby data
        """
        profile = self.get_baby_profile(profile_id)
        if not profile:
            return {}
        
        data = {
            'profile_info': {
                'baby_name': profile.baby_name,
                'age_months': profile.age_months,
                'birth_date': profile.birth_date.isoformat() if profile.birth_date else None
            },
            'current_status': self._get_current_status(profile),
            'preferences': profile.comfort_preferences,
            'environment': profile.current_environment
        }
        
        if include_history:
            data['feeding_history_count'] = len(profile.feeding_history)
            data['sleep_history_count'] = len(profile.sleep_history)
            data['cry_history_count'] = len(profile.cry_history)
            data['patterns'] = profile.get_historical_patterns()
        
        return data 