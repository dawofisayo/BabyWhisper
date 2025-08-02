#!/usr/bin/env python3
"""
FAST Context-Aware Demo - Shows intelligent prediction adjustments without audio processing
"""

import numpy as np
from datetime import datetime, timedelta
from src.context import BabyProfile, ContextManager


def simulate_base_predictions():
    """Simulate some base AI predictions to show context adjustments."""
    scenarios = [
        {
            'audio_type': 'hunger_cry.wav',
            'base_prediction': 'hunger',
            'base_probabilities': np.array([0.75, 0.10, 0.05, 0.05, 0.05]),  # hunger, pain, discomfort, tiredness, normal
            'description': 'Rhythmic crying pattern detected'
        },
        {
            'audio_type': 'discomfort_cry.wav', 
            'base_prediction': 'discomfort',
            'base_probabilities': np.array([0.20, 0.15, 0.60, 0.03, 0.02]),
            'description': 'Irregular whimpering pattern'
        },
        {
            'audio_type': 'tiredness_cry.wav',
            'base_prediction': 'tiredness', 
            'base_probabilities': np.array([0.10, 0.05, 0.15, 0.65, 0.05]),
            'description': 'Low energy, intermittent crying'
        }
    ]
    return scenarios


def create_baby_profiles():
    """Create different baby profiles with various contexts."""
    now = datetime.now()
    
    profiles = {
        'recently_fed': {
            'profile': BabyProfile("Recently Fed Baby", age_months=4),
            'context': 'Fed 15 minutes ago, awake 1 hour, diaper changed 2h ago'
        },
        'hungry_baby': {
            'profile': BabyProfile("Hungry Baby", age_months=6), 
            'context': 'Fed 4 hours ago, awake 2 hours, diaper clean'
        },
        'tired_baby': {
            'profile': BabyProfile("Tired Baby", age_months=3),
            'context': 'Fed 1.5h ago, awake 3 hours!, diaper clean'
        },
        'newborn': {
            'profile': BabyProfile("Newborn Baby", age_months=1),
            'context': 'Fed 2h ago, awake 2h, diaper changed 3h ago'
        }
    }
    
    # Set up contexts
    profiles['recently_fed']['profile'].update_feeding(now - timedelta(minutes=15))
    profiles['recently_fed']['profile'].update_sleep(sleep_end=now - timedelta(hours=1))
    profiles['recently_fed']['profile'].update_diaper_change(now - timedelta(hours=2))
    
    profiles['hungry_baby']['profile'].update_feeding(now - timedelta(hours=4))
    profiles['hungry_baby']['profile'].update_sleep(sleep_end=now - timedelta(hours=2))
    profiles['hungry_baby']['profile'].update_diaper_change(now - timedelta(minutes=30))
    
    profiles['tired_baby']['profile'].update_feeding(now - timedelta(hours=1.5))
    profiles['tired_baby']['profile'].update_sleep(sleep_end=now - timedelta(hours=3))
    profiles['tired_baby']['profile'].update_diaper_change(now - timedelta(minutes=45))
    
    profiles['newborn']['profile'].update_feeding(now - timedelta(hours=2))
    profiles['newborn']['profile'].update_sleep(sleep_end=now - timedelta(hours=2))
    profiles['newborn']['profile'].update_diaper_change(now - timedelta(hours=3))
    
    return profiles


def demo_context_intelligence():
    """Demonstrate intelligent context-aware adjustments."""
    
    print("🍼 FAST BabyWhisper Context Intelligence Demo 🍼")
    print("=" * 60)
    print("⚡ Skipping audio processing - focusing on smart context logic!")
    
    # Setup
    context_manager = ContextManager()
    classes = ['hunger', 'pain', 'discomfort', 'tiredness', 'normal']
    scenarios = simulate_base_predictions()
    profiles = create_baby_profiles()
    
    print(f"\n📋 Testing {len(scenarios)} cry scenarios with {len(profiles)} baby contexts...")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n" + "="*60)
        print(f"🎵 SCENARIO {i}: {scenario['audio_type']}")
        print(f"📝 {scenario['description']}")
        print("-" * 40)
        
        base_pred = scenario['base_prediction']
        base_prob = np.max(scenario['base_probabilities'])
        
        print(f"🤖 BASE AI PREDICTION: {base_pred} ({base_prob:.2f} confidence)")
        
        # Test with each baby profile
        for profile_name, profile_data in profiles.items():
            profile = profile_data['profile']
            context_desc = profile_data['context']
            
            print(f"\n👶 BABY CONTEXT: {profile.baby_name}")
            print(f"   📊 {context_desc}")
            
            # Apply context intelligence
            result = context_manager.apply_context_to_prediction(
                scenario['base_probabilities'],
                profile,
                classes
            )
            
            # Show results
            changed_flag = " 🔄 CHANGED!" if result['prediction_changed'] else ""
            print(f"   🎯 Context-Adjusted: {result['context_adjusted_prediction']} "
                  f"({result['context_adjusted_confidence']:.2f}){changed_flag}")
            
            # Show what context factors influenced the decision
            factors = result['context_factors']
            active_factors = [(k, v) for k, v in factors.items() if v > 0.05]
            if active_factors:
                factor_str = ", ".join([f"{k}: +{v:.2f}" for k, v in active_factors])
                print(f"   ⚡ Context boosts: {factor_str}")
            
            # Show the intelligent reasoning
            explanation_parts = result['explanation'].split(' | ')
            if len(explanation_parts) > 1:
                print(f"   🧠 Smart reasoning: {explanation_parts[1]}")
        
        # Show key insights for this scenario
        print(f"\n💡 KEY INSIGHTS:")
        if base_pred == 'hunger':
            print("   • Recently fed babies → system downgrades hunger, considers gas/discomfort")
            print("   • Hungry babies (4h since feeding) → system boosts hunger confidence")
        elif base_pred == 'discomfort':
            print("   • System considers diaper timing and age-appropriate discomfort patterns")
        elif base_pred == 'tiredness':
            print("   • Newborns tire faster (1.5h) vs older babies (3h+)")
            print("   • Night-time crying gets tiredness boost")
    
    print(f"\n" + "="*60)
    print("🎯 CONTEXT INTELLIGENCE DEMONSTRATION COMPLETE!")
    print("\n🧠 What Makes This Smart:")
    print("✅ Age-appropriate expectations (newborn vs 6-month feeding intervals)")
    print("✅ Temporal reasoning (time since feeding/sleeping/diaper change)")
    print("✅ Contextual probability adjustments based on care history") 
    print("✅ Intelligent explanations that help parents understand WHY")
    print("✅ Confidence boosting/dampening based on likelihood")
    
    # Show a detailed technical example
    print(f"\n🔬 TECHNICAL EXAMPLE:")
    hungry_baby = profiles['hungry_baby']['profile']
    recently_fed = profiles['recently_fed']['profile']
    
    print(f"Hungry Baby ({hungry_baby.baby_name}):")
    print(f"  ⏰ Hours since feeding: {hungry_baby.get_time_since_feeding():.1f}")
    print(f"  🍼 Is likely hungry: {hungry_baby.is_likely_hungry()}")
    print(f"  💤 Is likely tired: {hungry_baby.is_likely_tired()}")
    
    print(f"\nRecently Fed Baby ({recently_fed.baby_name}):")
    print(f"  ⏰ Hours since feeding: {recently_fed.get_time_since_feeding():.1f}")
    print(f"  🍼 Is likely hungry: {recently_fed.is_likely_hungry()}")
    print(f"  💤 Is likely tired: {recently_fed.is_likely_tired()}")
    
    print(f"\n🎓 Result: Same cry → Different interpretations based on context!")


if __name__ == "__main__":
    demo_context_intelligence() 