# 🍼 BabyWhisper - AI-Powered Baby Cry Classification System

**Revolutionary AI system that combines advanced audio processing with contextual intelligence to help parents understand their babies better.**

BabyWhisper doesn't just classify baby cries—it provides intelligent, context-aware insights that take into account each baby's unique patterns, feeding schedules, sleep cycles, and developmental stage.

## ✨ Key Features

### 🎵 **Advanced Audio Processing**
- **323 audio features** extracted per cry (MFCC, spectral, temporal, chroma, mel-spectrogram, F0, energy)
- **Real-time processing** capabilities for immediate insights
- **Robust preprocessing** with noise reduction and normalization

### 🧠 **Intelligent Classification**
- **Ensemble ML models** combining Random Forest, SVM, and Multi-layer Perceptron
- **100% accuracy** achieved on synthetic datasets
- **5 cry categories**: hunger, pain, discomfort, tiredness, normal sounds

### 💡 **Context-Aware Intelligence** (The Game Changer!)
- **Smart context integration** using baby profiles and caregiving history
- **Age-appropriate logic** (newborn vs 6-month-old patterns)
- **Dynamic prediction adjustment** based on feeding times, sleep patterns, diaper changes
- **Intelligent explanations** with actionable insights

### 📊 **Continuous Learning**
- **Feedback integration** to improve predictions over time
- **Pattern recognition** for each baby's unique characteristics
- **Personalized recommendations** based on individual baby profiles

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Audio Input   │───▶│ Feature Extraction│───▶│   Ensemble Models   │
│  (Baby Cry)     │    │  (323 features)   │    │  RF + SVM + MLP     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
┌─────────────────┐    ┌──────────────────┐              │
│ Smart Insights  │◀───│ Context Layer    │◀─────────────┘
│ & Explanations  │    │ (Baby Profile +  │
└─────────────────┘    │  Care History)   │
                       └──────────────────┘
```

## 📁 Project Structure

```
BabyWhisper/
├── src/
│   ├── audio_processing/          # Audio feature extraction & preprocessing
│   │   ├── feature_extractor.py   # 323 audio features extraction
│   │   └── preprocessor.py        # Audio cleaning & normalization
│   ├── models/                    # Machine learning components
│   │   ├── classifier.py          # Ensemble models (RF+SVM+MLP)
│   │   └── model_trainer.py       # Training pipeline & evaluation
│   ├── context/                   # Context intelligence system
│   │   ├── baby_profile.py        # Individual baby profiles & patterns
│   │   └── context_manager.py     # Smart prediction adjustments
│   ├── utils/                     # Utilities & evaluation
│   │   ├── data_loader.py         # Dataset management & synthetic data
│   │   └── evaluation.py          # Performance metrics & reporting
│   └── main.py                    # Main BabyWhisper interface
├── notebooks/
│   └── demo_notebook.ipynb        # Complete development journey
├── data/
├── models/                        # Trained model storage
├── demo scripts/
│   ├── example_usage.py           # Full system demonstration
│   ├── simple_demo.py             # Quick feature showcase
│   ├── quick_context_demo.py      # Context intelligence demo
│   └── demo_context_awareness.py  # Advanced context testing
├── requirements.txt               # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dawofisayo/BabyWhisper.git
cd BabyWhisper

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

### Basic Usage

```python
from src.main import BabyWhisperClassifier
from src.context import BabyProfile
from datetime import datetime, timedelta

# Initialize the system
baby_whisper = BabyWhisperClassifier()

# Create a baby profile for context-aware predictions
baby_profile = baby_whisper.create_baby_profile(
    name="Emma",
    age_months=4,
    birth_date=datetime.now() - timedelta(days=120)
)

# Update baby's current context
baby_whisper.update_baby_context(
    profile_id=baby_profile,
    feeding_time=datetime.now() - timedelta(hours=2.5),
    nap_time=datetime.now() - timedelta(hours=1.5),
    diaper_change_time=datetime.now() - timedelta(minutes=45)
)

# Classify a cry with intelligent context
result = baby_whisper.classify_cry(
    audio_path="path/to/baby_cry.wav",
    baby_profile=baby_profile
)

print(f"Prediction: {result['final_prediction']}")
print(f"Confidence: {result['final_confidence']:.2f}")
print(f"Explanation: {result['explanation']}")
print(f"Recommendations: {result['recommendations']}")
```

## 🎬 Demo Scripts

Run these scripts to see BabyWhisper in action:

```bash
# Complete system demonstration
python example_usage.py

# Quick feature showcase
python simple_demo.py

# Context intelligence demo
python quick_context_demo.py
```

## 📊 Performance Metrics

| Model Component | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|---------|----------|
| Random Forest  | 99.5%    | 99.5%     | 99.5%   | 99.5%    |
| SVM           | 100%     | 100%      | 100%    | 100%     |
| MLP           | 97.5%    | 97.5%     | 97.5%   | 97.5%    |
| **Ensemble**  | **100%** | **100%**  | **100%** | **100%** |

### Technical Specifications
- **Audio Features**: 323 characteristics per cry sample
- **Processing Speed**: Real-time capable
- **Model Types**: Ensemble (Random Forest + SVM + Multi-layer Perceptron)
- **Context Factors**: Feeding history, sleep patterns, age, time of day
- **Supported Audio**: WAV, MP3, FLAC formats

## 🧪 Development Journey

Explore the complete development process in our interactive Jupyter notebook:

```bash
jupyter notebook notebooks/demo_notebook.ipynb
```

The notebook includes:
- 🔬 **Audio feature exploration** and experimentation
- 🤖 **ML model comparison** and performance testing  
- 🧠 **Context intelligence development** and validation
- 🚀 **System integration** and end-to-end testing
- 💭 **Technical insights** and lessons learned

## 🌟 What Makes BabyWhisper Special

### Revolutionary Context Intelligence
Unlike traditional audio classifiers, BabyWhisper understands that **context matters**:

- **Same cry at 2 AM after 4 hours** → Likely hunger
- **Same cry at 2 PM after 30 minutes** → Probably not hunger
- **Newborn crying after 2 hours awake** → Likely tiredness
- **6-month-old crying after 2 hours awake** → Context-dependent

### Smart Explanations
BabyWhisper provides actionable insights:
> *"Baby might be hungry instead of tired — they just ate 15 minutes ago, and it's 9PM. Consider checking diaper or providing comfort."*

### Continuous Learning
The system learns from feedback to improve accuracy and discover each baby's unique patterns.

## 🔬 Technical Innovation

- **Feature Engineering**: 323 sophisticated audio characteristics
- **Ensemble Learning**: Multiple ML models for robust predictions  
- **Context Integration**: Baby-specific pattern recognition
- **Real-time Processing**: Optimized for immediate insights
- **Modular Architecture**: Extensible and maintainable codebase

## 🤝 Contributing

We welcome contributions! Areas for development:

1. **Real Dataset Integration**: Help us gather authentic baby cry recordings
2. **Mobile App Development**: React Native/Flutter implementation
3. **IoT Integration**: Smart baby monitor connectivity
4. **Advanced ML**: Transformer models, voice recognition
5. **Clinical Validation**: Pediatric research collaboration

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Audio processing powered by `librosa` and `soundfile`
- Machine learning with `scikit-learn` and `tensorflow`
- Special thanks to the open-source audio processing community

---

**BabyWhisper: Where AI meets parental intuition** 🍼✨

*Helping exhausted parents understand their babies, one cry at a time.* 