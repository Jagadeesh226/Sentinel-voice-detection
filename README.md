Sentinel Voice – English Fatigue Detection Pipeline
A personalized voice-based fatigue detection system designed to analyze changes in a speaker's voice and identify potential fatigue levels.
Overview
Sentinel Voice combines a global fatigue classification model with speaker-specific personalization.
The system detects three primary states:
🟢 Alert
🟡 Mild Fatigue
🔴 High Fatigue
Instead of relying only on a global model, Sentinel Voice creates a personalized vocal baseline for each registered speaker. Future recordings are compared against this baseline to detect deviations from the speaker's normal voice characteristics.
Features
🎙️ Speaker Identity Verification
Registers speakers using voice embeddings.
Verifies that an uploaded recording belongs to the selected speaker.
Calculates voice similarity before performing fatigue analysis.
🧠 Global Fatigue Classification
The global model predicts:
Alert
Mild Fatigue
High Fatigue
Prediction probabilities and confidence scores are also generated.
📊 Speech Analysis
The pipeline extracts:
Words Per Minute (WPM)
Words Per Second (WPS)
Speech transcription is used to calculate speech-rate features.
🔬 Acoustic Feature Extraction
The system analyzes the following personalized voice features:
Mean F0
Energy
Jitter
Shimmer
Harmonics-to-Noise Ratio (HNR)
👤 Personalized Speaker Baseline
Each speaker creates a baseline using multiple reliable recordings predicted as Alert.
The baseline stores statistical information for each feature:
Mean
Median
Standard Deviation
📈 Personalized Deviation Analysis
New recordings are compared against the speaker's baseline using Z-score-based deviation analysis.
Each feature is categorized as:
NORMAL
SLIGHT_DEVIATION
SIGNIFICANT_DEVIATION
EXTREME_DEVIATION
A combined personalized deviation score is also calculated.
🔀 Fusion-Based Final Assessment
The system combines:
Global Classifier Output
        +
Personalized Deviation Score
        ↓
Final Fatigue Assessment
        ↓
Risk Level
This allows Sentinel Voice to consider both general fatigue patterns and individual changes in vocal behavior.
Pipeline Architecture
Audio Input
    ↓
Audio Preprocessing
    ↓
Speaker Identity Verification
    ↓
Feature Extraction
    ├── Speech Features
    │     ├── WPM
    │     └── WPS
    │
    └── Acoustic Features
          ├── Mean F0
          ├── Energy
          ├── Jitter
          ├── Shimmer
          └── HNR
    ↓
Global Fatigue Classifier
    ↓
Speaker Personalization
    ├── Candidate Management
    ├── Baseline Creation
    └── Deviation Analysis
    ↓
Fusion
    ↓
Final Fatigue Assessment
    ↓
Risk Level
Speaker Personalization Workflow
1. Register Speaker
A new speaker is assigned a unique speaker_id.
2. Verify Speaker Identity
Voice embeddings are compared with the registered speaker identity.
Voice Recording
      ↓
Speaker Embedding
      ↓
Cosine Similarity
      ↓
Identity Verified
3. Create Candidate Recordings
Reliable Alert recordings are stored as baseline candidates.
4. Create Permanent Baseline
After collecting sufficient reliable candidates, the system creates a permanent personalized baseline.
Example:
{
    "wpm": {
        "mean": 185.84,
        "median": 183.97,
        "std": 2.94
    },
    "wps": {
        "mean": 3.09,
        "median": 3.06,
        "std": 0.04
    },
    "num_recordings": 3
}
5. Analyze Future Recordings
Once a baseline exists, new recordings are compared against the speaker's normal vocal characteristics.
Project Structure
sentinel_voice/
│
├── app.py
│
├── src/
│   ├── personalization/
│   │   ├── baseline.py
│   │   ├── candidate_manager.py
│   │   ├── deviation.py
│   │   ├── feature_extractor.py
│   │   ├── fusion.py
│   │   ├── speaker_identity.py
│   │   └── workflow.py
│   │
│   └── ...
│
├── data/
│   ├── baselines/
│   ├── baseline_candidates/
│   └── speaker_identities/
│
└── requirements.txt
Running the Application
Activate your virtual environment:
source jagu/bin/activate
Run the Streamlit application:
streamlit run app.py
The application will open in your browser.
Current Status
The English pipeline currently supports:
Speaker registration
Speaker identity verification
Alert, Mild Fatigue, and High Fatigue classification
Speech-rate analysis
Acoustic feature extraction
Personalized baseline creation
Candidate validation
Feature-level deviation analysis
Personalized deviation scoring
Fusion-based fatigue assessment
Risk-level generation
The pipeline has also been tested with multiple speakers and previously unseen voice recordings.
Future Work
Planned improvements include:
Hindi language support
Multilingual voice processing
Additional speaker testing
Larger validation datasets
Adaptive baseline updates using reliable future recordings
Further model evaluation and optimization
Tech Stack
Python
PyTorch
Streamlit
Whisper
WavLM
ECAPA-TDNN
NumPy
Audio processing and acoustic feature extraction libraries
