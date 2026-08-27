Sentinel Voice
Personalized English Voice-Based Fatigue Detection
Sentinel Voice is a personalized voice-fatigue detection system that analyzes a speaker's voice to identify potential fatigue.
The system combines a global fatigue classification model with speaker-specific voice analysis to provide a more personalized final assessment.
Features
Speaker Identity Verification
Registers speakers using voice embeddings
Verifies that an uploaded recording belongs to the selected speaker
Uses cosine similarity for speaker verification
Global Fatigue Classification
The system predicts three fatigue states:
Alert
Mild Fatigue
High Fatigue
Speech Analysis
The pipeline extracts:
Words Per Minute (WPM)
Words Per Second (WPS)
Acoustic Feature Analysis
The following voice features are analyzed:
Mean F0
Energy
Jitter
Shimmer
Harmonics-to-Noise Ratio (HNR)
Personalized Speaker Baseline
Each speaker has a personalized voice baseline created using multiple reliable Alert recordings.
For each feature, the baseline stores:
Mean
Median
Standard Deviation
Personalized Deviation Analysis
New recordings are compared with the speaker's baseline using Z-score-based deviation analysis.
Each feature can be categorized as:
NORMAL
SLIGHT_DEVIATION
SIGNIFICANT_DEVIATION
EXTREME_DEVIATION
Fusion-Based Assessment
The final fatigue assessment combines:
Global fatigue classifier output
Personalized voice deviation score
This produces a final fatigue assessment and risk level.
Pipeline Architecture
Audio Input
    │
    ▼
Audio Preprocessing
    │
    ▼
Speaker Identity Verification
    │
    ▼
Feature Extraction
    │
    ├── Speech Features
    │   ├── WPM
    │   └── WPS
    │
    └── Acoustic Features
        ├── Mean F0
        ├── Energy
        ├── Jitter
        ├── Shimmer
        └── HNR
    │
    ▼
Global Fatigue Classifier
    │
    ▼
Personalized Speaker Analysis
    │
    ├── Candidate Management
    ├── Baseline Creation
    └── Deviation Analysis
    │
    ▼
Fusion
    │
    ▼
Final Fatigue Assessment
    │
    ▼
Risk Level
Speaker Personalization Workflow
1. Register Speaker
A new speaker is assigned a unique speaker_id.
2. Speaker Verification
The uploaded recording is converted into a speaker embedding and compared with the registered speaker identity.
Voice Recording
       │
       ▼
Speaker Embedding
       │
       ▼
Cosine Similarity
       │
       ▼
Speaker Verified
3. Candidate Collection
Reliable recordings predicted as Alert are stored as baseline candidates.
4. Baseline Creation
After collecting enough reliable candidates, a permanent speaker baseline is created.
The baseline contains statistical information for:
WPM
WPS
Mean F0
Energy
Jitter
Shimmer
HNR
5. Personalized Analysis
Future recordings are compared against the speaker's baseline to measure changes in vocal characteristics.
Project Structure
sentinel_voice/
│
├── app.py
│
├── src/
│   │
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
├── requirements.txt
│
└── README.md
Running the Application
1. Activate the Virtual Environment
source jagu/bin/activate
2. Run the Streamlit Application
streamlit run app.py
The application will open in your browser.
Current Capabilities
Speaker registration
Speaker identity verification
Alert detection
Mild fatigue detection
High fatigue detection
Speech-rate analysis
Acoustic feature extraction
Personalized baseline creation
Candidate validation
Feature-level deviation analysis
Personalized deviation scoring
Fusion-based fatigue assessment
Risk-level generation
Tech Stack
Python
PyTorch
Streamlit
Whisper
WavLM
ECAPA-TDNN
NumPy
Audio processing libraries
Future Work
Hindi language support
Multilingual voice processing
Larger validation datasets
Adaptive speaker baseline updates
Continuous learning mechanisms
Further model optimization
