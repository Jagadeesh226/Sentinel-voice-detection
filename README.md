# Sentinel Voice Detection

An AI-based voice analysis system designed to detect potential **fatigue levels from speech**. The system analyzes acoustic, temporal, speech-rate, and deep speech representations, while also supporting **speaker verification and personalized baseline comparison**.

The project is part of the Sentinel fatigue detection pipeline, where voice acts as one of the modalities for assessing a person's fatigue state.

## Features

* Voice-based fatigue classification
* Multi-class fatigue detection:

  * Alert
  * Mild Fatigue
  * High Fatigue
* Acoustic feature extraction
* Speech-rate analysis
* Temporal speech analysis
* WavLM-based deep speech embeddings
* Feature fusion and Transformer-based sequence modeling
* Attention-based pooling
* Speaker verification using ECAPA-TDNN
* Personalized fatigue detection using speaker-specific baselines
* Candidate baseline management
* Voice enrollment workflow
* Real and synthetic speech datasets
* Streamlit application for inference

## System Architecture

The pipeline follows the flow below:

```text
Audio Input
    │
    ▼
Preprocessing
    ├── Audio Loading
    ├── Voice Activity Detection
    └── Normalization
    │
    ▼
Speech Analysis
    ├── Acoustic Features
    ├── Speech Rate
    ├── Temporal Features
    └── WavLM Embeddings
    │
    ▼
Feature Processing
    ├── Normalization
    ├── Projection
    └── Sequence Construction
    │
    ▼
Feature Fusion
    │
    ▼
Transformer Encoder
    │
    ▼
Attention Pooling
    │
    ▼
Fatigue Classifier
    │
    ▼
Alert / Mild Fatigue / High Fatigue
```

## Personalization Pipeline

In addition to general fatigue classification, the system supports speaker-specific analysis.

```text
Speaker Enrollment
        │
        ▼
Speaker Verification
        │
        ▼
Baseline Candidate Collection
        │
        ▼
Baseline Selection
        │
        ▼
Personalized Feature Extraction
        │
        ▼
Deviation from Personal Baseline
        │
        ▼
Personalized Fatigue Inference
```

The personalized approach allows the system to compare a person's current voice characteristics against their own baseline rather than relying only on population-level patterns.

## Project Structure

```text
sentinel_voice/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── baseline_candidates/
│   ├── baselines/
│   ├── dataset/
│   │   ├── real/
│   │   │   ├── paired/
│   │   │   └── unpaired/
│   │   └── synthetic/
│   ├── enrollment/
│   └── raw/
│
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/
│
├── src/
│   ├── features/
│   ├── models/
│   ├── personalization/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── speaker/
│   ├── utils/
│   ├── dataset.py
│   ├── inference.py
│   └── train.py
│
└── test/
    ├── evaluation scripts
    ├── feature tests
    ├── model tests
    ├── personalization tests
    └── pipeline tests
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Jagadeesh226/Sentinel-voice-detection.git
cd Sentinel-voice-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

The project includes both real and synthetic voice samples organized across three fatigue categories:

* **Alert**
* **Mild Fatigue**
* **High Fatigue**

The real dataset is further organized into:

* Paired samples
* Unpaired samples

The project also contains metadata and dataset split files used for training and evaluation.

## Speaker Personalization

The system supports personalized fatigue analysis through speaker-specific baselines.

The workflow includes:

1. Speaker enrollment
2. Speaker identity verification
3. Collection of baseline candidates
4. Selection and storage of stable baseline representations
5. Extraction of features from incoming speech
6. Comparison against the speaker's baseline
7. Personalized fatigue inference

This helps account for natural differences between speakers, such as speaking rate, pitch, energy, and other voice characteristics.

## Testing

The `test/` directory contains scripts for validating individual components and the complete pipeline.

Examples include:

```bash
python test/test_audio.py
python test/test_acoustic.py
python test/test_wavlm.py
python test/test_speaker.py
python test/test_baseline.py
python test/test_end_to_end.py
```

Model evaluation can be performed using:

```bash
python test/evaluate_model.py
```

## Technologies Used

* Python
* PyTorch
* WavLM
* Transformer Encoder
* Attention Mechanisms
* ECAPA-TDNN
* SpeechBrain
* Streamlit
* Librosa / Audio Processing Tools
* NumPy
* Pandas

## Fatigue Detection Classes

| Class        | Description                                          |
| ------------ | ---------------------------------------------------- |
| Alert        | Speech characteristics closer to the alert state     |
| Mild Fatigue | Moderate deviation from alert speech characteristics |
| High Fatigue | Significant deviation associated with higher fatigue |

## Disclaimer

This project is a research and development prototype for voice-based fatigue analysis. It is not intended to be used as a standalone medical or clinical diagnostic system.

## Future Improvements

* Larger and more diverse real-world datasets
* Improved speaker-independent generalization
* Continuous baseline adaptation
* Multilingual voice analysis
* Real-time streaming inference
* Integration with additional fatigue modalities
* Multimodal fatigue detection using voice, facial behavior, eye movement, and reaction time

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
