from src.speaker.verification import SpeakerVerifier


enrollment_file = "data/enrollment/worker_001.wav"
test_file = "data/raw/test.wav"


print("\nLoading speaker verification model...")

verifier = SpeakerVerifier()

score, prediction = verifier.verify(
    enrollment_file,
    test_file
)

print("\n==============================")
print("SPEAKER VERIFICATION")
print("==============================")

print(f"Similarity score: {score:.4f}")

if prediction == 1:
    print("Target speaker: YES")
else:
    print("Target speaker: NO")

print("==============================")