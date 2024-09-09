from faster_whisper import WhisperModel

model_size = "NbAiLabBeta/nb-whisper-base"
cpu_supported_compute_types = {"small": "int8", "large": "float32"}
precision = "small"

model = WhisperModel(
    model_size,
    device="cpu",
    compute_type=cpu_supported_compute_types[precision],
)

segments, info = model.transcribe("data/king.mp3", beam_size=5, language="no")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))