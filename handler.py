import os
import gc
import base64
import tempfile
import requests
import runpod
import whisperx

# ------------------------------------------------------------------
#  Runtime configuration (change via pod environment variables)
# ------------------------------------------------------------------
device        = os.getenv("DEVICE", "cuda")           # 'cpu' on a Mac
compute_type  = os.getenv("COMPUTE_TYPE", "float16")  # 'int8' if CPU‑only
batch_size    = int(os.getenv("BATCH_SIZE", "16"))
language_code = os.getenv("LANGUAGE_CODE", "en")
hf_token      = os.getenv("HF_AUTH_TOKEN")            # REQUIRED for diarization
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  Lazy‑loaded global models (keeps cold‑start times low)
# ------------------------------------------------------------------
_whisper_model  = None
_align_model    = None
_align_metadata = None
_diarizer       = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type
        )
    return _whisper_model


def _load_align():
    global _align_model, _align_metadata
    if _align_model is None or _align_metadata is None:
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
    return _align_model, _align_metadata

#Debug to check diarization
print(f"[server] Diarization flag received: {job_inp.get('diarize', False)}")
print(f"[server] HF_AUTH_TOKEN exists: {hf_token is not None}")

def _load_diarizer():
    global _diarizer
    if _diarizer is None:
        if hf_token is None:
            raise RuntimeError(
                "HF_AUTH_TOKEN not set — required for speaker diarization."
            )
        _diarizer = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
    return _diarizer
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#  Utility helpers
# ------------------------------------------------------------------
def _download_to_temp(url: str) -> str:
    """Download audio file from URL → temporary WAV path."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.write(resp.content)
    tf.flush()
    return tf.name


def _base64_to_temp(b64_str: str) -> str:
    """Decode base64 audio → temporary WAV path."""
    data = base64.b64decode(b64_str)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.write(data)
    tf.flush()
    return tf.name
# ------------------------------------------------------------------


def handler(event):
    """
    event["input"] must contain either:
        • "audio_url"      OR
        • "audio_base_64"

    Optional flags:
        • "diarize": true   # run speaker diarization and label words
    """
    job_inp = event.get("input", {})
    audio_path = None

    try:
        # 1️⃣  Get the audio into a temp file ---------------------------------
        if "audio_url" in job_inp:
            audio_path = _download_to_temp(job_inp["audio_url"])
        elif "audio_base_64" in job_inp:
            audio_path = _base64_to_temp(job_inp["audio_base_64"])
        else:
            return {"error": "Provide 'audio_url' or 'audio_base_64'."}

        # 2️⃣  Transcribe -----------------------------------------------------
        model = _load_whisper()
        result = model.transcribe(
            audio_path,
            batch_size=batch_size,
            language=language_code,
            print_progress=False,
        )

        # 3️⃣  Word‑level alignment ------------------------------------------
        align_model, metadata = _load_align()
        result = whisperx.align(
            result["segments"], align_model, metadata, audio_path, device
        )

        # 4️⃣  (Optional) Speaker diarization --------------------------------
        #Original code
        #if job_inp.get("diarize", False):
            #diarizer   = _load_diarizer()
            #dia_result = diarizer(audio_path)               # runs Pyannote
            #result     = whisperx.diarize.assign_word_speakers(
            #    dia_result, result
            #)
        if job_inp.get("diarize", False):
    print(f"[server] Diarization flag received: {job_inp.get('diarize', False)}")
    print(f"[server] HF_AUTH_TOKEN exists: {hf_token is not None}")
    try:
        diarizer = _load_diarizer()
        dia_result = diarizer(audio_path)  # runs Pyannote
        result = whisperx.diarize.assign_word_speakers(dia_result, result)
    except Exception as e:
        print(f"[server] Diarization failed: {e}")

        return result

    except Exception as exc:  # catch *everything* to avoid silent pod crashes
        return {"error": str(exc)}

    finally:
        # tidy up temp file & free GPU VRAM
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        gc.collect()


# ------------------------------------------------------------------
#  Start the serverless handler
# ------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
