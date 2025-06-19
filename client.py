#!/usr/bin/env python3
"""
pi_client.py – runs on Raspberry Pi

• Action recognition  : MoViNet-A2 INT8 (.tflite + tflite-runtime)
• OCR (scene text)    : PaddleOCR INT8
• Spoken feedback     : gTTS + GStreamer
• Genre classification: delegated to remote server via HTTP
"""
import argparse, threading, time, json
from pathlib import Path

import gi, requests, numpy as np
gi.require_version("Gst", "1.0"); gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

import tflite_runtime.interpreter as tflite
from gtts import gTTS
from paddleocr import PaddleOCR

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SERVER_URL      = "http://<SERVER_IP>:8000/genre"   # ★ 수정
MOVINET_MODEL   = "models/movinet.tflite"
LABEL_FILE      = "labels.txt"

ocr_engine      = PaddleOCR(use_angle_cls=False, lang="en")
with open(LABEL_FILE) as f:
    LABEL_MAP = [ln.strip() for ln in f]

FRAMES = 8  # MoViNet clip length
Gst.init(None)

# --------------------------------------------------------------------------- #
# MoViNet helpers
# --------------------------------------------------------------------------- #
mov_interp = tflite.Interpreter(MOVINET_MODEL)
runner          = mov_interp.get_signature_runner()
input_details   = runner.get_input_details()
movinet_state   = None               # will hold recurrent state between calls

def _qscale(name, state):
  """Scales the named state tensor input for the quantized model."""
  dtype = input_details[name]['dtype']
  scale, zero_point = input_details[name]['quantization']
  if 'frame_count' in name or dtype == np.float32 or scale == 0.0:
    return state
  return np.cast((state / scale + zero_point), dtype)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def run_movinet(frame):
    """Run MoViNet on a single RGB frame (H×W×3, uint8)."""
    global movinet_state
    if movinet_state is None:      # create zero-states once
        movinet_state = {n: _qscale(n, np.zeros(d["shape"], d["dtype"]))
                         for n, d in input_details.items() if n != "image"}

    outputs = runner(image=_qscale("image", frame), **movinet_state)
    logits  = outputs.pop("logits")[-1]             # last-frame logits
    movinet_state = outputs                         # carry state forward

    probs = softmax(logits)
    top_idx = int(probs.argmax())
    return f"action_{top_idx}", float(probs[top_idx])

# --------------------------------------------------------------------------- #
# OCR helper
# --------------------------------------------------------------------------- #
def run_ocr(frame):
    """PaddleOCR on a single RGB frame; returns best text ('' if none)."""
    results = ocr_engine.ocr(frame)
    cand = [(txt, sc)
            for line in results or []
            for txt, sc in zip(line["rec_texts"], line["rec_scores"])]
    if not cand:
        return ""
    return max(cand, key=lambda x: x[1])[0]         # highest-score text

# --------------------------------------------------------------------------- #
# Genre request + TTS
# --------------------------------------------------------------------------- #
def speak_genre(text):
    if not text:
        return
    prompt = f"Text on screen: {text}"
    try:
        resp = requests.post(SERVER_URL, json={"prompt": prompt}, timeout=5)
        genre = resp.json()["genre"]
    except Exception as e:
        print("[Genre] request failed:", e); return

    print(f"[Genre] {genre}")
    mp3 = Path(f"tts_{int(time.time())}.mp3")
    gTTS(f"The predicted genre is {genre}.", lang="en").save(mp3)

    player = Gst.parse_launch(
        f"filesrc location={mp3} ! decodebin ! audioconvert ! audioresample ! autoaudiosink")
    player.set_state(Gst.State.PLAYING)

    def _cleanup():
        player.set_state(Gst.State.NULL); mp3.unlink(missing_ok=True)
        return False
    GLib.timeout_add_seconds(6, _cleanup)

# --------------------------------------------------------------------------- #
# GStreamer pipeline + callback
# --------------------------------------------------------------------------- #
def on_video(sink, _):
    sample = sink.emit("pull-sample")
    buf    = sample.get_buffer()
    caps   = sample.get_caps()
    h = caps.get_structure(0).get_value("height")
    w = caps.get_structure(0).get_value("width")
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.ERROR
    frame = np.frombuffer(info.data, np.uint8).reshape(h, w, 3)
    buf.unmap(info)

    act_lbl, act_p = run_movinet(frame)
    txt = run_ocr(frame)
    print(f"[Infer] {act_lbl} (p={act_p:.2f}) | OCR='{txt}'")
    speak_genre(txt)
    return Gst.FlowReturn.OK

def build_pipeline(video_path):
    desc = (
        f"filesrc location={video_path} ! decodebin name=d "
        "d. ! queue ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=224,height=224 ! "
        "appsink name=video_sink emit-signals=true sync=false"
    )
    return Gst.parse_launch(desc)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True)
    args = ap.parse_args()
    if not Path(args.video).exists():
        ap.error(f"video '{args.video}' not found")

    pipe = build_pipeline(args.video)
    sink = pipe.get_by_name("video_sink"); sink.connect("new-sample", on_video, None)
    bus  = pipe.get_bus(); bus.add_signal_watch()
    loop = GLib.MainLoop()

    def _bus(_, msg):  # error / EOS handling
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error(); print("[GST] ERROR:", err, dbg); loop.quit()
        elif t == Gst.MessageType.EOS:
            loop.quit(); print("[GST] EOS")
        return True
    bus.connect("message", _bus, None)

    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()