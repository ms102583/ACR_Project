#!/usr/bin/env python3
"""
pi_movinet_client.py
• Action recognition : MoViNet-A2 INT8 (.tflite)
• OCR + Genre CLS    : 원격 서버에 HTTP POST
• Spoken feedback    : gTTS + GStreamer
"""
import re
import argparse, base64, io, threading, time, requests
from pathlib import Path

import numpy as np, gi
gi.require_version("Gst", "1.0"); gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib
from PIL import Image
from gtts import gTTS
import tflite_runtime.interpreter as tflite

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVINET_MODEL = "models/movinet-int8.tflite"
SERVER_URL    = "http://<SERVER_IP>:8000/ocr_genre"   # ★ 서버 주소만 바꾸세요
with open("labels.txt") as f:
    label_map = [ln.strip() for ln in f]

# ---------------------------------------------------------------------------
# 초기화 (MoViNet만 유지)
# ---------------------------------------------------------------------------
Gst.init(None)
mov_interp   = tflite.Interpreter(MOVINET_MODEL)
runner       = mov_interp.get_signature_runner()
input_details = runner.get_input_details()
movinet_prev_state = None    # recurrent states

def _qscale(name, state):
  """Scales the named state tensor input for the quantized model."""
  dtype = input_details[name]['dtype']
  scale, zero_point = input_details[name]['quantization']
  if 'frame_count' in name or dtype == np.float32 or scale == 0.0:
    return state
  return np.cast((state / scale + zero_point), dtype)

def get_top_p(probs, p=0.8, max_k=5, label_map=None):
    """Outputs the top model labels and probabilities on the given video."""
    # probs: 1D numpy array
    # label_map: list of string labels
    top_predictions = np.argsort(probs)[::-1]   # 내림차순 top k 인덱스
    i = 0
    total_prob = 0.0
    while i < min(len(top_predictions), max_k) and total_prob < p:
        total_prob += probs[top_predictions[i]]
        i += 1
    top_predictions = top_predictions[:i]  # cut off at p threshold
    top_labels = [label_map[idx] for idx in top_predictions]
    top_probs = probs[top_predictions]
    return list(zip(top_labels, top_probs))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def run_movinet(frame):
    global movinet_prev_state
    frame = frame.astype(np.float32)  # ensure uint8 type

    if movinet_prev_state is None:
        movinet_prev_state = {n: _qscale(n, np.zeros(d["shape"], d["dtype"]))
                              for n, d in input_details.items() if n != "image"}
    outputs = runner(image=_qscale("image", frame), **movinet_prev_state)
    logits  = outputs.pop("logits")[-1]
    movinet_prev_state = outputs
    probs = softmax(logits); idx = int(probs.argmax())
    return get_top_p(probs, p=0.8, label_map=label_map)

# ---------------------------------------------------------------------------
# 원격 OCR + Genre 호출
# ---------------------------------------------------------------------------
def remote_ocr_genre(frame, movinet_results):
    pil = Image.fromarray(frame)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        r = requests.post(SERVER_URL, json={"image": b64, }, timeout=5)
        data = r.json()
        return data.get("text", ""), data.get("genre", "unknown")
    except Exception as e:
        print("[Server] request failed:", e)
        return "", "unknown"

def speak_tts(text):
    mp3 = Path(f"tts_{int(time.time())}.mp3")
    gTTS(text, lang="en").save(mp3)
    player = Gst.parse_launch(
        f"filesrc location={mp3} ! decodebin ! audioconvert ! audioresample ! autoaudiosink")
    player.set_state(Gst.State.PLAYING)
    def _cleanup():
        player.set_state(Gst.State.NULL); mp3.unlink(missing_ok=True); return False
    GLib.timeout_add_seconds(6, _cleanup)

# ---------------------------------------------------------------------------
# appsink 콜백
# ---------------------------------------------------------------------------
def on_video(sink, _):
    sample = sink.emit("pull-sample")
    buf    = sample.get_buffer()
    caps   = sample.get_caps()
    h = caps.get_structure(0).get_value('height')
    w = caps.get_structure(0).get_value('width')
    ok, info = buf.map(Gst.MapFlags.READ);   # RGB (224×224×3)
    if not ok: return Gst.FlowReturn.ERROR
    frame = np.frombuffer(info.data, np.uint8).reshape(h, w, 3)
    buf.unmap(info)

    movinet_results = run_movinet(frame)
    # print(f"[MoviNet] {act_lbl} p={act_p:.2f}")

    print("[MoviNet]")
    print("\t".join([f"{act_lbl} (p={act_p:.2f})" for act_lbl, act_p in movinet_results]))
    
    f.write("\t".join([act_lbl for act_lbl, _ in movinet_results]) + "\n")
    f.flush()

    # text, genre = remote_ocr_genre(frame, movinet_results)
    # print(f"[OCR] '{text}'  |  [Genre] {genre}")
    # speak_tts(f"The predicted genre is {genre}.")

    return Gst.FlowReturn.OK

def build_pipeline(video_path):
    desc = (f"filesrc location={video_path} ! decodebin name=d "
            "d. ! queue ! videoconvert ! videoscale "
            "! video/x-raw,format=RGB,width=224,height=224 "
            "! appsink name=video_sink emit-signals=true sync=false")
    return Gst.parse_launch(desc)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global f
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True)
    ap.add_argument("--movinet_res", required=True)
    args = ap.parse_args()

    f = open(args.movinet_res, "w")

    pipeline = build_pipeline(args.video)
    sink = pipeline.get_by_name("video_sink")
    sink.connect("new-sample", on_video, None)

    bus = pipeline.get_bus(); bus.add_signal_watch()
    loop = GLib.MainLoop()
    bus.connect("message", lambda _, m:
        (loop.quit() if m.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS) else True))

    pipeline.set_state(Gst.State.PLAYING)
    try: loop.run()
    except KeyboardInterrupt: pass
    finally: pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()