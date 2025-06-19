#!/usr/bin/env python3
"""
local_experiment.py – GStreamer + TFLite + HuggingFace + gTTS

• Action recognition  : MoViNet-A2 INT8 (.tflite + tflite-runtime)
• OCR (scene text)    : DBNet INT8 + CRNN INT8
• Zero-shot genre CLS : facebook/bart-large-mnli (PyTorch)
• Spoken feedback     : gTTS + GStreamer
"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

import tflite_runtime.interpreter as tflite
import torch
from transformers import pipeline as hf_pipeline
from gtts import gTTS
from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVINET_MODEL = "models/movinet.tflite"

GENRE_LABELS = [
    "sports", "news", "cooking", "travel",
    "music", "drama", "comedy", "documentary",
]

FRAMES = 8  # movinet clip length

with open('labels.txt') as f:
    label_map = [line.strip() for line in f]

def get_top_k(probs, k=5, label_map=None):
    """Outputs the top k model labels and probabilities on the given video."""
    # probs: 1D numpy array
    # label_map: list of string labels
    top_predictions = np.argsort(probs)[::-1][:k]   # 내림차순 top k 인덱스
    top_labels = [label_map[idx] for idx in top_predictions]
    top_probs = probs[top_predictions]
    return list(zip(top_labels, top_probs))

# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------
Gst.init(None)

# MoViNet TFLite interpreter
mov_interp = tflite.Interpreter(MOVINET_MODEL)
mov_interp.allocate_tensors()
mov_in_details  = mov_interp.get_input_details()
mov_out_details = mov_interp.get_output_details()


image_in_idx = [d['index'] for d in mov_in_details if 'image' in d['name']][0]
image_in_detail = [d for d in mov_in_details if 'image' in d['name']][0]
state_in_details = [d for d in mov_in_details if 'image' not in d['name']]

logits_out_detail = [d for d in mov_out_details if tuple(d['shape']) == (1, 600)][0]
logits_out_idx = logits_out_detail['index']
state_out_name_to_idx = {d['name']: d['index'] for d in mov_out_details if tuple(d['shape']) != (1, 600)}

def init_state_dict(state_details):
    state = {}
    for d in state_details:
        state[d['name']] = np.zeros(d['shape'], dtype=d['dtype'])
    return state


ocr_engine = PaddleOCR(use_angle_cls=False, lang="en")

# Genre classifier (zero-shot)
device = 0 if torch.cuda.is_available() else -1

genre_clf = hf_pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=device)

# 공유 버퍼
action_frames = []
action_lock   = threading.Lock()
ocr_text      = None

# ---------------------------------------------------------------------------
# 후처리 헬퍼
# ---------------------------------------------------------------------------

def quantize_input(arr, detail):
    scale, zp = detail['quantization']
    if scale == 0:
        return arr.astype(detail['dtype'])
    q = (arr.astype(np.float32) / scale + zp).round().astype(detail['dtype'])
    return q
movinet_prev_state = None

def bus_call(bus, msg, loop):
    t = msg.type
    if t == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print("[GST] ERROR:", err, dbg)
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("[GST] End-of-stream")
        loop.quit()
    return True

def run_ocr(frame):
    results = ocr_engine.ocr(frame)

    candidates = []
    if results:
        for line in results:
            #for (text, score) in zip(line['rec_texts'], line['rec_scores']):
            #    candidates.append((text, score))
            if len(line['rec_texts']) > 0:
                candidates.append(max(zip(line['rec_texts'], line['rec_scores']), key=lambda x: x[1])[0])
            
    if not candidates:
        return ""

    # best = max(candidates, key=lambda x: x[1])

    # (txt, score) = best
    return candidates

def emit_genre_and_tts(text, movinet_result):
    # prompt = f"A person is {label}. Text on screen: {text}"
    prompt = f"Texts on screen: {', '.join(text)}\nDetected actions on screen, ranked by likelihood: {', '.join(movinet_result)}"
    res = genre_clf(prompt, GENRE_LABELS, multi_label=False)
    top_label = res["labels"][0]
    print(f"[Genre] {top_label} (scores: {dict(zip(res['labels'],res['scores']))})")
    # TTS 저장 및 재생
    tts_file = Path(f"tts_{int(time.time())}.mp3")
    gTTS(f"The predicted genre is {top_label}.", lang="en").save(tts_file)
    # GStreamer로 재생
    player = Gst.parse_launch(
        f"filesrc location={tts_file} ! decodebin ! audioconvert ! audioresample ! fakesink"
    )
    player.set_state(Gst.State.PLAYING)
    # 6초 후 정리
    def _cleanup():
        player.set_state(Gst.State.NULL)
        tts_file.unlink(missing_ok=True)
        return False
    GLib.timeout_add_seconds(6, _cleanup)

# ---------------------------------------------------------------------------
# appsink 콜백
# ---------------------------------------------------------------------------
def on_video(sink, _):
    movinet_result = f.readline().strip().split("\t")

    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()
    h = caps.get_structure(0).get_value('height')
    w = caps.get_structure(0).get_value('width')
    success, info = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR
    arr = np.frombuffer(info.data, np.uint8).reshape(h, w, 3)
    buf.unmap(info)
    # 여기서 frame(arr)만 run_movinet에 넘김
    
    txt = run_ocr(arr)
    # print(f"[Infer] {act_lbl} (p={act_p:.2f}) | OCR='{txt}'")
    # print(f"[Infer] OCR='{txt}'")
    emit_genre_and_tts(txt, movinet_result)
    return Gst.FlowReturn.OK

# ---------------------------------------------------------------------------
# 파이프라인 구성
# ---------------------------------------------------------------------------
def build_pipeline(video_path):
    desc = (
        f"filesrc location={video_path} ! decodebin name=d "
        "d. ! queue ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=224,height=224 ! "
        "appsink name=video_sink emit-signals=true sync=false"
    )
    return Gst.parse_launch(desc)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global f

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--movinet_res", required=True)
    args = parser.parse_args()

    f = open(args.movinet_res, "r")

    if not Path(args.video).exists():
        parser.error(f"Input video '{args.video}' not found")

    pipeline = build_pipeline(args.video)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    

    print("Pipeline built successfully")
    sink = pipeline.get_by_name("video_sink")
    print("sink found:", sink)
    sink.connect("new-sample", on_video, None)

    print("sink connected")

    loop = GLib.MainLoop()
    bus.connect("message", bus_call, loop)

    print("loop created")
    pipeline.set_state(Gst.State.PLAYING)
    print("pipeline set to PLAYING")
    try:
        loop.run()
        print("loop running")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
