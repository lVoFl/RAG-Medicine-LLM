import argparse
import json
import sys
from faster_whisper import WhisperModel


def parse_args():
    parser = argparse.ArgumentParser(description="Local ASR with faster-whisper")
    parser.add_argument("--audio", required=False, help="Audio file path")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    parser.add_argument("--compute-type", default="auto", help="int8/float16/auto")
    parser.add_argument("--language", default="zh", help="language hint, e.g. zh/en")
    parser.add_argument("--serve", action="store_true", help="Run as persistent stdin/stdout worker")
    return parser.parse_args()


def transcribe_once(model, audio_path: str, language: str):
    segments, info = model.transcribe(audio_path, language=language, vad_filter=True)

    text_parts = []
    seg_items = []
    for seg in segments:
        content = (seg.text or "").strip()
        if not content:
            continue
        text_parts.append(content)
        seg_items.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": content,
            }
        )

    payload = {
        "text": "".join(text_parts).strip(),
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_items,
    }
    return payload


def run_serve_mode(args):
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    for raw_line in sys.stdin:
        line = (raw_line or "").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            audio_path = str(req.get("audio") or "").strip()
            language = str(req.get("language") or args.language).strip() or args.language
            if not audio_path:
                raise ValueError("audio path is required")
            payload = transcribe_once(model, audio_path, language)
            resp = {"id": req_id, "ok": True, "result": payload}
        except Exception as exc:
            resp = {"id": req.get("id") if isinstance(req, dict) else None, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    args = parse_args()
    if args.serve:
        run_serve_mode(args)
        return

    if not args.audio:
        raise ValueError("--audio is required when not using --serve")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    payload = transcribe_once(model, args.audio, args.language)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
