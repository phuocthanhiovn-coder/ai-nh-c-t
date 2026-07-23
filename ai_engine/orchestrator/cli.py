"""CLI: python -m ai_engine.orchestrator.cli --input anh.jpg --command "..." --output ra.jpg"""
import argparse
import json
import sys

import cv2

cv2.setNumThreads(2)

from .planner import make_plan
from .engine import run_plan


def main():
    parser = argparse.ArgumentParser(description="Orchestrator v0 - chinh anh theo lenh")
    parser.add_argument("--input", type=str, help="Duong dan anh dau vao")
    parser.add_argument("--command", type=str, help="Lenh chinh anh bang ngon ngu tu nhien")
    parser.add_argument("--output", type=str, help="Duong dan anh dau ra")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Chi in plan, khong chay engine")
    parser.add_argument("--plan", type=str, help="JSON plan co san, bo qua LLM/fallback")
    args = parser.parse_args()

    if args.plan:
        parsed = json.loads(args.plan)
        plan = parsed.get("plan", parsed) if isinstance(parsed, dict) else parsed
        source = "explicit"
    else:
        if not args.command:
            print("[ERROR] Can --command hoac --plan.")
            sys.exit(1)
        plan, source = make_plan(args.command)

    print(f"[PLAN] source={source}")
    print(json.dumps({"plan": plan}, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    if not args.input or not args.output:
        print("[ERROR] Can --input va --output khi khong dung --dry-run.")
        sys.exit(1)

    result = run_plan(args.input, plan, args.output)
    print(f"[DONE] in_shape={result['in_shape']} out_shape={result['out_shape']} applied={result['applied']}")
    print(f"[DONE] output: {args.output}")


if __name__ == "__main__":
    main()
