"""Traite un ticket depuis la ligne de commande et affiche la trace des agents.

  python scripts/run_ticket.py data/samples/ticket_screenshot.png
  python scripts/run_ticket.py --text "Where is my order NS-128840?"
"""

from __future__ import annotations

import argparse
import json

from support_agent.graph import build_default_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--text")
    args = ap.parse_args()
    if not args.file and not args.text:
        ap.error("donner un fichier ou --text")

    state = build_default_graph().invoke({"source_file": args.file} if args.file else {"ticket_text": args.text})
    print(f"Modalité   : {state.get('modality')}")
    print(f"Ticket     : {state['ticket_text'][:300]}")
    print(f"Intention  : {state['intent']} (confiance {state['confidence']:.2f})")
    reason = f" -- {state['escalation_reason']}" if state["status"] == "escalated" else ""
    print(f"Statut     : {state['status']}{reason}")
    print(f"Sources    : {sorted({d.metadata['source'] for d in state.get('context_docs', [])})}")
    print(f"Réponse    :\n{state['answer']}")
    print("Trace      :", json.dumps(state["trace"]))


if __name__ == "__main__":
    main()
