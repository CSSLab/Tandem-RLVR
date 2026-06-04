from pathlib import Path

"""
Prototype: Board-grounded chess CoT claim verifier.

This script reads one experiment record, extracts a FEN, candidate moves,
thinking/CoT text, parses natural-language chess reasoning into coarse
atomic claims, and evaluates them with python-chess.

It is intentionally heuristic. The goal is to capture the framework:
  CoT steps -> claims -> atomic claims -> python-chess verification
  plus rough state-by-state scoring with reject/contaminated branches.

Install:
  pip install python-chess

Run:
  python chess_cot_claim_verifier.py --input "Pasted text(23).txt"
  python chess_cot_claim_verifier.py --input "Pasted text(23).txt" --json-out result.json
"""
import argparse
import json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import chess
except ImportError as e:
    raise SystemExit(
        "Missing dependency: python-chess. Install it with:\n"
        "  pip install python-chess"
    ) from e


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class AtomicClaim:
    step_id: int
    raw_text: str
    claim_type: str
    subject: str
    params: Dict[str, Any]
    board_scope: str = "pre"  # pre, post:<uci>, sequence:<uci uci ...>, unknown
    verdict: str = "unknown"  # true, false, unknown
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class StepEvaluation:
    step_id: int
    text: str
    claims: List[AtomicClaim] = field(default_factory=list)
    factuality: Optional[float] = None
    support: float = 0.0
    is_true_step: bool = False
    is_false_step: bool = False
    is_mixed_step: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class ReasoningState:
    state_id: str
    accepted_claims: List[AtomicClaim] = field(default_factory=list)
    rejected_claims: List[AtomicClaim] = field(default_factory=list)
    contaminated_claims: List[AtomicClaim] = field(default_factory=list)
    score: float = 0.0
    path: List[str] = field(default_factory=list)


# -----------------------------
# Utilities
# -----------------------------

PIECE_NAME_TO_TYPE = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}

PIECE_LETTER_TO_TYPE = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
    "P": chess.PAWN,
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def load_record(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON object or a JSONL file. If JSONL, returns the first non-empty object.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()

    # Try full JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass

    # Try JSONL.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse input as JSON object or JSONL.")


def extract_fen(record: Dict[str, Any]) -> str:
    """
    Prefer the record's input field, then thinking/raw_content.
    """
    candidates = [
        record.get("input", ""),
        record.get("thinking", ""),
        record.get("raw_content", ""),
    ]
    fen_pattern = re.compile(
        r'(?:"|FEN:?\s*)'
        r'([rnbqkpRNBQKP1-8/]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+)'
        r'(?:"|$)?'
    )
    for text in candidates:
        m = fen_pattern.search(text)
        if m:
            return m.group(1)

    # More permissive fallback.
    fallback = re.compile(r'([rnbqkpRNBQKP1-8/]{15,}\s+[wb]\s+\S+\s+\S+\s+\d+\s+\d+)')
    for text in candidates:
        m = fallback.search(text)
        if m:
            return m.group(1)

    raise ValueError("Could not find FEN.")


def extract_candidate_moves(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract MoveA:f5f2 / MoveB:f5b1 from input or thinking.
    """
    text = "\n".join([
        str(record.get("input", "")),
        str(record.get("thinking", "")),
        str(record.get("raw_content", "")),
    ])
    moves: Dict[str, str] = {}
    for label, uci in re.findall(r'\b(Move[A-Z])\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b', text):
        moves[label] = uci
    return moves


def extract_answer(record: Dict[str, Any]) -> Optional[str]:
    for key in ["answer", "parsed", "expected"]:
        val = record.get(key)
        if isinstance(val, str):
            m = re.search(r'\bMove[A-Z]\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b', val)
            if m:
                return m.group(1)
            m = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', val)
            if m:
                return m.group(1)
    return None


def get_cot_text(record: Dict[str, Any]) -> str:
    """
    Use thinking first. raw_content sometimes contains the full <think> block too.
    """
    return str(record.get("thinking") or record.get("raw_content") or "")


def split_steps(cot: str) -> List[str]:
    """
    Split chain-of-thought into rough reasoning steps.
    Handles newlines, explicit markers, and sentence boundaries.
    """
    cot = cot.replace("\r\n", "\n")
    cot = re.sub(r'</?think>', '\n', cot)
    cot = re.sub(r'\[(Done|Final Check|Output Generation|Final Answer Generation|Output Generation)\]', r'\n[\1]', cot)

    # First split by newlines; then split long paragraphs by sentence-ish boundaries.
    chunks: List[str] = []
    for para in cot.split("\n"):
        para = para.strip()
        if not para:
            continue
        # Keep board-piece list lines intact when they contain many commas.
        if para.startswith(("White:", "Black:", "MoveA:", "MoveB:", "TacticA:", "TacticB:", "FEN:")):
            chunks.append(para)
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\[])', para)
        chunks.extend([p.strip() for p in parts if p.strip()])

    # Remove very short meta-output repetitions if desired.
    return chunks


def square_name(s: str) -> Optional[chess.Square]:
    try:
        return chess.parse_square(s)
    except Exception:
        return None


def piece_name(piece: Optional[chess.Piece]) -> str:
    if piece is None:
        return "empty"
    color = "White" if piece.color == chess.WHITE else "Black"
    return f"{color} {piece.symbol().upper()}"


def board_after_sequence(board: chess.Board, moves: Iterable[str]) -> Tuple[Optional[chess.Board], str]:
    b = board.copy()
    seq = []
    for uci in moves:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return None, f"invalid UCI {uci}"
        if mv not in b.legal_moves:
            return None, f"illegal move {uci} on board {b.fen()}"
        b.push(mv)
        seq.append(uci)
    return b, f"after sequence {' '.join(seq)}"


def material_score(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for sq, p in board.piece_map().items():
        val = PIECE_VALUES[p.piece_type]
        total += val if p.color == color else -val
    return total


def move_material_delta(board: chess.Board, sequence: List[str], color: chess.Color) -> Optional[int]:
    before = material_score(board, color)
    b, _ = board_after_sequence(board, sequence)
    if b is None:
        return None
    after = material_score(b, color)
    return after - before


# -----------------------------
# Claim extraction
# -----------------------------

def extract_claims_from_step(step_id: int, text: str, board: chess.Board, candidate_moves: Dict[str, str]) -> List[AtomicClaim]:
    """
    Heuristic claim extractor.
    It maps common chess-CoT text patterns to atomic claims.

    Supported examples:
      - "f2 is a pawn"
      - "c1 is empty"
      - "Black Queen on f5"
      - "black to move"
      - "Qxf2+ Kxf2"
      - "f5f2 is legal"
      - "Qb1 attacks b2"
      - "Qxf2+ loses the queen"
      - "Re8 is checkmate"
    """
    claims: List[AtomicClaim] = []
    lower = text.lower()

    def add(claim_type: str, subject: str, params: Dict[str, Any], scope: str = "pre", conf: float = 1.0):
        claims.append(AtomicClaim(
            step_id=step_id,
            raw_text=text,
            claim_type=claim_type,
            subject=subject,
            params=params,
            board_scope=scope,
            confidence=conf,
        ))

    # Side to move claims.
    if re.search(r"\bblack(?:'s)?\s+(?:turn|to move)\b|\bit'?s black'?s turn\b", lower):
        add("side_to_move", "black to move", {"color": "black"})
    white_match = re.search(r"\bwhite(?:'s)?\s+(?:turn|to move)\b|\bit'?s white'?s turn\b", lower)
    if white_match:
        # Avoid false positive in "White to move? No" by inspecting the text that
        # immediately follows the phrase for a negation token.
        trailing = lower[white_match.end(): white_match.end() + 15]
        if not re.search(r'\bno\b|\bnot\b|\bnope\b', trailing):
            add("side_to_move", "white to move", {"color": "white"})

    # Board piece placement lines like:
    # White: King on g1, Queen on c6, Pawns on a4, d4, ...
    for color_word in ["White", "Black"]:
        if text.strip().startswith(color_word + ":"):
            color = chess.WHITE if color_word == "White" else chess.BLACK
            # The square list must stop at the next piece word; otherwise a greedy
            # capture attributes every square on the line to the first piece. We only
            # consume square-list characters (files a-h, ranks 1-8, commas, spaces).
            for piece_word, squares_blob in re.findall(
                r'(King|Queen|Rook|Bishop|Knight|Pawn|Pawns|Bishops|Knights|Rooks|Queens)\s+on\s+([a-h1-8,\s]+)',
                text,
                flags=re.I,
            ):
                pword = piece_word.lower().rstrip("s")
                ptype = PIECE_NAME_TO_TYPE.get(pword)
                if ptype is None:
                    continue
                for sq_str in re.findall(r'\b[a-h][1-8]\b', squares_blob):
                    add("piece_on_square", f"{color_word} {pword} on {sq_str}", {
                        "square": sq_str,
                        "piece_type": ptype,
                        "color": "white" if color == chess.WHITE else "black",
                    })


    # Corrected parse for "f2 is a pawn" with optional color.
    for m in re.finditer(r'\b([a-h][1-8])\s+is\s+(?:a|an)?\s*(?:(white|black)\s+)?(king|queen|rook|bishop|knight|pawn|empty)\b', lower):
        sq, color_word, pword = m.group(1), m.group(2), m.group(3)
        if pword == "empty":
            add("square_empty", f"{sq} is empty", {"square": sq})
        else:
            add("piece_on_square", f"{pword} on {sq}", {
                "square": sq,
                "piece_type": PIECE_NAME_TO_TYPE[pword],
                "color": color_word,  # may be None
            })

    # "Black Queen on f5", "White King on g1".
    for color_word, piece_word, sq in re.findall(
        r'\b(White|Black)\s+(King|Queen|Rook|Bishop|Knight|Pawn)\s+on\s+([a-h][1-8])\b',
        text,
        flags=re.I,
    ):
        add("piece_on_square", f"{color_word} {piece_word} on {sq}", {
            "square": sq,
            "piece_type": PIECE_NAME_TO_TYPE[piece_word.lower()],
            "color": color_word.lower(),
        })

    # Explicit UCI moves: f5f2, f5b1, g1f2, etc.
    for uci in re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text):
        try:
            chess.Move.from_uci(uci)
        except ValueError:
            continue
        # If the text is discussing the move as a move, verify legality.
        if any(word in lower for word in ["move", "play", "taking", "takes", "capture", "recapture", "qxf", "qxb", "q"]):
            add("move_legal", f"{uci} legal", {"move": uci})

    # Candidate label references.
    for label, uci in candidate_moves.items():
        if label.lower() in lower:
            add("candidate_move_exists", f"{label} is {uci}", {"label": label, "move": uci}, conf=0.5)

    # Capture/check notation such as Qxf2+ and Kxf2.
    # Convert SAN-ish pattern if possible by trying legal SAN on current board and after candidate moves.
    san_like = re.findall(r'\b([KQRBN]?[a-h]?x?[a-h][1-8][+#]?|[KQRBN][a-h][1-8][+#]?)\b', text)
    for san in san_like:
        # Avoid treating square-only tokens as SAN.
        if re.fullmatch(r'[a-h][1-8]', san):
            continue
        add("san_mentioned", f"SAN-like move {san}", {"san": san}, conf=0.4)

    # Attacks claims: "Qb1 attacks b2", "queen on b1 attacks b2", "attacks the pawn on b2"
    # Pattern 1: "Qb1 attacks b2"
    for mover, target in re.findall(r'\b([KQRBN]?[a-h][1-8])\s+attacks?\s+(?:the\s+\w+\s+on\s+)?([a-h][1-8])\b', text, flags=re.I):
        add("attacks_square_after_move_or_from_square", f"{mover} attacks {target}", {
            "attacker_expr": mover,
            "target": target,
        })

    # Pattern 2: "Queen on b1 attacks b2"
    for piece_word, from_sq, target_sq in re.findall(
        r'\b(queen|rook|bishop|knight|king|pawn)\s+on\s+([a-h][1-8])\s+attacks?\s+(?:the\s+\w+\s+on\s+)?([a-h][1-8])\b',
        lower,
    ):
        add("piece_attacks_square", f"{piece_word} on {from_sq} attacks {target_sq}", {
            "piece_type": PIECE_NAME_TO_TYPE[piece_word],
            "from": from_sq,
            "target": target_sq,
        })

    # "cannot capture", "can capture", "recaptures with King/Kxf2".
    # This is approximated through SAN/UCI sequence evaluation elsewhere.
    if "recapture" in lower or "recaptures" in lower or "kxf2" in lower:
        add("recapture_possible_contextual", "recapture is possible", {
            "text": text,
        }, conf=0.5)

    # Check/checkmate claims around explicit UCI moves.
    # A mate/check annotation refers to the *last* move of any UCI line in the step
    # (e.g. "d2d8 b8d8 d1d8 Checkmate!" -> d1d8 is mate after the preceding moves).
    # Use whole-word matching so "material"/"estimate" do not trigger "mate".
    ucis_in_step = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text)
    mentions_mate = re.search(r'\b(checkmate|checkmated|mate|mates|mated)\b', lower) is not None
    mentions_check = re.search(r'\bchecks?\b|\bchecking\b|\bchecked\b', lower) is not None
    if ucis_in_step and (mentions_mate or mentions_check):
        move = ucis_in_step[-1]
        setup = ucis_in_step[:-1]
        scope = f"sequence:{' '.join(setup)}" if setup else "pre"
        if mentions_mate:
            add("move_is_checkmate", f"{move} is checkmate", {"move": move, "setup": setup}, scope=scope)
        else:
            add("move_gives_check", f"{move} gives check", {"move": move, "setup": setup}, scope=scope)

    # "loses the queen", "down Queen", "loses Queen for pawn"
    if re.search(r'los(?:es|ing)\s+(?:the\s+)?queen|down\s+queen|loses queen for (?:a )?pawn', lower):
        # Attach to any first candidate move in sentence, else unknown.
        ucis = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text)
        add("material_loss_queen_contextual", "side loses queen after sequence/context", {
            "moves_in_text": ucis,
            "text": text,
        }, conf=0.6)

    # "safe", "keeps the queen", "maintains material".
    if re.search(r'\bkeeps?\s+(?:the\s+)?queen|maintains material|safe\b', lower):
        ucis = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text)
        add("material_not_lost_contextual", "move does not immediately lose major material", {
            "moves_in_text": ucis,
            "text": text,
        }, conf=0.4)

    # "c1 is empty", "there isn't [a piece]".
    for sq in re.findall(r'\b([a-h][1-8])\s+is\s+empty\b', lower):
        add("square_empty", f"{sq} is empty", {"square": sq})

    return claims


# -----------------------------
# Claim evaluation
# -----------------------------

def eval_atomic_claim(claim: AtomicClaim, board: chess.Board, candidate_moves: Dict[str, str]) -> AtomicClaim:
    """
    Evaluate one atomic claim with python-chess where possible.
    Mutates and returns the claim.
    """
    c = claim
    p = c.params

    def set_verdict(v: str, evidence: str):
        c.verdict = v
        c.evidence = evidence
        return c

    if c.claim_type == "side_to_move":
        expected = chess.WHITE if p["color"] == "white" else chess.BLACK
        return set_verdict("true" if board.turn == expected else "false", f"board.turn={board.turn}")

    if c.claim_type == "piece_on_square":
        sq = square_name(p["square"])
        if sq is None:
            return set_verdict("unknown", "invalid square")
        piece = board.piece_at(sq)
        if piece is None:
            return set_verdict("false", f"{p['square']} is empty")
        type_ok = piece.piece_type == p["piece_type"]
        color = p.get("color")
        color_ok = True
        if color in ("white", "black"):
            color_ok = piece.color == (chess.WHITE if color == "white" else chess.BLACK)
        verdict = "true" if type_ok and color_ok else "false"
        return set_verdict(verdict, f"{p['square']} contains {piece_name(piece)}")

    if c.claim_type == "square_empty":
        sq = square_name(p["square"])
        if sq is None:
            return set_verdict("unknown", "invalid square")
        piece = board.piece_at(sq)
        return set_verdict("true" if piece is None else "false", f"{p['square']} contains {piece_name(piece)}")

    if c.claim_type == "move_legal":
        try:
            mv = chess.Move.from_uci(p["move"])
        except ValueError:
            return set_verdict("false", "invalid UCI")
        return set_verdict("true" if mv in board.legal_moves else "false", f"legal={mv in board.legal_moves}")

    if c.claim_type == "move_gives_check":
        setup = p.get("setup", [])
        if setup:
            b, msg = board_after_sequence(board, setup)
            if b is None:
                return set_verdict("unknown", f"setup sequence not playable: {msg}")
        else:
            b, msg = board, "current board"
        try:
            mv = chess.Move.from_uci(p["move"])
        except ValueError:
            return set_verdict("false", "invalid UCI")
        if mv not in b.legal_moves:
            return set_verdict("false", f"move is illegal on {msg}, so cannot be a legal checking move")
        return set_verdict("true" if b.gives_check(mv) else "false", f"gives_check={b.gives_check(mv)} on {msg}")

    if c.claim_type == "move_is_checkmate":
        setup = p.get("setup", [])
        if setup:
            b, msg = board_after_sequence(board, setup)
            if b is None:
                return set_verdict("unknown", f"setup sequence not playable: {msg}")
        else:
            b, msg = board.copy(), "current board"
        try:
            mv = chess.Move.from_uci(p["move"])
        except ValueError:
            return set_verdict("false", "invalid UCI")
        if mv not in b.legal_moves:
            return set_verdict("false", f"move is illegal on {msg}, so cannot be checkmate")
        b.push(mv)
        return set_verdict("true" if b.is_checkmate() else "false", f"post_move_checkmate={b.is_checkmate()} on {msg}")

    if c.claim_type == "piece_attacks_square":
        from_sq = square_name(p["from"])
        target = square_name(p["target"])
        if from_sq is None or target is None:
            return set_verdict("unknown", "invalid square")
        piece = board.piece_at(from_sq)
        if piece is None:
            return set_verdict("false", f"{p['from']} is empty")
        if piece.piece_type != p["piece_type"]:
            return set_verdict("false", f"{p['from']} contains {piece_name(piece)}, not claimed piece")
        attacks = board.attacks(from_sq)
        return set_verdict("true" if target in attacks else "false", f"attacks={p['target'] in [chess.square_name(s) for s in attacks]}")

    if c.claim_type == "attacks_square_after_move_or_from_square":
        expr = p["attacker_expr"]
        target_sq = square_name(p["target"])
        if target_sq is None:
            return set_verdict("unknown", "invalid target")

        # If expr is like b1 or Qb1, evaluate as a piece located on that square
        # after a candidate move to that square, if available.
        sq_match = re.search(r'([a-h][1-8])$', expr)
        if not sq_match:
            return set_verdict("unknown", "could not parse attacker expression")
        from_sq_str = sq_match.group(1)
        from_sq = square_name(from_sq_str)

        # First evaluate on current board.
        if from_sq is not None and board.piece_at(from_sq):
            attacks = target_sq in board.attacks(from_sq)
            return set_verdict("true" if attacks else "false", f"current board: {from_sq_str} attacks {p['target']} = {attacks}")

        # Then try after candidate moves ending on from_sq.
        for label, uci in candidate_moves.items():
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if chess.square_name(mv.to_square) == from_sq_str and mv in board.legal_moves:
                b = board.copy()
                b.push(mv)
                piece = b.piece_at(mv.to_square)
                attacks = piece is not None and target_sq in b.attacks(mv.to_square)
                return set_verdict("true" if attacks else "false", f"after {label}:{uci}, {from_sq_str} attacks {p['target']} = {attacks}")

        return set_verdict("unknown", "attacker square empty on current board and no matching candidate move")

    if c.claim_type == "san_mentioned":
        # Try to parse SAN on current board.
        san = p["san"]
        try:
            mv = board.parse_san(san)
            return set_verdict("true", f"SAN parses as legal move {mv.uci()} on current board")
        except Exception:
            # Try after each candidate move.
            for label, uci in candidate_moves.items():
                try:
                    cmv = chess.Move.from_uci(uci)
                    if cmv not in board.legal_moves:
                        continue
                    b = board.copy()
                    b.push(cmv)
                    mv = b.parse_san(san)
                    return set_verdict("true", f"SAN parses as legal move {mv.uci()} after {label}:{uci}")
                except Exception:
                    pass
            return set_verdict("unknown", "SAN-like token did not parse on current/candidate post-move boards")

    if c.claim_type == "recapture_possible_contextual":
        # Heuristic: find a 2-move UCI sequence in text and check legality.
        ucis = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', p["text"])
        if len(ucis) >= 2:
            b, msg = board_after_sequence(board, ucis[:2])
            return set_verdict("true" if b is not None else "false", msg)
        # Special-case Kxf2 after f5f2 if candidate exists.
        if "kxf2" in p["text"].lower():
            for label, uci in candidate_moves.items():
                if uci.endswith("f2"):
                    try:
                        mv = chess.Move.from_uci(uci)
                        if mv not in board.legal_moves:
                            return set_verdict("false", f"{uci} illegal")
                        b = board.copy()
                        b.push(mv)
                        try:
                            reply = b.parse_san("Kxf2")
                            return set_verdict("true", f"Kxf2 legal after {label}:{uci} as {reply.uci()}")
                        except Exception as e:
                            return set_verdict("false", f"Kxf2 not legal after {label}:{uci}: {e}")
                    except Exception:
                        pass
        return set_verdict("unknown", "not enough information to verify recapture")

    if c.claim_type == "material_loss_queen_contextual":
        ucis = p.get("moves_in_text", [])
        # If text contains a concrete 2-move line, verify material delta for side to move.
        if len(ucis) >= 2:
            delta = move_material_delta(board, ucis[:2], board.turn)
            if delta is None:
                return set_verdict("unknown", f"sequence {ucis[:2]} not legal")
            return set_verdict("true" if delta <= -8 else "false", f"material delta for side to move after {ucis[:2]} = {delta}")

        # Try candidate moves ending on an occupied pawn square, then opponent K recaptures.
        for label, uci in candidate_moves.items():
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if mv not in board.legal_moves:
                continue
            b = board.copy()
            captured = board.piece_at(mv.to_square)
            b.push(mv)
            # Try all legal replies that capture the moved queen on destination.
            replies = []
            for reply in list(b.legal_moves):
                if reply.to_square == mv.to_square:
                    replies.append(reply)
            for reply in replies:
                b2 = b.copy()
                b2.push(reply)
                delta = material_score(b2, board.turn) - material_score(board, board.turn)
                if delta <= -8:
                    return set_verdict("true", f"{label}:{uci} can be answered by {reply.uci()}, material delta={delta}")
        return set_verdict("unknown", "could not verify queen loss with simple sequence")

    if c.claim_type == "material_not_lost_contextual":
        ucis = p.get("moves_in_text", [])
        if ucis:
            uci = ucis[0]
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                return set_verdict("unknown", "invalid UCI")
            if mv not in board.legal_moves:
                return set_verdict("false", f"{uci} illegal")
            b = board.copy()
            b.push(mv)
            # If opponent can immediately capture the moved queen, mark false.
            moved_piece = b.piece_at(mv.to_square)
            if moved_piece and moved_piece.piece_type == chess.QUEEN:
                can_capture_queen = any(reply.to_square == mv.to_square for reply in b.legal_moves)
                return set_verdict("true" if not can_capture_queen else "false", f"opponent can immediately capture queen={can_capture_queen}")
        return set_verdict("unknown", "no concrete move to verify")

    if c.claim_type == "candidate_move_exists":
        return set_verdict("true", "candidate move extracted from prompt")

    return set_verdict("unknown", f"no evaluator for claim type {c.claim_type}")


def evaluate_step(step_id: int, text: str, board: chess.Board, candidate_moves: Dict[str, str]) -> StepEvaluation:
    claims = extract_claims_from_step(step_id, text, board, candidate_moves)
    # print(claims)
    evaluated = [eval_atomic_claim(c, board, candidate_moves) for c in claims]

    verifiable = [c for c in evaluated if c.verdict in ("true", "false")]
    true_count = sum(1 for c in verifiable if c.verdict == "true")
    factuality = None if not verifiable else true_count / len(verifiable)

    ev = StepEvaluation(step_id=step_id, text=text, claims=evaluated, factuality=factuality)

    if factuality is None:
        ev.notes.append("no verifiable claims")
    elif factuality >= 0.8:
        ev.is_true_step = True
    elif factuality <= 0.5:
        ev.is_false_step = True
    else:
        ev.is_mixed_step = True

    # Rough support score: claims involving candidate/final move, check, mate, legal, material.
    support_types = {
        "move_legal",
        "move_gives_check",
        "move_is_checkmate",
        "material_loss_queen_contextual",
        "material_not_lost_contextual",
        "recapture_possible_contextual",
    }
    support_claims = [c for c in evaluated if c.claim_type in support_types and c.verdict == "true"]
    ev.support = len(support_claims)

    return ev


# -----------------------------
# Beam-style state scoring
# -----------------------------

def update_reasoning_beam(
    beam: List[ReasoningState],
    step_eval: StepEvaluation,
    beam_size: int = 5,
    alpha_fact: float = 1.0,
    beta_support: float = 0.25,
    lambda_local: float = 0.5,
    lambda_contam: float = 1.0,
) -> List[ReasoningState]:
    """
    Minimal implementation of the accept/reject/contaminate transition.
    """
    new_states: List[ReasoningState] = []

    true_claims = [c for c in step_eval.claims if c.verdict == "true"]
    false_claims = [c for c in step_eval.claims if c.verdict == "false"]

    for st in beam:
        if step_eval.factuality is None:
            # Unknown step: keep state with tiny penalty.
            ns = ReasoningState(
                state_id=st.state_id + f"->U{step_eval.step_id}",
                accepted_claims=list(st.accepted_claims),
                rejected_claims=list(st.rejected_claims),
                contaminated_claims=list(st.contaminated_claims),
                score=st.score - 0.05,
                path=st.path + [f"s{step_eval.step_id}:unknown"],
            )
            new_states.append(ns)

        elif step_eval.is_true_step:
            reward = alpha_fact * step_eval.factuality + beta_support * step_eval.support
            ns = ReasoningState(
                state_id=st.state_id + f"->A{step_eval.step_id}",
                accepted_claims=list(st.accepted_claims) + true_claims,
                rejected_claims=list(st.rejected_claims),
                contaminated_claims=list(st.contaminated_claims),
                score=st.score + reward,
                path=st.path + [f"s{step_eval.step_id}:accept(+{reward:.2f})"],
            )
            new_states.append(ns)

        elif step_eval.is_false_step:
            # Reject branch: treat false step as local error.
            nr = ReasoningState(
                state_id=st.state_id + f"->R{step_eval.step_id}",
                accepted_claims=list(st.accepted_claims) + true_claims,
                rejected_claims=list(st.rejected_claims) + false_claims,
                contaminated_claims=list(st.contaminated_claims),
                score=st.score - lambda_local + beta_support * step_eval.support,
                path=st.path + [f"s{step_eval.step_id}:reject(-{lambda_local:.2f})"],
            )
            new_states.append(nr)

            # Contaminated branch: model may rely on false premise.
            nc = ReasoningState(
                state_id=st.state_id + f"->C{step_eval.step_id}",
                accepted_claims=list(st.accepted_claims) + true_claims,
                rejected_claims=list(st.rejected_claims),
                contaminated_claims=list(st.contaminated_claims) + false_claims,
                score=st.score - lambda_contam + beta_support * step_eval.support,
                path=st.path + [f"s{step_eval.step_id}:contam(-{lambda_contam:.2f})"],
            )
            new_states.append(nc)

        else:
            # Mixed: partial accept with penalty proportional to false part.
            factuality = step_eval.factuality or 0.0
            reward = alpha_fact * factuality + beta_support * step_eval.support
            penalty = lambda_local * (1.0 - factuality)
            ns = ReasoningState(
                state_id=st.state_id + f"->P{step_eval.step_id}",
                accepted_claims=list(st.accepted_claims) + true_claims,
                rejected_claims=list(st.rejected_claims) + false_claims,
                contaminated_claims=list(st.contaminated_claims),
                score=st.score + reward - penalty,
                path=st.path + [f"s{step_eval.step_id}:partial(+{reward:.2f},-{penalty:.2f})"],
            )
            new_states.append(ns)

    new_states.sort(key=lambda x: x.score, reverse=True)
    return new_states[:beam_size]


def final_answer_support(board: chess.Board, answer_uci: Optional[str], accepted_claims: List[AtomicClaim]) -> Dict[str, Any]:
    """
    Minimal support checker for final move.
    For general "which move is better" tasks, there may not be a unique tactical proof.
    This function verifies legality/check/mate and whether accepted claims mention the answer move.
    """
    result = {
        "answer_uci": answer_uci,
        "legal": None,
        "gives_check": None,
        "is_checkmate": None,
        "mentioned_by_verified_claims": False,
        "support_score": 0.0,
        "notes": [],
    }
    if not answer_uci:
        result["notes"].append("No answer UCI extracted")
        return result

    try:
        mv = chess.Move.from_uci(answer_uci)
    except ValueError:
        result["legal"] = False
        result["notes"].append("Answer is invalid UCI")
        return result

    legal = mv in board.legal_moves
    result["legal"] = legal
    if legal:
        result["gives_check"] = board.gives_check(mv)
        b = board.copy()
        b.push(mv)
        result["is_checkmate"] = b.is_checkmate()

    for c in accepted_claims:
        if answer_uci in json.dumps(c.params) or answer_uci in c.raw_text:
            if c.verdict == "true":
                result["mentioned_by_verified_claims"] = True
                break

    # Simple support scoring.
    score = 0.0
    if result["legal"]:
        score += 0.4
    if result["mentioned_by_verified_claims"]:
        score += 0.4
    if result["gives_check"]:
        score += 0.1
    if result["is_checkmate"]:
        score += 0.1
    result["support_score"] = score
    return result


def compute_global_scores(step_evals: List[StepEvaluation], best_state: ReasoningState, answer_support: Dict[str, Any]) -> Dict[str, Any]:
    claims = [c for ev in step_evals for c in ev.claims]
    verifiable = [c for c in claims if c.verdict in ("true", "false")]
    true_count = sum(1 for c in verifiable if c.verdict == "true")
    false_count = sum(1 for c in verifiable if c.verdict == "false")

    factuality = None if not verifiable else true_count / len(verifiable)
    coverage = None if not claims else len(verifiable) / len(claims)

    # Crude non-contamination: fewer contaminated claims in best state is better.
    contam_total = len(best_state.contaminated_claims)
    non_contam = 1.0 / (1.0 + contam_total)

    # Crude recovery: detects explicit correction phrases after false steps.
    false_step_ids = {c.step_id for c in claims if c.verdict == "false"}
    recovered = 0
    # Whole-word matching so correction cues are not spuriously found inside words
    # like "knight" (no), "notation"/"cannot" (not), or "now" (no).
    correction_re = re.compile(
        r"\b(actually|wait|no|nope|doesn't|isn't|not|never|wrong|incorrect|mistake|oops)\b"
    )
    for sid in false_step_ids:
        later_text = " ".join(ev.text.lower() for ev in step_evals if ev.step_id > sid)
        if correction_re.search(later_text):
            recovered += 1
    recovery = 1.0 if not false_step_ids else recovered / len(false_step_ids)

    # State consistency placeholder: contradiction graph is not implemented fully.
    # Here we use the absence of contaminated claims and false accepted claims as proxy.
    state_consistency = non_contam

    support = float(answer_support.get("support_score", 0.0))

    # Weighted proxy.
    weights = {
        "factuality": 0.25,
        "state_consistency": 0.20,
        "answer_support": 0.35,
        "recovery": 0.10,
        "non_contamination": 0.10,
    }

    # If factuality is None, set it to 0 for combined score but report None.
    fact_for_combined = factuality if factuality is not None else 0.0
    faith_proxy = (
        weights["factuality"] * fact_for_combined
        + weights["state_consistency"] * state_consistency
        + weights["answer_support"] * support
        + weights["recovery"] * recovery
        + weights["non_contamination"] * non_contam
    )

    return {
        "num_steps": len(step_evals),
        "num_claims": len(claims),
        "num_verifiable_claims": len(verifiable),
        "num_true_claims": true_count,
        "num_false_claims": false_count,
        "coverage": coverage,
        "factuality": factuality,
        "state_consistency_proxy": state_consistency,
        "answer_support": support,
        "recovery_proxy": recovery,
        "non_contamination_proxy": non_contam,
        "faithfulness_proxy": faith_proxy,
        "weights": weights,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def analyze_record(record: Dict[str, Any], beam_size: int = 5) -> Dict[str, Any]:
    fen = extract_fen(record)
    board = chess.Board(fen)
    candidate_moves = extract_candidate_moves(record)
    answer_uci = extract_answer(record)

    cot = get_cot_text(record)
    steps = split_steps(cot)

    step_evals: List[StepEvaluation] = []
    beam = [ReasoningState(state_id="R0", score=0.0, path=["start"])]

    for i, step in enumerate(steps, start=1):
        ev = evaluate_step(i, step, board, candidate_moves)
        step_evals.append(ev)
        beam = update_reasoning_beam(beam, ev, beam_size=beam_size)

    best_state = beam[0] if beam else ReasoningState(state_id="empty")
    ans_support = final_answer_support(board, answer_uci, best_state.accepted_claims)
    global_scores = compute_global_scores(step_evals, best_state, ans_support)

    return {
        "fen": fen,
        "candidate_moves": candidate_moves,
        "answer_uci": answer_uci,
        "board_turn": "white" if board.turn == chess.WHITE else "black",
        "global_scores": global_scores,
        "answer_support": ans_support,
        "best_state": {
            "state_id": best_state.state_id,
            "score": best_state.score,
            "path": best_state.path,
            "num_accepted": len(best_state.accepted_claims),
            "num_rejected": len(best_state.rejected_claims),
            "num_contaminated": len(best_state.contaminated_claims),
        },
        "steps": [
            {
                "step_id": ev.step_id,
                "text": ev.text,
                "factuality": ev.factuality,
                "support": ev.support,
                "class": (
                    "true" if ev.is_true_step else
                    "false" if ev.is_false_step else
                    "mixed" if ev.is_mixed_step else
                    "unknown"
                ),
                "claims": [asdict(c) for c in ev.claims],
                "notes": ev.notes,
            }
            for ev in step_evals
        ],
    }


def print_summary(result: Dict[str, Any], max_steps: int = 30) -> None:
    print("=" * 80)
    print("Chess CoT Claim Verification Summary")
    print("=" * 80)
    print(f"FEN: {result['fen']}")
    print(f"Turn: {result['board_turn']}")
    print(f"Candidate moves: {result['candidate_moves']}")
    print(f"Answer UCI: {result['answer_uci']}")
    print()

    print("Global scores:")
    for k, v in result["global_scores"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()

    print("Best reasoning state:")
    for k, v in result["best_state"].items():
        print(f"  {k}: {v}")
    print()

    print("Step-level claim evaluations:")
    for step in result["steps"][:max_steps]:
        if not step["claims"]:
            continue
        print("-" * 80)
        print(f"Step {step['step_id']} [{step['class']}], factuality={step['factuality']}, support={step['support']}")
        print(step["text"])
        for c in step["claims"]:
            print(f"  - {c['claim_type']} | {c['subject']} | verdict={c['verdict']} | evidence={c['evidence']}")

    if len(result["steps"]) > max_steps:
        print(f"\n... omitted {len(result['steps']) - max_steps} steps. Use --json-out to inspect full output.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON/JSONL experiment file.")
    parser.add_argument("--json-out", default=None, help="Optional path to save full analysis JSON.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-steps-print", type=int, default=40)
    args = parser.parse_args()

    record = load_record(args.input)
    result = analyze_record(record, beam_size=args.beam_size)

    print_summary(result, max_steps=args.max_steps_print)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved full JSON result to: {args.json_out}")


if __name__ == "__main__":
    main()

