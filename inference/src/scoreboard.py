import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from inference.src.tennis_state_machine import (
    TennisScoringStateMachine,
    Player,
    RallyState,
)

logger = logging.getLogger(__name__)

Color = Tuple[int, int, int]

BLACK: Color = (0, 0, 0)
WHITE: Color = (220, 220, 220)
DARK_GREY: Color = (20, 20, 20)
MID_GREY: Color = (40, 40, 40)
DIM_GREY: Color = (90, 90, 90)
GREEN: Color = (90, 160, 100)
GOLD: Color = (90, 160, 190)
RED: Color = (70, 70, 170)
CYAN: Color = (140, 120, 60)
ORANGE: Color = (50, 90, 160)

FONT = cv2.FONT_HERSHEY_SIMPLEX

_STATE_COLOURS: Dict[str, Tuple[Color, Color]] = {
        "fault": ((35, 25, 80), (65, 50, 130)),
        "double": ((25, 15, 65), (50, 35, 100)),
        "ace": ((20, 65, 30), (40, 110, 55)),
        "winner": ((20, 65, 30), (40, 110, 55)),
        "rally": ((20, 60, 80), (35, 110, 130)),
        "play": ((20, 60, 80), (35, 110, 130)),
        "let": ((20, 55, 90), (35, 95, 150)),
        "tiebreak": ((20, 50, 70), (35, 90, 120)),
        "match_over": ((20, 60, 25), (45, 120, 50)),
}
_DEFAULT_STATE_COLOUR: Tuple[Color, Color] = ((25, 25, 25), (60, 60, 60))

_EVENT_DOT_COLOURS: Dict[str, Color] = {
        "fault": RED,
        "ace": GREEN,
        "winner": (130, 170, 60),
        "bounce": GOLD,
        "hit": GOLD,
        "serve": CYAN,
        "net": ORANGE,
        "let": ORANGE,
        "out": RED,
        "point": GREEN,
}

_NAME_COL_W = 210
_SCORE_COL_W = 54
_PAD_LEFT = 14
_ROW_H = 46
_HEADER_H = 28
_MARGIN_LEFT = 22
_MARGIN_TOP = 14
_FEED_W = 290
_FEED_ROW_H = 38
_MAX_FEED_ROWS = 5
_RADIO_R = 7
_RADIO_FILL_R = 4


def _rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        colour: Color,
        opacity: float = 0.92,
) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def _border(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        colour: Color,
        thickness: int = 1,
) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)


def _line(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        colour: Color = (50, 50, 50),
) -> None:
        cv2.line(frame, (x1, y1), (x2, y2), colour, 1)


def _text(
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        scale: float,
        colour: Color,
        thickness: int = 1,
) -> None:
        cv2.putText(frame, label, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def _text_size(label: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
        (w, h), _ = cv2.getTextSize(label, FONT, scale, thickness)
        return w, h


def _dim(colour: Color, factor: float = 0.38) -> Color:
        return tuple(int(c * factor) for c in colour)  # type: ignore[return-value]


def _circle(
        frame: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        colour: Color,
        filled: bool = True,
        thickness: int = 1,
) -> None:
        fill = -1 if filled else thickness
        cv2.circle(frame, (cx, cy), radius, colour, fill, cv2.LINE_AA)


def _state_colours(state: str) -> Tuple[Color, Color]:
        low = state.lower()
        for keyword, colours in _STATE_COLOURS.items():
                if keyword in low:
                        return colours
        return _DEFAULT_STATE_COLOUR


def _event_dot_colour(label: str) -> Color:
        low = label.lower()
        for keyword, colour in _EVENT_DOT_COLOURS.items():
                if keyword in low:
                        return colour
        return DIM_GREY


def _names_match(player: str, server: str) -> bool:
        if not server:
                return False
        a, b = player.strip().lower(), server.strip().lower()
        return a == b or b.startswith(a) or a.startswith(b)


class Scoreboard:
        _EVENT_LOG_CAP = 500

        def __init__(
                self,
                frame_width: int = 1280,
                frame_height: int = 720,
                enable_auto_scoring: bool = True,
                rally_timeout_frames: int = 90,
        ) -> None:
                self.frame_width = frame_width
                self.frame_height = frame_height
                self.enable_auto_scoring = enable_auto_scoring
                self.rally_timeout_frames = rally_timeout_frames
                self.event_log: List[Dict] = []
                self.court_bounds: Optional[np.ndarray] = None
                self.frame_count: int = 0

                self.state_machine = TennisScoringStateMachine()
                self.state_machine.frame_height = frame_height

        def set_court_bounds(self, keypoints: np.ndarray) -> None:
                """Set court boundary keypoints and configure net Y for the state machine."""
                self.court_bounds = keypoints
                if keypoints is not None and len(keypoints) >= 16:
                        net_y = int((keypoints[14][1] + keypoints[15][1]) / 2)
                        self.state_machine.set_net_y(net_y)

        def update(
                self,
                ball_position: Optional[Tuple[int, int]] = None,
                player_positions: Optional[Dict] = None,
                events: Optional[List[Dict]] = None,
        ) -> None:
                """Feed detected events into the state machine and update score."""
                self.frame_count += 1

                if not self.enable_auto_scoring:
                        return

                if events:
                        for event in events:
                                event_type = event.get("type", "")
                                position = event.get("position")
                                frame = event.get("frame", self.frame_count)
                                out_reason = event.get("out_reason")

                                result = self.state_machine.process_event(
                                        event_type=event_type,
                                        position=position,
                                        frame=frame,
                                        out_reason=out_reason,
                                )
                                msg = result.get("message", "")
                                if msg:
                                        self.log_event(msg, frame)

                # Check for failed return (timeout)
                timeout_result = self.state_machine.check_failed_return(self.frame_count)
                if timeout_result and timeout_result.get("message"):
                        self.log_event(timeout_result["message"], self.frame_count)

        def draw_scoreboard(
                self,
                frame: np.ndarray,
                player_positions: Optional[Dict] = None,
        ) -> np.ndarray:
                """Draw the scoreboard overlay using current state machine score."""
                score = self.state_machine.get_score_display()
                return self.draw(frame, score)

        def manual_award_point(self, player: Player) -> None:
                """Manually award a point to a player."""
                self.state_machine.manual_award_point(player)
                name = self.state_machine.player_names[player]
                self.log_event(f"Manual point → {name}", self.frame_count)

        def reset_score(self) -> None:
                """Reset all scores to zero."""
                self.state_machine.reset_score()
                self.event_log.clear()
                self.log_event("Score reset", self.frame_count)

        @staticmethod
        def default_score() -> Dict:
                return {
                        "upper": {"name": "Player 1", "sets": 0, "games": 0, "points": "0"},
                        "lower": {"name": "Player 2", "sets": 0, "games": 0, "points": "0"},
                        "server": "Player 1",
                        "state": "waiting_for_serve",
                        "last_message": "",
                }

        def log_event(self, event_name: str, frame_number: int = 0) -> None:
                self.event_log.append({"event": event_name, "frame": frame_number})
                if len(self.event_log) > self._EVENT_LOG_CAP:
                        self.event_log = self.event_log[-self._EVENT_LOG_CAP :]

        def draw(self, frame: np.ndarray, score: Dict) -> np.ndarray:
                out = frame.copy()
                self._draw_score_panel(out, score)
                self._draw_state_badge(out, score)
                self._draw_event_feed(out)
                return out

        def draw_debug_overlay(
                self, frame: np.ndarray, frame_count: int, feature_flags: Dict
        ) -> np.ndarray:
                panel_h = _HEADER_H + _ROW_H * 2
                debug_top = _MARGIN_TOP + panel_h + 10

                entries: List[Tuple[str, Color]] = [(f"Frame: {frame_count}", WHITE)]
                for name, enabled in feature_flags.items():
                        colour = (55, 140, 65) if enabled else DIM_GREY
                        entries.append((f"{name}: {'ON' if enabled else 'OFF'}", colour))

                panel_bottom = debug_top + 22 * len(entries) + 10
                _rect(frame, 10, debug_top, 310, panel_bottom, BLACK, opacity=0.60)

                y = debug_top + 22
                for label, colour in entries:
                        _text(frame, label, 20, y, 0.45, colour)
                        y += 22

                return frame

        def _draw_score_panel(self, frame: np.ndarray, score: Dict) -> None:
                panel_w = _NAME_COL_W + _SCORE_COL_W * 3
                panel_h = _HEADER_H + _ROW_H * 2
                px = _MARGIN_LEFT
                py = _MARGIN_TOP
                server = score.get("server", "")

                _rect(frame, px, py, px + panel_w, py + panel_h, DARK_GREY, opacity=0.94)
                _border(frame, px, py, px + panel_w, py + panel_h, (55, 55, 55))

                is_tiebreak = score.get("is_tiebreak", False)
                self._draw_score_headers(frame, px, py, panel_w, is_tiebreak)

                for col in range(3):
                        col_x = px + _NAME_COL_W + _SCORE_COL_W * col
                        _line(frame, col_x, py, col_x, py + panel_h)

                for row, (side, fallback) in enumerate((("upper", "Player 1"), ("lower", "Player 2"))):
                        data = score.get(side, {})
                        self._draw_player_row(frame, px, py, panel_w, row, data, fallback, server)

        @staticmethod
        def _draw_score_headers(
                frame: np.ndarray, px: int, py: int, panel_w: int, is_tiebreak: bool = False
        ) -> None:
                hb = py + _HEADER_H
                _rect(frame, px, py, px + panel_w, hb, (12, 12, 12), opacity=0.97)
                _line(frame, px, hb, px + panel_w, hb)

                pt_label = "TB" if is_tiebreak else "PT"
                for i, label in enumerate(("SET", "GM", pt_label)):
                        cx = px + _NAME_COL_W + _SCORE_COL_W * i + _SCORE_COL_W // 2
                        lw, _ = _text_size(label, 0.32)
                        col = CYAN if label == "TB" else DIM_GREY
                        _text(frame, label, cx - lw // 2, py + 20, 0.32, col)

        @staticmethod
        def _draw_player_row(
                frame: np.ndarray,
                px: int,
                py: int,
                panel_w: int,
                row: int,
                data: Dict,
                fallback: str,
                server: str,
        ) -> None:
                ry = py + _HEADER_H + row * _ROW_H
                cy = ry + _ROW_H // 2
                ty = cy + 6
                name = data.get("name", fallback)[:20]
                serving = _names_match(name, server)

                if row == 1:
                        _rect(frame, px, ry, px + panel_w, ry + _ROW_H, MID_GREY, opacity=0.25)
                _line(frame, px, ry, px + panel_w, ry)

                rx = px + _PAD_LEFT + _RADIO_R
                ring_col = GREEN if serving else DIM_GREY
                _circle(frame, rx, cy, _RADIO_R, ring_col, filled=False)
                if serving:
                        _circle(frame, rx, cy, _RADIO_FILL_R, GREEN)

                nx = px + _PAD_LEFT + _RADIO_R * 2 + 8
                scale = 0.40 if serving else 0.37
                colour = WHITE if serving else DIM_GREY
                _text(frame, name, nx, ty, scale, colour, 1)

                values = [
                        (str(data.get("sets", 0)), GOLD),
                        (str(data.get("games", 0)), WHITE),
                        (str(data.get("points", "0")), WHITE),
                ]
                for i, (val, base_col) in enumerate(values):
                        cx = px + _NAME_COL_W + _SCORE_COL_W * i + _SCORE_COL_W // 2
                        col = base_col if serving else _dim(base_col)
                        vw, _ = _text_size(val, 0.48, 1)
                        _text(frame, val, cx - vw // 2, ty, 0.48, col, 1)

        @staticmethod
        def _draw_state_badge(frame: np.ndarray, score: Dict) -> None:
                raw = score.get("state", "")
                if not raw:
                        return

                label = raw.replace("_", " ").upper()[:32]
                bg, border = _state_colours(raw)

                lw, lh = _text_size(label, 0.45, 1)
                bx = (frame.shape[1] - lw) // 2 - 16
                by = _MARGIN_TOP
                bw = lw + 32
                bh = lh + 18

                _rect(frame, bx, by, bx + bw, by + bh, bg, opacity=0.88)
                _border(frame, bx, by, bx + bw, by + bh, border)
                _text(frame, label, bx + 16, by + lh + 10, 0.45, WHITE, 1)

                msg = score.get("last_message", "")[:55]
                if msg:
                        mw, _ = _text_size(msg, 0.36)
                        _text(frame, msg, (frame.shape[1] - mw) // 2, by + bh + 18, 0.36, DIM_GREY)

        def _draw_event_feed(self, frame: np.ndarray) -> None:
                visible = min(len(self.event_log), _MAX_FEED_ROWS)
                total_rows = _MAX_FEED_ROWS

                panel_h = _HEADER_H + _FEED_ROW_H * total_rows
                px = frame.shape[1] - _FEED_W - _MARGIN_LEFT
                py = _MARGIN_TOP

                _rect(frame, px, py, px + _FEED_W, py + panel_h, DARK_GREY, opacity=0.94)
                _border(frame, px, py, px + _FEED_W, py + panel_h, (55, 55, 55))

                hb = py + _HEADER_H
                _rect(frame, px, py, px + _FEED_W, hb, (12, 12, 12), opacity=0.97)
                _line(frame, px, hb, px + _FEED_W, hb)

                _circle(frame, px + _PAD_LEFT, py + _HEADER_H // 2, 4, GREEN)
                _text(frame, "DETECTED EVENTS", px + _PAD_LEFT + 14, py + 20, 0.32, GREEN)

                fl = "FRAME"
                flw, _ = _text_size(fl, 0.32)
                _text(frame, fl, px + _FEED_W - flw - _PAD_LEFT, py + 20, 0.32, DIM_GREY)

                frame_col_x = px + _FEED_W - 70
                _line(frame, frame_col_x, py, frame_col_x, py + panel_h)

                recent = list(reversed(self.event_log[-visible:])) if self.event_log else []

                for row in range(total_rows):
                        ry = hb + row * _FEED_ROW_H
                        cy = ry + _FEED_ROW_H // 2
                        ty = cy + 5

                        if row % 2 == 1:
                                _rect(frame, px, ry, px + _FEED_W, ry + _FEED_ROW_H, MID_GREY, opacity=0.25)
                        if row > 0:
                                _line(frame, px, ry, px + _FEED_W, ry)

                        dot_x = px + _PAD_LEFT + 4

                        if row < len(recent):
                                ev = recent[row]
                                label = ev.get("event", "")[:28]
                                fnum = ev.get("frame", 0)
                                dot_col = _event_dot_colour(label)

                                fade = max(0.30, 1.0 - row * 0.14)
                                text_col = _dim(WHITE, fade)

                                _circle(frame, dot_x, cy, 5, dot_col)
                                _circle(frame, dot_x, cy, 5, _dim(dot_col, 0.6), filled=False)
                                _text(frame, label.upper(), px + _PAD_LEFT + 18, ty, 0.32, text_col)

                                if fnum > 0:
                                        fn = f"#{fnum}"
                                        fnw, _ = _text_size(fn, 0.30)
                                        _text(
                                                frame,
                                                fn,
                                                px + _FEED_W - fnw - _PAD_LEFT,
                                                ty,
                                                0.30,
                                                _dim(DIM_GREY, fade),
                                        )
                        else:
                                _circle(frame, dot_x, cy, 3, (35, 35, 35))
                                _text(frame, "—", px + _PAD_LEFT + 18, ty, 0.34, (35, 35, 35))