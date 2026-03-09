import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RallyState(Enum):
    WAITING_FOR_SERVE = "waiting_for_serve"
    SERVE_IN_FLIGHT = "serve_in_flight"
    SERVE_BOUNCED = "serve_bounced"
    RALLY_ACTIVE = "rally_active"
    POINT_OVER = "point_over"
    MATCH_OVER = "match_over"


class CourtSide(Enum):
    UPPER = "upper"
    LOWER = "lower"


class Player(Enum):
    UPPER = 1
    LOWER = 2


class FaultType(Enum):
    DOUBLE_BOUNCE = "double_bounce"
    DOUBLE_HIT = "double_hit"
    OUT_OF_BOUNDS = "out_of_bounds"
    NET = "net"
    FAILED_RETURN = "failed_return"
    SERVE_FAULT = "serve_fault"
    DOUBLE_FAULT = "double_fault"


SIDE_FOR_PLAYER = {
    Player.UPPER: CourtSide.UPPER,
    Player.LOWER: CourtSide.LOWER,
}


def _player_for_side(side: Optional[CourtSide]) -> Optional[Player]:
    if side == CourtSide.UPPER:
        return Player.UPPER
    if side == CourtSide.LOWER:
        return Player.LOWER
    return None


@dataclass
class GameScore:
    player_points: Dict[Player, int] = field(
        default_factory=lambda: {Player.UPPER: 0, Player.LOWER: 0}
    )
    player_games: Dict[Player, int] = field(
        default_factory=lambda: {Player.UPPER: 0, Player.LOWER: 0}
    )
    player_sets: Dict[Player, int] = field(
        default_factory=lambda: {Player.UPPER: 0, Player.LOWER: 0}
    )
    is_tiebreak: bool = False

    def point_strings(self) -> Tuple[str, str]:
        up = self.player_points[Player.UPPER]
        lo = self.player_points[Player.LOWER]

        if self.is_tiebreak:
            return str(up), str(lo)

        point_map = {0: "0", 1: "15", 2: "30", 3: "40"}

        if up >= 3 and lo >= 3:
            if up == lo:
                return "DEUCE", "DEUCE"
            return ("ADV", "40") if up > lo else ("40", "ADV")

        return point_map.get(up, "40"), point_map.get(lo, "40")


@dataclass
class RallyContext:
    current_server: Player
    last_hitter: Optional[Player] = None
    last_hit_side: Optional[CourtSide] = None
    last_bounce_side: Optional[CourtSide] = None
    last_bounce_frame: int = -1
    last_hit_frame: int = -1
    bounce_count_this_side: int = 0
    hit_after_bounce: bool = False
    serve_fault_count: int = 0
    hit_count: int = 0
    net_crossed_after_hit: bool = False


@dataclass
class PointRecord:
    frame: int
    winner: Player
    fault_type: Optional[str]
    message: str
    score_snapshot: Dict


class TennisScoringStateMachine:
    NET_Y: Optional[int] = None
    MAX_SERVE_FAULTS = 2
    MAX_BOUNCES_ALLOWED = 1
    FAILED_RETURN_FRAME_GAP = 120

    SETS_TO_WIN = 2
    TIEBREAK_AT = 6
    TIEBREAK_MIN_POINTS = 7
    TIEBREAK_MIN_LEAD = 2

    def __init__(
        self,
        player1_name: str = "Player 1 (Rear)",
        player2_name: str = "Player 2 (Front)",
        *,
        best_of: int = 3,
    ):
        self.state = RallyState.WAITING_FOR_SERVE
        self.score = GameScore()
        self.rally = RallyContext(current_server=Player.UPPER)
        self.player_names = {Player.UPPER: player1_name, Player.LOWER: player2_name}
        self.frame_height = 720
        self._last_message = ""
        self.match_winner: Optional[Player] = None

        self.SETS_TO_WIN = max(1, best_of // 2 + 1)

        self._tiebreak_server_at_start: Optional[Player] = None
        self._tiebreak_points_played = 0

        self.point_history: List[PointRecord] = []

        logger.info(
            "TennisScoringStateMachine initialised "
            f"(best-of-{best_of}, sets_to_win={self.SETS_TO_WIN})"
        )

    def set_net_y(self, net_y: int):
        self.NET_Y = net_y
        logger.info(f"[StateMachine] Net Y set to {net_y}")

    @property
    def is_match_over(self) -> bool:
        return self.state == RallyState.MATCH_OVER

    def get_serve_context(self) -> Dict:
        is_serve = self.state in {
            RallyState.WAITING_FOR_SERVE,
            RallyState.SERVE_IN_FLIGHT,
        }
        server_half = "far" if self.rally.current_server == Player.UPPER else "near"
        total_pts = (
            self.score.player_points[Player.UPPER] + self.score.player_points[Player.LOWER]
        )
        point_side = "deuce" if total_pts % 2 == 0 else "ad"
        return {
            "is_serve": is_serve,
            "server_half": server_half,
            "point_side": point_side,
            "serve_number": self.rally.serve_fault_count + 1,
        }

    def get_score_display(self) -> Dict:
        up_pts, lo_pts = self.score.point_strings()
        serve_num = self.rally.serve_fault_count + 1

        return {
            "upper": {
                "name": self.player_names[Player.UPPER],
                "sets": self.score.player_sets[Player.UPPER],
                "games": self.score.player_games[Player.UPPER],
                "points": up_pts,
            },
            "lower": {
                "name": self.player_names[Player.LOWER],
                "sets": self.score.player_sets[Player.LOWER],
                "games": self.score.player_games[Player.LOWER],
                "points": lo_pts,
            },
            "server": self.player_names[self.rally.current_server],
            "state": self.state.value,
            "serve_number": min(serve_num, 2),
            "last_message": self._last_message,
            "is_tiebreak": self.score.is_tiebreak,
            "match_over": self.is_match_over,
        }

    def check_failed_return(self, current_frame: int) -> Optional[Dict]:
        if self.state not in {RallyState.SERVE_BOUNCED, RallyState.RALLY_ACTIVE}:
            return None
        if self.rally.last_bounce_frame < 0:
            return None

        gap = current_frame - self.rally.last_bounce_frame
        if gap < self.FAILED_RETURN_FRAME_GAP:
            return None

        if self.rally.last_hitter is not None:
            winner = self.rally.last_hitter
        elif self.rally.last_bounce_side is not None:
            failing_player = _player_for_side(self.rally.last_bounce_side)
            winner = self._opponent(failing_player) if failing_player else None
        else:
            winner = None

        if winner is None:
            return None

        loser = self._opponent(winner)
        result = self._result(
            new_state=RallyState.POINT_OVER,
            point_over=True,
            point_winner=winner,
            fault_type=FaultType.FAILED_RETURN,
            message=(
                "Failed return (timeout) — "
                f"{self.player_names[loser]} did not return, "
                f"{self.player_names[winner]} wins point"
            ),
        )
        self.state = result["new_state"]
        self._award_point(
            winner,
            frame=current_frame,
            fault_type_str=FaultType.FAILED_RETURN.value,
            message=result["message"],
        )
        result["score"] = self.get_score_display()
        return result

    def process_event(
        self,
        event_type: str,
        position: Optional[Tuple[int, int]],
        frame: int = 0,
        out_reason: Optional[str] = None,
    ) -> Dict:
        if self.is_match_over:
            logger.debug("[StateMachine] Match is over — ignoring event.")
            return self._noop()

        side = self._get_side(position)
        logger.info(
            f"[StateMachine] {event_type} side={side.value if side else '?'} "
            f"state={self.state.value} frame={frame} out_reason={out_reason}"
        )

        if self.state == RallyState.POINT_OVER:
            logger.warning(
                "[StateMachine] Received event while in POINT_OVER — "
                "auto-resetting to WAITING_FOR_SERVE."
            )
            self.state = RallyState.WAITING_FOR_SERVE

        handler = {
            RallyState.WAITING_FOR_SERVE: self._on_waiting_for_serve,
            RallyState.SERVE_IN_FLIGHT: self._on_serve_in_flight,
            RallyState.SERVE_BOUNCED: self._on_serve_bounced,
            RallyState.RALLY_ACTIVE: self._on_rally_active,
        }.get(self.state)

        if handler is None:
            logger.error(f"[StateMachine] No handler for state {self.state.value}")
            return self._noop()

        result = handler(event_type, side, frame, out_reason)
        self.state = result["new_state"]

        if result["point_over"]:
            self._award_point(
                result["point_winner"],
                frame=frame,
                fault_type_str=result.get("fault_type"),
                message=result.get("message", ""),
            )
            result["score"] = self.get_score_display()

        return result

    def _on_waiting_for_serve(
        self,
        event_type: str,
        side: Optional[CourtSide],
        frame: int,
        out_reason: Optional[str],
    ) -> Dict:
        if event_type == "HIT":
            detected_server = _player_for_side(side) or self.rally.current_server
            self.rally.current_server = detected_server
            self.rally.last_hitter = detected_server
            self.rally.last_hit_side = side
            self.rally.last_hit_frame = frame
            self.rally.hit_after_bounce = False
            self.rally.hit_count = 1
            self.rally.net_crossed_after_hit = False
            serve_num = self.rally.serve_fault_count + 1

            logger.info(
                "[StateMachine] Server detected: "
                f"{self.player_names[detected_server]} "
                f"(side={side.value if side else '?'})"
            )
            return self._result(
                new_state=RallyState.SERVE_IN_FLIGHT,
                message=(
                    f"{self.player_names[detected_server]} serves "
                    f"({'1st' if serve_num == 1 else '2nd'} serve)"
                ),
            )

        return self._noop()

    def _on_serve_in_flight(
        self,
        event_type: str,
        side: Optional[CourtSide],
        frame: int,
        out_reason: Optional[str],
    ) -> Dict:
        if event_type == "SERVE_FAULT":
            return self._handle_serve_fault(frame, out_reason or "serve_fault")
        if event_type == "NET":
            return self._handle_serve_fault(frame, "net")
        if event_type == "OUT":
            return self._handle_serve_fault(frame, out_reason or "out")

        if event_type == "BOUNCE":
            receiver = self._opponent(self.rally.current_server)
            receiver_side = SIDE_FOR_PLAYER[receiver]

            if side is not None and side != receiver_side:
                return self._handle_serve_fault(frame, "wrong_half")

            self.rally.last_bounce_side = side
            self.rally.last_bounce_frame = frame
            self.rally.bounce_count_this_side = 1
            self.rally.net_crossed_after_hit = True
            return self._result(
                new_state=RallyState.SERVE_BOUNCED,
                message="Serve bounced (in)",
            )

        return self._noop()

    def _on_serve_bounced(
        self,
        event_type: str,
        side: Optional[CourtSide],
        frame: int,
        out_reason: Optional[str],
    ) -> Dict:
        server = self.rally.current_server
        receiver = self._opponent(server)

        if event_type == "HIT":
            self.rally.last_hitter = receiver
            self.rally.last_hit_side = side
            self.rally.last_hit_frame = frame
            self.rally.hit_after_bounce = True
            self.rally.bounce_count_this_side = 0
            self.rally.hit_count += 1
            self.rally.net_crossed_after_hit = False
            return self._result(
                new_state=RallyState.RALLY_ACTIVE,
                message=f"{self.player_names[receiver]} returns serve",
            )

        if event_type == "BOUNCE":
            if side is not None and side == self.rally.last_bounce_side:
                return self._result(
                    new_state=RallyState.POINT_OVER,
                    point_over=True,
                    point_winner=server,
                    fault_type=FaultType.FAILED_RETURN,
                    message=(
                        f"Double bounce — {self.player_names[server]} wins point "
                        "(receiver failed to return serve)"
                    ),
                )

            logger.warning(
                "[StateMachine] Bounce switched sides after serve without "
                "an intervening hit — possible detection error."
            )
            self.rally.last_bounce_side = side
            self.rally.last_bounce_frame = frame
            self.rally.bounce_count_this_side = 1
            return self._noop()

        if event_type == "OUT":
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=server,
                fault_type=FaultType.OUT_OF_BOUNDS,
                message=(
                    f"Out ({out_reason or 'out'}) — "
                    f"{self.player_names[server]} wins point"
                ),
            )

        if event_type == "NET":
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=server,
                fault_type=FaultType.NET,
                message=f"Net — {self.player_names[server]} wins point",
            )

        return self._noop()

    def _on_rally_active(
        self,
        event_type: str,
        side: Optional[CourtSide],
        frame: int,
        out_reason: Optional[str],
    ) -> Dict:
        if event_type == "OUT":
            winner = self._opponent(self.rally.last_hitter) if self.rally.last_hitter else None
            winner_name = self.player_names[winner] if winner else "?"
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=winner,
                fault_type=FaultType.OUT_OF_BOUNDS,
                message=f"Out ({out_reason or 'out'}) — {winner_name} wins point",
            )

        if event_type == "NET":
            winner = self._opponent(self.rally.last_hitter) if self.rally.last_hitter else None
            winner_name = self.player_names[winner] if winner else "?"
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=winner,
                fault_type=FaultType.NET,
                message=f"Net — {winner_name} wins point",
            )

        if event_type == "BOUNCE":
            return self._handle_rally_bounce(side, frame)

        if event_type == "HIT":
            return self._handle_rally_hit(side, frame)

        return self._noop()

    def _handle_rally_bounce(self, side: Optional[CourtSide], frame: int) -> Dict:
        same_side = side is not None and side == self.rally.last_bounce_side

        if same_side:
            hit_crossed = (
                self.rally.last_hitter is not None
                and self.rally.last_hit_frame > self.rally.last_bounce_frame
                and self.rally.last_hit_side != self.rally.last_bounce_side
            )

            if not hit_crossed:
                if self.rally.last_hitter is not None:
                    winner = self.rally.last_hitter
                else:
                    winner = _player_for_side(self._opposite_side(side)) or Player.UPPER

                loser = self._opponent(winner)
                return self._result(
                    new_state=RallyState.POINT_OVER,
                    point_over=True,
                    point_winner=winner,
                    fault_type=FaultType.DOUBLE_BOUNCE,
                    message=(
                        f"Double bounce — {self.player_names[loser]} failed to return, "
                        f"{self.player_names[winner]} wins point"
                    ),
                )

        self.rally.last_bounce_side = side
        self.rally.last_bounce_frame = frame
        self.rally.bounce_count_this_side = (
            self.rally.bounce_count_this_side + 1 if same_side else 1
        )

        return self._result(
            new_state=RallyState.RALLY_ACTIVE,
            message="Ball bounced",
        )

    def _handle_rally_hit(self, side: Optional[CourtSide], frame: int) -> Dict:
        expected_hitter = (
            self._opponent(self.rally.last_hitter) if self.rally.last_hitter else None
        )

        if (
            self.rally.last_hitter is not None
            and side is not None
            and side == self.rally.last_hit_side
            and self.rally.last_bounce_side != side
        ):
            winner = self._opponent(self.rally.last_hitter)
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=winner,
                fault_type=FaultType.DOUBLE_HIT,
                message=f"Double hit — {self.player_names[winner]} wins point",
            )

        hitter = expected_hitter or (
            _player_for_side(side)
            or self._opponent(self.rally.last_hitter or self.rally.current_server)
        )

        self.rally.last_hitter = hitter
        self.rally.last_hit_side = side
        self.rally.last_hit_frame = frame
        self.rally.hit_after_bounce = (
            self.rally.last_bounce_side == side and self.rally.last_bounce_frame < frame
        )
        self.rally.bounce_count_this_side = 0
        self.rally.hit_count += 1
        self.rally.net_crossed_after_hit = False

        return self._result(
            new_state=RallyState.RALLY_ACTIVE,
            message=f"{self.player_names[hitter]} hits",
        )

    def _award_point(
        self,
        winner: Optional[Player],
        *,
        frame: int = 0,
        fault_type_str: Optional[str] = None,
        message: str = "",
    ):
        if winner is None:
            logger.warning("[StateMachine] _award_point called with no winner.")
            return

        logger.info(f"[StateMachine] Point → {self.player_names[winner]}")
        self.point_history.append(
            PointRecord(
                frame=frame,
                winner=winner,
                fault_type=fault_type_str,
                message=message,
                score_snapshot=self.get_score_display(),
            )
        )

        if self.score.is_tiebreak:
            self._award_tiebreak_point(winner)
            return

        pts = self.score.player_points
        loser = self._opponent(winner)
        w, l = pts[winner], pts[loser]

        if w < 3:
            pts[winner] += 1
        elif l < 3:
            self._award_game(winner)
            return
        elif w == 3 and l == 3:
            pts[winner] = 4
        elif w == 4:
            self._award_game(winner)
            return
        elif l == 4:
            pts[winner] = 3
            pts[loser] = 3
        else:
            logger.warning(
                f"[StateMachine] Unexpected point state w={w} l={l}, "
                f"awarding game to {self.player_names[winner]}"
            )
            self._award_game(winner)
            return

        self._reset_rally()

    def _award_tiebreak_point(self, winner: Player):
        self.score.player_points[winner] += 1
        self._tiebreak_points_played += 1

        w = self.score.player_points[winner]
        l = self.score.player_points[self._opponent(winner)]

        if w >= self.TIEBREAK_MIN_POINTS and (w - l) >= self.TIEBREAK_MIN_LEAD:
            self._award_game(winner)
            return

        if self._tiebreak_points_played == 1:
            self.rally.current_server = self._opponent(self.rally.current_server)
        elif self._tiebreak_points_played > 1 and (self._tiebreak_points_played - 1) % 2 == 0:
            self.rally.current_server = self._opponent(self.rally.current_server)

        self._reset_rally()

    def _award_game(self, winner: Player):
        logger.info(f"[StateMachine] Game → {self.player_names[winner]}")

        was_tiebreak = self.score.is_tiebreak
        self.score.player_games[winner] += 1
        self.score.player_points = {Player.UPPER: 0, Player.LOWER: 0}
        self.score.is_tiebreak = False

        wg = self.score.player_games[winner]
        lg = self.score.player_games[self._opponent(winner)]

        if was_tiebreak:
            self._award_set(winner, after_tiebreak=True)
            return

        if wg >= 6 and wg - lg >= 2:
            self._award_set(winner)
            return

        if wg == self.TIEBREAK_AT and lg == self.TIEBREAK_AT:
            self._start_tiebreak()
            return

        self.rally.current_server = self._opponent(self.rally.current_server)
        self._reset_rally()

    def _start_tiebreak(self):
        logger.info("[StateMachine] Tiebreak started")
        self.score.is_tiebreak = True
        self.score.player_points = {Player.UPPER: 0, Player.LOWER: 0}
        self._tiebreak_server_at_start = self.rally.current_server
        self._tiebreak_points_played = 0
        self._reset_rally()

    def _award_set(self, winner: Player, *, after_tiebreak: bool = False):
        logger.info(f"[StateMachine] Set → {self.player_names[winner]}")
        self.score.player_sets[winner] += 1
        self.score.player_games = {Player.UPPER: 0, Player.LOWER: 0}
        self.score.is_tiebreak = False

        if self.score.player_sets[winner] >= self.SETS_TO_WIN:
            self._end_match(winner)
            return

        if after_tiebreak and self._tiebreak_server_at_start is not None:
            self.rally.current_server = self._opponent(self._tiebreak_server_at_start)
        else:
            self.rally.current_server = self._opponent(self.rally.current_server)

        self._reset_rally()

    def _end_match(self, winner: Player):
        self.match_winner = winner
        self.state = RallyState.MATCH_OVER
        self._last_message = f"Match over — {self.player_names[winner]} wins!"
        logger.info(
            f"[StateMachine] Match won by {self.player_names[winner]} "
            f"({self.score.player_sets[Player.UPPER]}-"
            f"{self.score.player_sets[Player.LOWER]} sets)"
        )

    def _reset_rally(self):
        server = self.rally.current_server
        self.rally = RallyContext(current_server=server)
        self.state = RallyState.WAITING_FOR_SERVE

    def _handle_serve_fault(self, frame: int, reason: str) -> Dict:
        self.rally.serve_fault_count += 1
        server = self.rally.current_server
        server_name = self.player_names[server]

        if self.rally.serve_fault_count >= self.MAX_SERVE_FAULTS:
            receiver = self._opponent(server)
            return self._result(
                new_state=RallyState.POINT_OVER,
                point_over=True,
                point_winner=receiver,
                fault_type=FaultType.DOUBLE_FAULT,
                message=(
                    f"Double fault ({reason}) — "
                    f"{self.player_names[receiver]} wins point"
                ),
            )

        fault_count = self.rally.serve_fault_count
        self.rally = RallyContext(current_server=server)
        self.rally.serve_fault_count = fault_count
        return self._result(
            new_state=RallyState.WAITING_FOR_SERVE,
            message=f"Fault ({reason}) — {server_name} 2nd serve",
        )

    def _get_side(self, position: Optional[Tuple[int, int]]) -> Optional[CourtSide]:
        if position is None:
            return None
        net_y = self.NET_Y if self.NET_Y is not None else self.frame_height // 2
        return CourtSide.UPPER if position[1] < net_y else CourtSide.LOWER

    @staticmethod
    def _opposite_side(side: Optional[CourtSide]) -> Optional[CourtSide]:
        if side == CourtSide.UPPER:
            return CourtSide.LOWER
        if side == CourtSide.LOWER:
            return CourtSide.UPPER
        return None

    @staticmethod
    def _opponent(player: Optional[Player]) -> Player:
        return Player.LOWER if player == Player.UPPER else Player.UPPER

    def _result(
        self,
        new_state: RallyState,
        message: str = "",
        point_over: bool = False,
        point_winner: Optional[Player] = None,
        fault_type: Optional[FaultType] = None,
    ) -> Dict:
        if message:
            self._last_message = message
        return {
            "valid": True,
            "point_over": point_over,
            "point_winner": point_winner,
            "fault_type": fault_type.value if fault_type else None,
            "new_state": new_state,
            "message": message,
        }

    def _noop(self) -> Dict:
        return {
            "valid": True,
            "point_over": False,
            "point_winner": None,
            "fault_type": None,
            "new_state": self.state,
            "message": "",
        }

    def manual_award_point(self, player: Player):
        logger.info(f"[StateMachine] Manual point → {self.player_names[player]}")
        self._award_point(player, message="Manual override")

    def undo_last_point(self) -> bool:
        if not self.point_history:
            logger.warning("[StateMachine] No point history to undo.")
            return False

        record = self.point_history.pop()
        snap = record.score_snapshot

        self.score.player_sets[Player.UPPER] = snap["upper"]["sets"]
        self.score.player_sets[Player.LOWER] = snap["lower"]["sets"]
        self.score.player_games[Player.UPPER] = snap["upper"]["games"]
        self.score.player_games[Player.LOWER] = snap["lower"]["games"]

        self.score.is_tiebreak = snap.get("is_tiebreak", False)
        for player, key in ((Player.UPPER, "upper"), (Player.LOWER, "lower")):
            self.score.player_points[player] = self._display_to_points(
                snap[key]["points"],
                self.score.is_tiebreak,
            )

        self.match_winner = None
        self.state = RallyState.WAITING_FOR_SERVE
        self._reset_rally()
        self._last_message = f"Undo: reverted point (was {record.message})"
        logger.info(f"[StateMachine] Undo: reverted point at frame {record.frame}")
        return True

    @staticmethod
    def _display_to_points(display: str, is_tiebreak: bool) -> int:
        if is_tiebreak:
            try:
                return int(display)
            except ValueError:
                return 0
        return {"0": 0, "15": 1, "30": 2, "40": 3, "ADV": 4, "DEUCE": 3}.get(display, 0)

    def reset_score(self):
        self.score = GameScore()
        self.rally = RallyContext(current_server=Player.UPPER)
        self.state = RallyState.WAITING_FOR_SERVE
        self._last_message = ""
        self.match_winner = None
        self.point_history.clear()
        self._tiebreak_server_at_start = None
        self._tiebreak_points_played = 0
        logger.info("[StateMachine] Score reset.")