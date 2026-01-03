import logging
import cv2
import numpy as np
from typing import Optional, Dict, Tuple, List
from enum import Enum
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameState(Enum):
    WAITING_FOR_SERVE = "waiting_for_serve"
    IN_PLAY = "in_play"
    POINT_OVER = "point_over"

class Player(Enum):
    UPPER = 1  # Rear player (top of screen)
    LOWER = 2  # Front player (bottom of screen)


class Scoreboard:
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720,
                 enable_auto_scoring: bool = True,
                 rally_timeout_frames: int = 50):
        """
        Initialize scoreboard
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            enable_auto_scoring: Enable automatic scoring based on events
            rally_timeout_frames: Frames to wait before considering rally over
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.enable_auto_scoring = enable_auto_scoring
        self.rally_timeout_frames = rally_timeout_frames
        
        # Score state
        self.points = {Player.UPPER: 0, Player.LOWER: 0}
        self.games = {Player.UPPER: 0, Player.LOWER: 0}
        self.sets = {Player.UPPER: 0, Player.LOWER: 0}
        
        # Game state
        self.game_state = GameState.WAITING_FOR_SERVE
        self.server = Player.UPPER
        self.last_shot_player: Optional[Player] = None
        
        # Rally tracking
        self.rally_sequence: List[Dict] = []
        self.frames_since_last_event = 0
        self.last_ball_position: Optional[Tuple[int, int]] = None
        self.ball_history = deque(maxlen=30)
        
        # Event validation - position-based
        self.last_bounce_side: Optional[str] = None  # 'upper' or 'lower'
        self.last_hit_side: Optional[str] = None
        
        self.court_bounds: Optional[Dict[str, int]] = None
        self.mid_court_y: Optional[int] = None  # Net Y position
        
        # Court key points
        # These 6 points define: 4 corners + 2 net posts
        self.court_points: List[Tuple[int, int]] = []
        self.net_left: Optional[Tuple[int, int]] = None
        self.net_right: Optional[Tuple[int, int]] = None
        
        self.frame_count = 0
        self.total_rallies = 0
        self.total_points = 0
        logger.info("Scoreboard initialized")
    
    def set_court_bounds(self, court_lines: Optional[np.ndarray] = None):
        if court_lines is not None:
            # Assumed court_lines is shape (N, 2) with x,y coordinates
            min_x = np.min(court_lines[:, 0])
            max_x = np.max(court_lines[:, 0])
            min_y = np.min(court_lines[:, 1])
            max_y = np.max(court_lines[:, 1])
            
            self.court_bounds = {
                'min_x': int(min_x),
                'max_x': int(max_x),
                'min_y': int(min_y),
                'max_y': int(max_y)
            }
            # Set net posts (left and right edges at net line)
            # self.net_left = (int(min_x), self.mid_court_y)
            # self.net_right = (int(max_x), self.mid_court_y)
            # self.mid_court_y = self.net_left[1]

            
            # Define 6 court points: 4 corners + 2 net posts
            # self.court_points = [
            #     (int(min_x), int(min_y)),  # Top-left corner
            #     (int(max_x), int(min_y)),  # Top-right corner
            #     (int(min_x), int(max_y)),  # Bottom-left corner
            #     (int(max_x), int(max_y)),  # Bottom-right corner
            #     self.net_left,              # Left net post
            #     self.net_right              # Right net post
            # ]

            logger.info(f"Court bounds set: {self.court_bounds}")
            logger.info(f"Net position (Y): {self.mid_court_y}")
            logger.info(f"Net posts: Left={self.net_left}, Right={self.net_right}")
        
        self.net_left = (404,290)
        self.net_right = (865,290)
        self.mid_court_y = self.net_left[1]
        self.court_points = [
            (476,238),                  # Top-left corner
            (785,238),                  # Top-right corner
            (165,461),                  # Bottom-left corner
            (1124,464),                 # Bottom-right corner
            self.net_left,              # Left net post
            self.net_right              # Right net post
        ]
            
    
    def identify_player_by_position(self, y_position: int) -> Player:
        if self.mid_court_y is None: 
            threshold = self.frame_height // 2
        else:
            threshold = self.mid_court_y
        
        return Player.UPPER if y_position < threshold else Player.LOWER
    
    def get_court_side(self, y_position: int) -> str:
        if self.mid_court_y is None:
            threshold = self.frame_height // 2
        else:
            threshold = self.mid_court_y
        
        return 'upper' if y_position < threshold else 'lower'
    
    def is_ball_near_net(self, position: Tuple[int, int], tolerance: int = 30) -> bool:
        if self.mid_court_y is None:
            return False
        
        _, y = position
        return abs(y - self.mid_court_y) < tolerance
    
    def update(self, 
               ball_position: Optional[Tuple[int, int]] = None,
               player_positions: Optional[Dict[int, Tuple[int, int]]] = None,
               events: Optional[List[Dict]] = None) -> bool:
        self.frame_count += 1
        point_scored = False
        
        # Update ball history
        if ball_position is not None:
            self.ball_history.append(ball_position)
            self.last_ball_position = ball_position
        else:
            self.ball_history.append(None)
        
        self.frames_since_last_event += 1
        
        if events and self.enable_auto_scoring:
            for event in events:
                event_type = event.get("type")
                event_pos = event.get("position")
                event_frame = event.get("frame", self.frame_count)
                
                if event_pos is None:
                    continue
                
                # Get which side of net this event occurred on
                event_side = self.get_court_side(event_pos[1])
                event_player = self.identify_player_by_position(event_pos[1])

                is_valid = True
                
                if event_type == "HIT":
                    # Duplicate hit: same side of net as last hit
                    if self.last_hit_side == event_side and len(self.rally_sequence) > 0:
                        # Check if last event was also a HIT on same side
                        last_hit_events = [e for e in self.rally_sequence if e['type'] == 'HIT']
                        if last_hit_events and last_hit_events[-1].get('side') == event_side:
                            is_valid = False  # Duplicate hit on same side
                            logger.debug(f"Skipping duplicate HIT on {event_side} side")
                    
                    if is_valid:
                        self.last_hit_side = event_side
                    
                elif event_type == "BOUNCE":
                    # Duplicate bounce: same side of net as last bounce
                    if self.last_bounce_side == event_side and len(self.rally_sequence) > 0:
                        # Check if last bounce was on same side
                        last_bounce_events = [e for e in self.rally_sequence if e['type'] == 'BOUNCE']
                        last_hit_events = [e for e in self.rally_sequence if e['type'] == 'HIT']

                        # if bounce then hit on opposite side - valid
                        if last_bounce_events and last_bounce_events[-1].get('side') == event_side:
                            # Check if there was a HIT on opposite side after last bounce
                            if last_hit_events and last_hit_events[-1].get('side') != event_side and last_hit_events[-1].get('frame') > last_bounce_events[-1].get('frame'):
                                is_valid = True
                            # This is a DOUBLE BOUNCE on same side - point is over!
                            else:
                                logger.info(f"Double bounce detected on {event_side} side!")
                                # Opponent of the side wins
                                winner = Player.UPPER if event_side == 'lower' else Player.LOWER
                                self._award_point(winner)
                                self._reset_rally()
                                self.game_state = GameState.WAITING_FOR_SERVE
                                return True
                    
                    if is_valid:
                        self.last_bounce_side = event_side
                
                if is_valid:
                    rally_event = {
                        "type": event_type,
                        "position": event_pos,
                        "frame": event_frame,
                        "player": event_player,
                        "side": event_side
                    }
                    self.rally_sequence.append(rally_event)
                    self.frames_since_last_event = 0
                    
                    # TODO: Check if ball hit the net
                    if self.is_ball_near_net(event_pos):
                        logger.info(f"Ball near net at frame {event_frame}")
                        # TODO: Implement net hit detection logic
                    
                    if event_type == "HIT":
                        self.last_shot_player = event_player
                        self.game_state = GameState.IN_PLAY

        if (self.game_state == GameState.IN_PLAY and 
            self.frames_since_last_event > self.rally_timeout_frames):
            point_scored = self._evaluate_rally()
            self._reset_rally()
            self.game_state = GameState.WAITING_FOR_SERVE
        
        # Check for out of bounds
        if (self.enable_auto_scoring and 
            ball_position is not None and 
            self.court_bounds is not None):
            if self._is_ball_out_of_bounds(ball_position):
                # Ball went out - opponent of last shot wins point
                if self.last_shot_player is not None:
                    winner = Player.LOWER if self.last_shot_player == Player.UPPER else Player.UPPER
                    self._award_point(winner)
                    point_scored = True
                    logger.info(f"Ball out of bounds! Point to {winner.name}")
                    self._reset_rally()
        
        return point_scored
    
    def _is_ball_out_of_bounds(self, position: Tuple[int, int]) -> bool:
        """Check if ball position is outside court boundaries"""
        if self.court_bounds is None:
            return False
        
        x, y = position
        margin = 20  # Pixel margin for error
        
        return (x < self.court_bounds['min_x'] - margin or
                x > self.court_bounds['max_x'] + margin or
                y < self.court_bounds['min_y'] - margin or
                y > self.court_bounds['max_y'] + margin)
    
    def _evaluate_rally(self) -> bool:
        """
        Evaluate rally sequence to determine point winner
        Uses position-based logic: double bounce on SAME SIDE of net
        
        Returns:
            True if point was awarded
        """
        if len(self.rally_sequence) == 0:
            return False

        bounces = [e for e in self.rally_sequence if e['type'] == 'BOUNCE']
        hits = [e for e in self.rally_sequence if e['type'] == 'HIT']

        # Tennis rules:
        # - Ball can bounce once on each side before being hit back
        # - If ball bounces TWICE ON SAME SIDE, point is over
        # - The player on the opposite side wins
        
        winner = None
        
        # Check for double bounce on same side
        if len(bounces) >= 2:
            # Check last two bounces
            last_bounce = bounces[-1]
            prev_bounce = bounces[-2]
            
            if last_bounce.get('side') == prev_bounce.get('side'):
                # Check if there was a HIT on opposite side after last bounce
                if hits and hits[-1].get('side') != last_bounce.get('side') and hits[-1].get('frame') > last_bounce.get('frame'):
                    # Double bounce on same side!
                    double_bounce_side = last_bounce.get('side')
                    # Opponent wins (player on other side)
                    winner = Player.UPPER if double_bounce_side == 'lower' else Player.LOWER
                    logger.info(f"Double bounce on {double_bounce_side} side! Point to {winner.name}")
        
        if winner is not None:
            self._award_point(winner)
            return True
        
        return False
    
    def _award_point(self, winner: Player):
        """
        Award point to player and update score
        
        Args:
            winner: Player who won the point
        """
        loser = Player.LOWER if winner == Player.UPPER else Player.UPPER
        
        # Tennis scoring: 0 -> 15 -> 30 -> 40 -> Game
        winner_points = self.points[winner]
        loser_points = self.points[loser]
        
        # Standard scoring
        if winner_points == 0:
            self.points[winner] = 15
        elif winner_points == 15:
            self.points[winner] = 30
        elif winner_points == 30:
            self.points[winner] = 40
        elif winner_points == 40:
            if loser_points < 40:
                # Winner wins game
                self._award_game(winner)
            elif loser_points == 40:
                # Deuce -> Advantage
                self.points[winner] = 50  # "Advantage" represented as 50
                self.points[loser] = 40
        elif winner_points == 50:  # Has advantage
            # Win game
            self._award_game(winner)
        
        self.total_points += 1
        logger.info(f"Point to {winner.name}! Score: {self._format_score()}")
    
    def _award_game(self, winner: Player):
        """Award game to player and reset point scores"""
        self.games[winner] += 1
        self.points = {Player.UPPER: 0, Player.LOWER: 0}
        
        # Check for set win (6 games with 2-game lead, or 7-6)
        loser = Player.LOWER if winner == Player.UPPER else Player.UPPER
        if (self.games[winner] >= 6 and 
            self.games[winner] - self.games[loser] >= 2):
            self._award_set(winner)
        
        # Alternate server
        self.server = Player.LOWER if self.server == Player.UPPER else Player.UPPER
        
        logger.info(f"Game to {winner.name}! Games: {self.games[Player.UPPER]}-{self.games[Player.LOWER]}")
    
    def _award_set(self, winner: Player):
        """Award set to player and reset game scores"""
        self.sets[winner] += 1
        self.games = {Player.UPPER: 0, Player.LOWER: 0}
        logger.info(f"Set to {winner.name}! Sets: {self.sets[Player.UPPER]}-{self.sets[Player.LOWER]}")
    
    def _reset_rally(self):
        """Reset rally tracking state"""
        self.rally_sequence = []
        self.frames_since_last_event = 0
        self.last_bounce_side = None
        self.last_hit_side = None
        self.total_rallies += 1
    
    def manual_award_point(self, player: Player):
        """Manually award point to player (for keyboard control)"""
        self._award_point(player)
        logger.info(f"Manual point awarded to {player.name}")
    
    def reset_score(self):
        """Reset all scores to 0-0"""
        self.points = {Player.UPPER: 0, Player.LOWER: 0}
        self.games = {Player.UPPER: 0, Player.LOWER: 0}
        self.sets = {Player.UPPER: 0, Player.LOWER: 0}
        self._reset_rally()
        logger.info("Score reset to 0-0")
    
    def _format_score(self) -> str:
        """Format current score as string"""
        # Point score
        point_map = {0: "0", 15: "15", 30: "30", 40: "40", 50: "AD"}
        upper_pts = point_map.get(self.points[Player.UPPER], str(self.points[Player.UPPER]))
        lower_pts = point_map.get(self.points[Player.LOWER], str(self.points[Player.LOWER]))
        
        # Check for deuce
        if self.points[Player.UPPER] == 40 and self.points[Player.LOWER] == 40:
            points_str = "DEUCE"
        else:
            points_str = f"{upper_pts}-{lower_pts}"
        
        return (f"Sets: {self.sets[Player.UPPER]}-{self.sets[Player.LOWER]} | "
                f"Games: {self.games[Player.UPPER]}-{self.games[Player.LOWER]} | "
                f"Points: {points_str}")
    
    def draw_scoreboard(self, frame: np.ndarray, 
                       player_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> np.ndarray:

        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Map player IDs to Upper/Lower based on Y position
        player_mapping = {}
        if player_positions:
            for player_id, pos in player_positions.items():
                player_type = self.identify_player_by_position(pos[1])
                player_mapping[player_id] = player_type
        overlay = result_frame.copy()
        scoreboard_height = 120
        scoreboard_y = 10
        padding = 15
        
        cv2.rectangle(overlay, 
                     (padding, scoreboard_y), 
                     (width - padding, scoreboard_y + scoreboard_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
        
        color_white = (255, 255, 255)
        color_yellow = (0, 255, 255)
        color_green = (0, 255, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_large = 0.8
        font_medium = 0.6
        font_small = 0.5
        thickness = 2
        
        title = "TENNIS SCOREBOARD"
        title_size = cv2.getTextSize(title, font, font_medium, thickness)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(result_frame, title, (title_x, scoreboard_y + 25),
                   font, font_medium, color_yellow, thickness, cv2.LINE_AA)
        
        y_offset = scoreboard_y + 55
        
        upper_name = "Player 1 (Rear)"
        upper_color = color_green if self.server == Player.UPPER else color_white
        cv2.putText(result_frame, upper_name, (padding + 20, y_offset),
                   font, font_small, upper_color, 1, cv2.LINE_AA)
        
        lower_name = "Player 2 (Front)"
        lower_color = color_green if self.server == Player.LOWER else color_white
        cv2.putText(result_frame, lower_name, (padding + 20, y_offset + 50),
                   font, font_small, lower_color, 1, cv2.LINE_AA)
        
        sets_x = padding + 250
        cv2.putText(result_frame, "Sets", (sets_x, scoreboard_y + 35),
                   font, font_small, color_yellow, 1, cv2.LINE_AA)
        cv2.putText(result_frame, str(self.sets[Player.UPPER]), (sets_x + 10, y_offset),
                   font, font_large, color_white, thickness, cv2.LINE_AA)
        cv2.putText(result_frame, str(self.sets[Player.LOWER]), (sets_x + 10, y_offset + 50),
                   font, font_large, color_white, thickness, cv2.LINE_AA)
        
        games_x = sets_x + 80
        cv2.putText(result_frame, "Games", (games_x, scoreboard_y + 35),
                   font, font_small, color_yellow, 1, cv2.LINE_AA)
        cv2.putText(result_frame, str(self.games[Player.UPPER]), (games_x + 20, y_offset),
                   font, font_large, color_white, thickness, cv2.LINE_AA)
        cv2.putText(result_frame, str(self.games[Player.LOWER]), (games_x + 20, y_offset + 50),
                   font, font_large, color_white, thickness, cv2.LINE_AA)
        
        points_x = games_x + 120
        cv2.putText(result_frame, "Points", (points_x, scoreboard_y + 35),
                   font, font_small, color_yellow, 1, cv2.LINE_AA)
        
        point_map = {0: "0", 15: "15", 30: "30", 40: "40", 50: "AD"}
        upper_pts = point_map.get(self.points[Player.UPPER], str(self.points[Player.UPPER]))
        lower_pts = point_map.get(self.points[Player.LOWER], str(self.points[Player.LOWER]))
        
        if self.points[Player.UPPER] == 40 and self.points[Player.LOWER] == 40:
            cv2.putText(result_frame, "DEUCE", (points_x + 10, y_offset + 25),
                       font, font_medium, color_yellow, 2, cv2.LINE_AA)
        else:
            cv2.putText(result_frame, upper_pts, (points_x + 10, y_offset),
                       font, font_large, color_white, thickness, cv2.LINE_AA)
            cv2.putText(result_frame, lower_pts, (points_x + 10, y_offset + 50),
                       font, font_large, color_white, thickness, cv2.LINE_AA)
        
        state_x = width - padding - 200
        state_text = f"State: {self.game_state.value}"
        cv2.putText(result_frame, state_text, (state_x, y_offset + 25),
                   font, font_small, color_white, 1, cv2.LINE_AA)

        if self.court_points:
            for i, point in enumerate(self.court_points):
                color = (0, 255, 255) if i >= 4 else (255, 0, 255)
                cv2.circle(result_frame, point, 5, color, -1)
        
        if self.net_left and self.net_right:
            cv2.line(result_frame, self.net_left, self.net_right, (0, 255, 255), 2)
        
        return result_frame
    
    def get_state(self) -> Dict:
        """Get current scoreboard state as dictionary"""
        return {
            "points": {
                "upper": self.points[Player.UPPER],
                "lower": self.points[Player.LOWER]
            },
            "games": {
                "upper": self.games[Player.UPPER],
                "lower": self.games[Player.LOWER]
            },
            "sets": {
                "upper": self.sets[Player.UPPER],
                "lower": self.sets[Player.LOWER]
            },
            "server": self.server.name,
            "game_state": self.game_state.value,
            "total_points": self.total_points,
            "total_rallies": self.total_rallies,
            "frame_count": self.frame_count
        }
