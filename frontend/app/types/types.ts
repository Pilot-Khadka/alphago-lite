export type Stone = "black" | "white" | null;
export type GameMode = "human-human" | "human-bot" | "bot-bot" | null;
export type GameStatus = "waiting" | "playing" | "paused" | "finished";

export interface GameState {
  id: string;
  board_size: number;
  board_state: Stone[][];
  current_player: "black" | "white";
  gameMode: GameMode;
  status: GameStatus;
  is_over: boolean;
  move_history: any[];
  analysis?: any;
}

export interface GameInfoProps {
  gameState: GameState;
}

export interface GameControlsProps {
  gameState: GameState;
  gameMode: GameMode;
  loading: boolean;
  onPass: () => void;
  onTogglePause: () => void;
  onAnalyze: () => void;
  onReset: () => void;
  isPaused: boolean;
}

export interface GoBoardProps {
  gameState: GameState | null;
  gameMode: GameMode;
  size: number;
  cellSize: number;
  loading: boolean;
  onMove: (row: number, col: number) => void;
}

export interface GameModeSelectionProps {
  onStartGame: (mode: GameMode) => void;
  loading: boolean;
}

export interface MoveHistoryProps {
  moveHistory: any[];
}