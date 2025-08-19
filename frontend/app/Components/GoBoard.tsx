"use client"
import { useState } from "react";

type Stone = "black" | "white" | null;
type GameMode = "human-human" | "human-bot" | "bot-bot" | null;

interface GoBoardProps {
  size?: number;
  cellSize?: number;
  onMove?: (x: number, y: number) => void;
}

export default function GoBoard({ size = 9, cellSize = 50, onMove }: GoBoardProps) {
  const [board, setBoard] = useState<Stone[][]>(
    Array.from({ length: size }, () => Array(size).fill(null))
  );
  const [turn, setTurn] = useState<Stone>("black");
  const [gameMode, setGameMode] = useState<GameMode>(null);

  const startNewGame = (mode: GameMode) => {
    setBoard(Array.from({ length: size }, () => Array(size).fill(null)));
    setTurn("black");
    setGameMode(mode);
  };

  const handleClick = (x: number, y: number) => {
    if (!gameMode) return;
    if (board[x][y]) return;
    const newBoard = board.map(row => [...row]);
    newBoard[x][y] = turn;
    setBoard(newBoard);
    setTurn(turn === "black" ? "white" : "black");
    onMove?.(x, y);
  };

  const boardPx = cellSize * (size - 1) + 40;

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      {/* Modern overlay */}
      {!gameMode && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: 12,
            zIndex: 10,
            color: "white",
          }}
        >
          <h1 style={{ marginBottom: 24 }}>Start New Game</h1>
          <div style={{ display: "flex", gap: 16 }}>
            <button
              onClick={() => startNewGame("human-human")}
              style={{
                padding: "10px 20px",
                fontSize: 16,
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
                backgroundColor: "#4CAF50",
                color: "white",
              }}
            >
              Human vs Human
            </button>
            <button
              onClick={() => startNewGame("human-bot")}
              style={{
                padding: "10px 20px",
                fontSize: 16,
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
                backgroundColor: "#2196F3",
                color: "white",
              }}
            >
              Human vs Bot
            </button>
            <button
              onClick={() => startNewGame("bot-bot")}
              style={{
                padding: "10px 20px",
                fontSize: 16,
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
                backgroundColor: "#f44336",
                color: "white",
              }}
            >
              Bot vs Bot
            </button>
          </div>
        </div>
      )}

      {/* Go board */}
      <svg
        width={boardPx}
        height={boardPx}
        style={{
          background: "#DDB67D",
          margin: 20,
          border: "2px solid #333",
          borderRadius: 12,
        }}
      >
        {/* Grid lines */}
        {Array.from({ length: size }).map((_, i) => (
          <g key={i} stroke="black">
            <line
              x1={20}
              y1={20 + i * cellSize}
              x2={20 + (size - 1) * cellSize}
              y2={20 + i * cellSize}
            />
            <line
              x1={20 + i * cellSize}
              y1={20}
              x2={20 + i * cellSize}
              y2={20 + (size - 1) * cellSize}
            />
          </g>
        ))}

        {/* Hoshi points */}
        {size === 9 &&
          [2, 6].flatMap(x => [2, 6].map(y => ({ x, y })))
            .concat([{ x: 4, y: 4 }])
            .map(({ x, y }, idx) => (
              <circle key={idx} cx={20 + x * cellSize} cy={20 + y * cellSize} r={5} fill="black" />
            ))}

        {/* Stones (clickable) */}
        {board.map((row, x) =>
          row.map((cell, y) => (
            <circle
              key={`${x}-${y}`}
              cx={20 + x * cellSize}
              cy={20 + y * cellSize}
              r={cellSize / 2 - 4}
              fill={cell || "transparent"}
              stroke="black"
              strokeWidth={cell ? 1 : 0}
              onClick={() => !cell && handleClick(x, y)}
              style={{ cursor: !cell && gameMode ? "pointer" : "default" }}
            />
          ))
        )}
      </svg>
    </div>
  );
}
