import React from 'react';
import { GoBoardProps } from '../types/types';

export const GoBoard: React.FC<GoBoardProps> = ({
  gameState,
  gameMode,
  size,
  cellSize,
  loading,
  onMove
}) => {
  const boardPx = cellSize * (size - 1) + 40;
  const board = gameState?.board_state || Array.from({ length: size }, () => Array(size).fill(null));

  const allowClicks = gameState && gameState.status === "playing" && !loading &&
    (gameMode === "human-human" ||
     (gameMode === "human-bot" && gameState.current_player === "black"));

  const handleCellClick = (row: number, col: number) => {
    if (!allowClicks || board[row][col]) return;
    onMove(row, col);
  };

  return (
    <div className="bg-white/70 backdrop-blur-lg border border-slate-300 rounded-2xl p-6 shadow-xl transition-all duration-300">
      <svg
        width={boardPx}
        height={boardPx}
        style={{
          background: "linear-gradient(135deg, #DDB67D 0%, #E8C99D 50%, #F4E4BC 100%)",
          borderRadius: "16px",
          filter: loading ? "blur(1px) opacity(0.7)" : "none",
          transition: "all 0.3s ease",
        }}
        className="shadow-lg"
      >
        {Array.from({ length: size }).map((_, i) => (
          <g key={i} stroke="#8B6914" strokeWidth={1.5} opacity={0.7}>
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

        {size === 19 &&
          [3, 9, 15].flatMap(x => [3, 9, 15].map(y => ({ x, y })))
            .map(({ x, y }, idx) => (
              <circle
                key={idx}
                cx={20 + x * cellSize}
                cy={20 + y * cellSize}
                r={4}
                fill="#654321"
                opacity={0.5}
              />
            ))}

        {size === 9 &&
          [2, 6].flatMap(x => [2, 6].map(y => ({ x, y })))
            .concat([{ x: 4, y: 4 }])
            .map(({ x, y }, idx) => (
              <circle
                key={idx}
                cx={20 + x * cellSize}
                cy={20 + y * cellSize}
                r={3}
                fill="#654321"
                opacity={0.5}
              />
            ))}

        {allowClicks && Array.from({ length: size }).map((_, x) =>
          Array.from({ length: size }).map((_, y) => {
            if (board[x][y]) return null;
            return (
              <circle
                key={`hover-${x}-${y}`}
                cx={20 + x * cellSize}
                cy={20 + y * cellSize}
                r={cellSize / 2 - 2}
                fill="transparent"
                stroke="transparent"
                className="hover:fill-slate-800 hover:fill-opacity-10 transition-all duration-200 cursor-pointer"
                onClick={() => handleCellClick(x, y)}
              />
            );
          })
        )}

        {board.map((row, x) =>
          row.map((cell, y) => {
            if (!cell) return null;

            const isBlack = cell === 'black';
            const shadowColor = isBlack ? 'rgba(0,0,0,0.5)' : 'rgba(0,0,0,0.2)';

            return (
              <g key={`${x}-${y}`}>
                <ellipse
                  cx={20 + x * cellSize + 2}
                  cy={20 + y * cellSize + 2}
                  rx={cellSize / 2 - 2}
                  ry={cellSize / 2 - 3}
                  fill={shadowColor}
                />

                <circle
                  cx={20 + x * cellSize}
                  cy={20 + y * cellSize}
                  r={cellSize / 2 - 2}
                  fill={isBlack
                    ? "url(#blackStoneGradient)"
                    : "url(#whiteStoneGradient)"
                  }
                  stroke={isBlack ? "#0a0a0a" : "#e0e0e0"}
                  strokeWidth={1}
                />
              </g>
            );
          })
        )}

        <defs>
          <radialGradient id="blackStoneGradient" cx="35%" cy="35%">
            <stop offset="0%" stopColor="#4a4a4a" />
            <stop offset="60%" stopColor="#2a2a2a" />
            <stop offset="100%" stopColor="#1a1a1a" />
          </radialGradient>
          <radialGradient id="whiteStoneGradient" cx="35%" cy="35%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="60%" stopColor="#f5f5f5" />
            <stop offset="100%" stopColor="#e8e8e8" />
          </radialGradient>
        </defs>
      </svg>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/10 rounded-2xl backdrop-blur-sm">
          <div className="bg-white px-6 py-3 rounded-xl shadow-lg flex items-center space-x-3 border border-slate-300">
            <div className="w-4 h-4 border-2 border-slate-800 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm font-light text-slate-700">Processing move...</span>
          </div>
        </div>
      )}
    </div>
  );
};