// GoBoard.tsx
import React from 'react';
import { GoBoardProps } from '../types/types';

export const GoBoard: React.FC<GoBoardProps> = ({ 
  gameState, 
  gameMode, 
  size, 
  cellSize, 
  loading, 
  onMove,
  isDark
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
    <div className={`${
      isDark 
        ? 'bg-gray-800/30 border-gray-700 shadow-2xl' 
        : 'bg-white/90 border-gray-200 shadow-xl'
    } backdrop-blur-lg border rounded-3xl p-6 transition-all duration-300`}>
      <svg
        width={boardPx}
        height={boardPx}
        style={{
          background: isDark 
            ? "linear-gradient(135deg, #8B4513 0%, #A0522D 100%)"
            : "linear-gradient(135deg, #DDB67D 0%, #F4E4BC 100%)",
          borderRadius: "16px",
          filter: loading ? "blur(1px) opacity(0.7)" : "none",
          transition: "all 0.3s ease",
        }}
        className="shadow-lg"
      >
        {/* Grid lines */}
        {Array.from({ length: size }).map((_, i) => (
          <g key={i} stroke={isDark ? "#654321" : "#8B7355"} strokeWidth={1.5}>
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

        {/* Star points for 9x9 board */}
        {size === 9 &&
          [2, 6].flatMap(x => [2, 6].map(y => ({ x, y })))
            .concat([{ x: 4, y: 4 }])
            .map(({ x, y }, idx) => (
              <circle 
                key={idx} 
                cx={20 + x * cellSize} 
                cy={20 + y * cellSize} 
                r={3} 
                fill={isDark ? "#4A4A4A" : "#333333"}
                opacity={0.6}
              />
            ))}

        {/* Intersection hover effects */}
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
                className="hover:fill-black hover:fill-opacity-10 transition-all duration-200 cursor-pointer"
                onClick={() => handleCellClick(x, y)}
              />
            );
          })
        )}

        {/* Stones */}
        {board.map((row, x) =>
          row.map((cell, y) => {
            if (!cell) return null;
            
            const isBlack = cell === 'black';
            const shadowColor = isBlack 
              ? (isDark ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.6)')
              : (isDark ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.3)');
            
            return (
              <g key={`${x}-${y}`}>
                {/* Stone shadow */}
                <ellipse
                  cx={20 + x * cellSize + 2}
                  cy={20 + y * cellSize + 2}
                  rx={cellSize / 2 - 2}
                  ry={cellSize / 2 - 3}
                  fill={shadowColor}
                />
                
                {/* Stone */}
                <circle
                  cx={20 + x * cellSize}
                  cy={20 + y * cellSize}
                  r={cellSize / 2 - 2}
                  fill={isBlack 
                    ? "url(#blackStoneGradient)" 
                    : "url(#whiteStoneGradient)"
                  }
                  stroke={isBlack ? "#1a1a1a" : "#d0d0d0"}
                  strokeWidth={1}
                />
              </g>
            );
          })
        )}

        {/* Gradients for stones */}
        <defs>
          <radialGradient id="blackStoneGradient" cx="30%" cy="30%">
            <stop offset="0%" stopColor="#4a4a4a" />
            <stop offset="70%" stopColor="#1a1a1a" />
            <stop offset="100%" stopColor="#000000" />
          </radialGradient>
          <radialGradient id="whiteStoneGradient" cx="30%" cy="30%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="70%" stopColor="#f0f0f0" />
            <stop offset="100%" stopColor="#d0d0d0" />
          </radialGradient>
        </defs>
      </svg>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-20 rounded-3xl">
          <div className={`${
            isDark ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'
          } px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2`}>
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm font-medium">Processing move...</span>
          </div>
        </div>
      )}
    </div>
  );
};