import React from 'react';
import { GameControlsProps } from '../types/types';

export const GameControls: React.FC<GameControlsProps> = ({
  gameState,
  gameMode,
  loading,
  onPass,
  onTogglePause,
  onAnalyze,
  onReset,
  isPaused
}) => {
  const showPassButton = gameMode === "human-human" ||
    (gameMode === "human-bot" && gameState.current_player === "black");

  const Button: React.FC<{
    onClick: () => void;
    disabled?: boolean;
    children: React.ReactNode;
  }> = ({ onClick, disabled, children }) => {

    const baseStyles =
      "px-4 py-3 rounded-xl font-light transition-colors duration-200 border";

    if (disabled) {
      return (
        <button
          onClick={onClick}
          disabled
          className={`${baseStyles} bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed`}
        >
          {children}
        </button>
      );
    }

    return (
      <button
        onClick={onClick}
        className={`${baseStyles} bg-slate-200 hover:bg-slate-300 text-slate-700 border-slate-300`}
      >
        {children}
      </button>
    );
  };

  return (
    <div className="bg-slate-100/80 backdrop-blur-md border border-slate-200 rounded-2xl p-6 shadow-sm transition-all duration-300">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-8 h-8 rounded-lg bg-slate-200 text-slate-600 flex items-center justify-center">
          <span className="text-sm font-medium">⚡</span>
        </div>
        <h3 className="text-lg font-light tracking-wide text-slate-700">
          Game Controls
        </h3>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {showPassButton && (
          <Button
            onClick={onPass}
            disabled={loading || gameState.status !== "playing"}
          >
            <span className="flex items-center justify-center space-x-2">
              <span>⏭️</span>
              <span>Pass Turn</span>
            </span>
          </Button>
        )}

        <Button
          onClick={onTogglePause}
          disabled={loading || gameState.is_over}
        >
          <span className="flex items-center justify-center space-x-2">
            <span>{isPaused ? '▶️' : '⏸️'}</span>
            <span>{isPaused ? "Resume" : "Pause"}</span>
          </span>
        </Button>

        <Button
          onClick={onAnalyze}
          disabled={loading}
        >
          <span className="flex items-center justify-center space-x-2">
            <span>🧠</span>
            <span>Analyze Position</span>
          </span>
        </Button>

        <Button
          onClick={onReset}
          disabled={loading}
        >
          <span className="flex items-center justify-center space-x-2">
            <span>🔄</span>
            <span>New Game</span>
          </span>
        </Button>
      </div>

      {loading && (
        <div className="mt-4 flex items-center justify-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" />
          <div className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" style={{ animationDelay: '0.15s' }} />
          <div className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" style={{ animationDelay: '0.3s' }} />
          <span className="text-sm text-slate-500 font-light ml-2">
            Processing...
          </span>
        </div>
      )}
    </div>
  );
};