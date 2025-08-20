// GameControls.tsx
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
  isPaused,
  isDark
}) => {
  const showPassButton = gameMode === "human-human" || 
    (gameMode === "human-bot" && gameState.current_player === "black");

  const Button: React.FC<{
    onClick: () => void;
    disabled?: boolean;
    variant: 'pass' | 'pause' | 'analyze' | 'reset';
    children: React.ReactNode;
  }> = ({ onClick, disabled, variant, children }) => {
    const getVariantStyles = () => {
      const baseStyles = "px-4 py-2.5 rounded-xl font-medium transition-all duration-200 transform hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed";
      
      if (disabled) {
        return `${baseStyles} ${
          isDark 
            ? 'bg-gray-700 text-gray-500 border border-gray-600' 
            : 'bg-gray-100 text-gray-400 border border-gray-200'
        }`;
      }

      switch (variant) {
        case 'pass':
          return `${baseStyles} ${
            isDark 
              ? 'bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-500 hover:to-orange-500 text-white shadow-lg hover:shadow-yellow-500/25' 
              : 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-400 hover:to-orange-400 text-white shadow-lg hover:shadow-yellow-500/25'
          }`;
        case 'pause':
          return `${baseStyles} ${
            isDark 
              ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg hover:shadow-blue-500/25' 
              : 'bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-400 hover:to-indigo-400 text-white shadow-lg hover:shadow-blue-500/25'
          }`;
        case 'analyze':
          return `${baseStyles} ${
            isDark 
              ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white shadow-lg hover:shadow-purple-500/25' 
              : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-400 hover:to-pink-400 text-white shadow-lg hover:shadow-purple-500/25'
          }`;
        case 'reset':
          return `${baseStyles} ${
            isDark 
              ? 'bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white shadow-lg hover:shadow-red-500/25' 
              : 'bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-400 hover:to-rose-400 text-white shadow-lg hover:shadow-red-500/25'
          }`;
      }
    };

    return (
      <button
        onClick={onClick}
        disabled={disabled}
        className={getVariantStyles()}
      >
        {children}
      </button>
    );
  };

  return (
    <div className={`${
      isDark 
        ? 'bg-gray-800/50 border-gray-700' 
        : 'bg-white/70 border-gray-200'
    } backdrop-blur-lg border rounded-2xl p-6 transition-all duration-300`}>
      <div className="flex items-center space-x-3 mb-4">
        <div className={`w-8 h-8 rounded-lg ${
          isDark ? 'bg-indigo-500/20 text-indigo-400' : 'bg-indigo-500/10 text-indigo-600'
        } flex items-center justify-center`}>
          <span className="text-sm font-bold">⚡</span>
        </div>
        <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Game Controls
        </h3>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {showPassButton && (
          <Button
            onClick={onPass}
            disabled={loading || gameState.status !== "playing"}
            variant="pass"
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
          variant="pause"
        >
          <span className="flex items-center justify-center space-x-2">
            <span>{isPaused ? '▶️' : '⏸️'}</span>
            <span>{isPaused ? "Resume" : "Pause"}</span>
          </span>
        </Button>
        
        <Button
          onClick={onAnalyze}
          disabled={loading}
          variant="analyze"
        >
          <span className="flex items-center justify-center space-x-2">
            <span>🧠</span>
            <span>Analyze Position</span>
          </span>
        </Button>
        
        <Button
          onClick={onReset}
          disabled={loading}
          variant="reset"
        >
          <span className="flex items-center justify-center space-x-2">
            <span>🔄</span>
            <span>New Game</span>
          </span>
        </Button>
      </div>

      {loading && (
        <div className="mt-4 flex items-center justify-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            isDark ? 'bg-blue-400' : 'bg-blue-600'
          } animate-pulse`} />
          <div className={`w-2 h-2 rounded-full ${
            isDark ? 'bg-blue-400' : 'bg-blue-600'
          } animate-pulse`} style={{ animationDelay: '0.1s' }} />
          <div className={`w-2 h-2 rounded-full ${
            isDark ? 'bg-blue-400' : 'bg-blue-600'
          } animate-pulse`} style={{ animationDelay: '0.2s' }} />
          <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'} ml-2`}>
            Processing...
          </span>
        </div>
      )}
    </div>
  );
};