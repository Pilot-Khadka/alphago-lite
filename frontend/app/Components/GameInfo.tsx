// GameInfo.tsx
import React from 'react';
import { GameInfoProps } from '../types/types';

export const GameInfo: React.FC<GameInfoProps> = ({ gameState, isDark }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'playing':
        return isDark ? 'text-green-400' : 'text-green-600';
      case 'paused':
        return isDark ? 'text-yellow-400' : 'text-yellow-600';
      case 'finished':
        return isDark ? 'text-red-400' : 'text-red-600';
      default:
        return isDark ? 'text-gray-400' : 'text-gray-600';
    }
  };

  const getPlayerColor = (player: string) => {
    return player === 'black' 
      ? (isDark ? 'text-gray-300' : 'text-gray-800')
      : (isDark ? 'text-gray-400' : 'text-gray-600');
  };

  return (
    <div className={`${
      isDark 
        ? 'bg-gray-800/50 border-gray-700' 
        : 'bg-white/70 border-gray-200'
    } backdrop-blur-lg border rounded-2xl p-6 transition-all duration-300 hover:shadow-lg`}>
      <div className="flex items-center space-x-3 mb-4">
        <div className={`w-8 h-8 rounded-lg ${
          isDark ? 'bg-blue-500/20 text-blue-400' : 'bg-blue-500/10 text-blue-600'
        } flex items-center justify-center`}>
          <span className="text-sm font-bold">ℹ</span>
        </div>
        <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Game Info
        </h3>
      </div>
      
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Game ID
          </span>
          <code className={`px-2 py-1 rounded text-xs font-mono ${
            isDark ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'
          }`}>
            {gameState.id.slice(0, 8)}...
          </code>
        </div>
        
        <div className="flex justify-between items-center">
          <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Current Player
          </span>
          <div className="flex items-center space-x-2">
            <div className={`w-4 h-4 rounded-full ${
              gameState.current_player === 'black' ? 'bg-gray-800' : 'bg-gray-300'
            } border-2 ${isDark ? 'border-gray-600' : 'border-gray-400'}`} />
            <span className={`text-sm font-medium ${getPlayerColor(gameState.current_player)}`}>
              {gameState.current_player.charAt(0).toUpperCase() + gameState.current_player.slice(1)}
            </span>
          </div>
        </div>
        
        <div className="flex justify-between items-center">
          <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Status
          </span>
          <span className={`text-sm font-medium capitalize ${getStatusColor(gameState.status)}`}>
            {gameState.status}
          </span>
        </div>
        
        {gameState.is_over && (
          <div className={`mt-4 p-3 rounded-lg text-center ${
            isDark 
              ? 'bg-green-900/30 border border-green-700 text-green-400' 
              : 'bg-green-50 border border-green-200 text-green-700'
          }`}>
            <span className="font-medium">🎉 Game Completed!</span>
          </div>
        )}
      </div>
    </div>
  );
};