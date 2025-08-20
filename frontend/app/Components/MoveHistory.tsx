// MoveHistory.tsx
import React from 'react';
import { MoveHistoryProps } from '../types/types';

export const MoveHistory: React.FC<MoveHistoryProps> = ({ moveHistory, isDark }) => {
  if (moveHistory.length === 0) return null;

  const formatMove = (move: any) => {
    if (move.move_type === "play") {
      const col = String.fromCharCode(65 + move.position.col);
      const row = move.position.row + 1;
      return `${col}${row}`;
    }
    return move.move_type.toUpperCase();
  };

  const getPlayerColor = (player: string) => {
    return player === 'black' 
      ? (isDark ? 'text-gray-300' : 'text-gray-800')
      : (isDark ? 'text-gray-400' : 'text-gray-600');
  };

  const getMoveTypeIcon = (moveType: string) => {
    switch (moveType) {
      case 'play':
        return '●';
      case 'pass':
        return '⏭️';
      default:
        return '•';
    }
  };

  return (
    <div className={`${
      isDark 
        ? 'bg-gray-800/50 border-gray-700' 
        : 'bg-white/70 border-gray-200'
    } backdrop-blur-lg border rounded-2xl p-6 transition-all duration-300`}>
      <div className="flex items-center space-x-3 mb-4">
        <div className={`w-8 h-8 rounded-lg ${
          isDark ? 'bg-green-500/20 text-green-400' : 'bg-green-500/10 text-green-600'
        } flex items-center justify-center`}>
          <span className="text-sm font-bold">📋</span>
        </div>
        <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Move History
        </h3>
        <span className={`text-sm px-2 py-1 rounded-full ${
          isDark 
            ? 'bg-gray-700 text-gray-300' 
            : 'bg-gray-100 text-gray-600'
        }`}>
          {moveHistory.length}
        </span>
      </div>

      <div className="space-y-2">
        <div className={`max-h-48 overflow-y-auto ${
          isDark ? 'scrollbar-dark' : 'scrollbar-light'
        }`}>
          {moveHistory.slice(-10).map((move, idx) => (
            <div 
              key={idx} 
              className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
                isDark 
                  ? 'hover:bg-gray-700/50 bg-gray-700/20' 
                  : 'hover:bg-gray-50 bg-gray-50/50'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-4 h-4 rounded-full ${
                    move.player === 'black' 
                      ? 'bg-gray-800 border-2 border-gray-600' 
                      : 'bg-gray-200 border-2 border-gray-400'
                  }`} />
                  <span className={`text-sm font-medium ${getPlayerColor(move.player)}`}>
                    {move.player.charAt(0).toUpperCase()}
                  </span>
                </div>
                
                <span className="text-lg">
                  {getMoveTypeIcon(move.move_type)}
                </span>
                
                <code className={`px-2 py-1 rounded text-xs font-mono ${
                  isDark 
                    ? 'bg-gray-800 text-gray-300' 
                    : 'bg-gray-100 text-gray-700'
                }`}>
                  {formatMove(move)}
                </code>
              </div>
              
              <span className={`text-xs ${
                isDark ? 'text-gray-500' : 'text-gray-400'
              }`}>
                #{moveHistory.length - moveHistory.slice(-10).length + idx + 1}
              </span>
            </div>
          ))}
        </div>
        
        {moveHistory.length > 10 && (
          <div className={`text-center text-sm ${
            isDark ? 'text-gray-500' : 'text-gray-400'
          } pt-2 border-t ${
            isDark ? 'border-gray-700' : 'border-gray-200'
          }`}>
            Showing last 10 moves of {moveHistory.length}
          </div>
        )}
      </div>
      
      {moveHistory.length === 0 && (
        <div className={`text-center py-8 ${
          isDark ? 'text-gray-500' : 'text-gray-400'
        }`}>
          <div className="text-2xl mb-2">🎯</div>
          <p className="text-sm">No moves yet</p>
          <p className="text-xs mt-1">Game history will appear here</p>
        </div>
      )}
    </div>
  );
};