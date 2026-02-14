import React from 'react';
import { MoveHistoryProps } from '../types/types';

export const MoveHistory: React.FC<MoveHistoryProps> = ({ moveHistory }) => {
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
    return player === 'black' ? 'text-slate-800' : 'text-slate-600';
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
    <div className="bg-white/70 backdrop-blur-lg border border-slate-300 rounded-2xl p-6 shadow-lg transition-all duration-300">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-8 h-8 rounded-lg bg-slate-800/10 text-slate-700 flex items-center justify-center">
          <span className="text-sm font-bold">📋</span>
        </div>
        <h3 className="text-lg font-light tracking-wide text-slate-800">
          Move History
        </h3>
        <span className="text-sm px-2 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
          {moveHistory.length}
        </span>
      </div>

      <div className="space-y-2">
        <div className="max-h-48 overflow-y-auto custom-scrollbar">
          {moveHistory.slice(-10).map((move, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-3 rounded-lg transition-colors hover:bg-slate-50 bg-slate-50/50 border border-slate-200/50"
            >
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-4 h-4 rounded-full shadow-sm ${
                    move.player === 'black'
                      ? 'bg-gradient-to-br from-slate-700 to-slate-900 border-2 border-slate-600'
                      : 'bg-gradient-to-br from-slate-100 to-white border-2 border-slate-300'
                  }`} />
                  <span className={`text-sm font-normal ${getPlayerColor(move.player)}`}>
                    {move.player.charAt(0).toUpperCase()}
                  </span>
                </div>

                <span className="text-base">
                  {getMoveTypeIcon(move.move_type)}
                </span>

                <code className="px-2 py-1 rounded text-xs font-mono bg-slate-100 text-slate-700 border border-slate-200">
                  {formatMove(move)}
                </code>
              </div>

              <span className="text-xs text-slate-400">
                #{moveHistory.length - moveHistory.slice(-10).length + idx + 1}
              </span>
            </div>
          ))}
        </div>

        {moveHistory.length > 10 && (
          <div className="text-center text-sm text-slate-500 font-light pt-2 border-t border-slate-200">
            Showing last 10 moves of {moveHistory.length}
          </div>
        )}
      </div>

      {moveHistory.length === 0 && (
        <div className="text-center py-8 text-slate-500">
          <div className="text-2xl mb-2">🎯</div>
          <p className="text-sm font-light">No moves yet</p>
          <p className="text-xs mt-1 text-slate-400">Game history will appear here</p>
        </div>
      )}
    </div>
  );
};