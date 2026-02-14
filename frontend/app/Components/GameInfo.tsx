import React from 'react';
import { GameInfoProps } from '../types/types';

export const GameInfo: React.FC<GameInfoProps> = ({ gameState }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'playing':
        return 'text-emerald-600';
      case 'paused':
        return 'text-amber-600';
      case 'finished':
        return 'text-rose-600';
      default:
        return 'text-slate-600';
    }
  };

  const getPlayerColor = (player: string) => {
    return player === 'black' ? 'text-slate-800' : 'text-slate-600';
  };

  return (
    <div className="bg-white/70 backdrop-blur-lg border border-slate-300 rounded-2xl p-6 shadow-lg transition-all duration-300 hover:shadow-xl">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-8 h-8 rounded-lg bg-slate-800/10 text-slate-700 flex items-center justify-center">
          <span className="text-sm font-bold">ℹ</span>
        </div>
        <h3 className="text-lg font-light tracking-wide text-slate-800">
          Game Info
        </h3>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-slate-600 font-light">
            Game ID
          </span>
          <code className="px-3 py-1 rounded-lg text-xs font-mono bg-slate-100 text-slate-700 border border-slate-200">
            {gameState.id.slice(0, 8)}...
          </code>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-slate-600 font-light">
            Current Player
          </span>
          <div className="flex items-center space-x-2">
            <div className={`w-5 h-5 rounded-full shadow-sm ${
              gameState.current_player === 'black'
                ? 'bg-gradient-to-br from-slate-700 to-slate-900 border-2 border-slate-600'
                : 'bg-gradient-to-br from-slate-100 to-white border-2 border-slate-300'
            }`} />
            <span className={`text-sm font-normal ${getPlayerColor(gameState.current_player)}`}>
              {gameState.current_player.charAt(0).toUpperCase() + gameState.current_player.slice(1)}
            </span>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-slate-600 font-light">
            Status
          </span>
          <span className={`text-sm font-normal capitalize ${getStatusColor(gameState.status)}`}>
            {gameState.status}
          </span>
        </div>

        {gameState.is_over && (
          <div className="mt-4 p-3 rounded-lg text-center bg-emerald-50 border border-emerald-200">
            <span className="font-normal text-emerald-700">🎉 Game Completed!</span>
          </div>
        )}
      </div>
    </div>
  );
};