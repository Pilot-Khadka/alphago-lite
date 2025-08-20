// GameModeSelection.tsx
import React from 'react';
import { GameModeSelectionProps } from '../types/types';

export const GameModeSelection: React.FC<GameModeSelectionProps> = ({ 
  onStartGame, 
  loading, 
  isDark 
}) => {
  const ModeButton: React.FC<{
    mode: "human-human" | "human-bot" | "bot-bot";
    icon: string;
    title: string;
    description: string;
    gradient: string;
  }> = ({ mode, icon, title, description, gradient }) => (
    <button
      onClick={() => onStartGame(mode)}
      disabled={loading}
      className={`group relative p-6 rounded-2xl transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed border-2 ${
        loading 
          ? (isDark ? 'border-gray-600 bg-gray-700' : 'border-gray-300 bg-gray-100')
          : `${gradient} border-transparent hover:shadow-2xl`
      }`}
    >
      <div className="text-center">
        <div className={`text-4xl mb-3 transition-transform duration-300 ${
          loading ? 'grayscale' : 'group-hover:scale-110'
        }`}>
          {icon}
        </div>
        <h3 className={`text-xl font-bold mb-2 ${
          loading 
            ? (isDark ? 'text-gray-500' : 'text-gray-400')
            : 'text-white'
        }`}>
          {title}
        </h3>
        <p className={`text-sm ${
          loading 
            ? (isDark ? 'text-gray-600' : 'text-gray-500')
            : 'text-white/80'
        }`}>
          {description}
        </p>
      </div>
      
      {!loading && (
        <div className="absolute inset-0 bg-white/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      )}
    </button>
  );

  return (
    <div className={`absolute inset-0 ${
      isDark 
        ? 'bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95' 
        : 'bg-gradient-to-br from-white/95 via-gray-50/95 to-white/95'
    } backdrop-blur-xl flex flex-col justify-center items-center rounded-3xl z-10`}>
      <div className="text-center mb-8">
        <div className={`text-6xl mb-4 ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          囲
        </div>
        <h1 className={`text-3xl font-bold mb-2 ${
          isDark ? 'text-white' : 'text-gray-900'
        }`}>
          Choose Game Mode
        </h1>
        <p className={`text-lg ${
          isDark ? 'text-gray-400' : 'text-gray-600'
        }`}>
          Select how you'd like to play Go
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full px-6">
        <ModeButton
          mode="human-human"
          icon="👥"
          title="Human vs Human"
          description="Play against another person locally"
          gradient="bg-gradient-to-br from-emerald-500 to-teal-600"
        />
        
        <ModeButton
          mode="human-bot"
          icon="🤖"
          title="Human vs AI"
          description="Challenge our Go AI bot"
          gradient="bg-gradient-to-br from-blue-500 to-indigo-600"
        />
        
        <ModeButton
          mode="bot-bot"
          icon="⚡"
          title="AI vs AI"
          description="Watch two AI bots play each other"
          gradient="bg-gradient-to-br from-purple-500 to-pink-600"
        />
      </div>

      {loading && (
        <div className={`mt-8 flex items-center space-x-3 ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          <div className="relative">
            <div className="w-8 h-8 border-4 border-blue-200 rounded-full" />
            <div className="absolute top-0 left-0 w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
          <div>
            <p className="font-medium">Creating your game...</p>
            <p className={`text-sm ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              This may take a moment
            </p>
          </div>
        </div>
      )}
      
      <div className={`mt-8 text-center ${
        isDark ? 'text-gray-500' : 'text-gray-400'
      }`}>
        <p className="text-sm">
          ✨ Professional Go experience with modern design
        </p>
      </div>
    </div>
  );
};