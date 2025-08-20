// GoGame.tsx
"use client"
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { GoBoard } from './GoBoard';
import { GameModeSelection } from './GameModeSelection';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { MoveHistory } from './MoveHistory';
import { GameState, GameMode } from '../types/types';

const API_URL = 'http://127.0.0.1:5000/api';

export const GoGame: React.FC<{ size: number }> = ({ size }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameMode, setGameMode] = useState<GameMode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [isDark, setIsDark] = useState(true);

  // Poll for game state updates, especially for bot-bot games
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (gameState && gameState.id && (gameMode === "bot-bot" || (gameMode === "human-bot" && gameState.current_player === "white")) && !isPaused && !gameState.is_over) {
      interval = setInterval(() => {
        fetchGameStatus();
      }, 1000); // Poll every second
    } else {
      if (interval) {
        clearInterval(interval);
      }
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [gameState, gameMode, isPaused]);

  const fetchGameStatus = useCallback(async () => {
    if (!gameState?.id) return;
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}`);
      if (!response.ok) {
        throw new Error("Failed to fetch game state");
      }
      const data = await response.json();
      setGameState(data.game);
    } catch (err: any) {
      setError(err.message || "An error occurred fetching game status.");
      setLoading(false);
    }
  }, [gameState]);

  const startNewGame = useCallback(async (mode: GameMode) => {
    setLoading(true);
    setError(null);
    setGameMode(mode);
    try {
      const response = await fetch(`${API_URL}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board_size: size, mode }),
      });
      if (!response.ok) {
        throw new Error("Failed to create new game");
      }
      const data = await response.json();
      setGameState(data.game);
    } catch (err: any) {
      setError(err.message || "An error occurred while creating the game.");
    } finally {
      setLoading(false);
    }
  }, [size]);

  const handleMove = useCallback(async (row: number, col: number) => {
    if (!gameState?.id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}/moves`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row, col }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to make move");
      }
      const data = await response.json();
      setGameState(data.game);
    } catch (err: any) {
      setError(err.message || "An error occurred while making the move.");
    } finally {
      setLoading(false);
    }
  }, [gameState]);

  const handlePass = useCallback(async () => {
    if (!gameState?.id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}/moves`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "pass" }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to pass turn");
      }
      const data = await response.json();
      setGameState(data.game);
    } catch (err: any) {
      setError(err.message || "An error occurred while passing the turn.");
    } finally {
      setLoading(false);
    }
  }, [gameState]);
  
  const handleTogglePause = useCallback(async () => {
    if (!gameState?.id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}/pause`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paused: !isPaused }),
      });
      if (!response.ok) {
        throw new Error("Failed to pause/resume game");
      }
      const data = await response.json();
      setGameState(data.game);
      setIsPaused(!isPaused);
    } catch (err: any) {
      setError(err.message || "An error occurred while pausing/resuming.");
    } finally {
      setLoading(false);
    }
  }, [gameState, isPaused]);

  const handleAnalyze = useCallback(async () => {
    if (!gameState?.id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}/analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        throw new Error("Failed to trigger analysis");
      }
      const data = await response.json();
      setGameState(data.game);
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  }, [gameState]);

  const handleReset = useCallback(async () => {
    if (!gameState?.id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/games/${gameState.id}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error("Failed to delete game");
      }
      setGameState(null);
      setGameMode(null);
    } catch (err: any) {
      setError(err.message || "An error occurred while resetting the game.");
    } finally {
      setLoading(false);
    }
  }, [gameState]);

  return (
    <div className={`p-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
      <div className={`flex items-center justify-between mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
        <h1 className="text-xl font-bold">Go Game</h1>
        <button
          onClick={() => setIsDark(!isDark)}
          className={`p-2 rounded-full transition-colors duration-300 ${
            isDark ? 'bg-gray-800 text-gray-400 hover:text-white' : 'bg-gray-200 text-gray-600 hover:text-gray-900'
          }`}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>

      {error && <div className={`text-red-500 mb-4 ${isDark ? 'bg-red-900/50' : 'bg-red-100'} p-3 rounded-lg`}>{error}</div>}

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="w-full lg:w-1/3 space-y-6">
          {gameState && <GameInfo gameState={gameState} isDark={isDark} />}
          {gameState && (
            <GameControls
              gameState={gameState}
              gameMode={gameMode}
              loading={loading}
              onPass={handlePass}
              onTogglePause={handleTogglePause}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              isPaused={isPaused}
              isDark={isDark}
            />
          )}
          {gameState && <MoveHistory moveHistory={gameState.move_history} isDark={isDark} />}
        </div>
        <div className="w-full lg:w-2/3">
          {!gameMode ? (
            <GameModeSelection onStartGame={startNewGame} loading={loading} isDark={isDark} />
          ) : (
            <GoBoard
              gameState={gameState}
              gameMode={gameMode}
              size={size}
              cellSize={40}
              loading={loading}
              onMove={handleMove}
              isDark={isDark}
            />
          )}
        </div>
      </div>
    </div>
  );
};