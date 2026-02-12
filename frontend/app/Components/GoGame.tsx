// GoGame.tsx
// GoGame.tsx
"use client"
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { GoBoard } from './GoBoard';
import { GameModeSelection } from './GameModeSelection';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { MoveHistory } from './MoveHistory';
import { GameState, GameMode } from '../types/types';
import { io, Socket } from "socket.io-client";

// The URL of your Flask backend.
const API_URL = 'http://127.0.0.1:5000/api';

export const GoGame: React.FC<{ size: number }> = ({ size }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameMode, setGameMode] = useState<GameMode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [isDark, setIsDark] = useState(true);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Only set up the connection once
    if (!socketRef.current) {
        socketRef.current = io('http://127.0.0.1:5000');
        socketRef.current.on('connect', () => {
            console.log('Connected to server via WebSocket');
            // If we have a game ID, join the room immediately
            if (gameState?.id) {
                socketRef.current?.emit('join_game', { room: gameState.id });
            }
        });
        socketRef.current.on('game_update', (data: { game: GameState }) => {
            console.log('Game update received:', data.game);
            setGameState(data.game);
            setLoading(false);
        });
        socketRef.current.on('disconnect', () => {
            console.log('Disconnected from server');
        });
    }

    // Join the room whenever the gameState changes to a new game
    if (gameState?.id && socketRef.current?.connected) {
        console.log(`Attempting to join game room: ${gameState.id}`);
        socketRef.current?.emit('join_game', { room: gameState.id });
    }

    return () => {
        // Clean up the socket connection when the component unmounts
        if (socketRef.current) {
            socketRef.current.disconnect();
            socketRef.current = null;
        }
    };
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
      // Do NOT set game state here. The WebSocket will handle the update.
    } catch (err: any) {
      setError(err.message || "An error occurred while making the move.");
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
      // Do NOT set game state here. The WebSocket will handle the update.
    } catch (err: any) {
      setError(err.message || "An error occurred while passing the turn.");
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
      // Do NOT set game state here. The WebSocket will handle the update.
    } catch (err: any) {
      setError(err.message || "An error occurred while pausing/resuming.");
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
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
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