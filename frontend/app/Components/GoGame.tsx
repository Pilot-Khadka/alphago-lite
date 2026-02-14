"use client"
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { GoBoard } from './GoBoard';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { MoveHistory } from './MoveHistory';
import { GameState, GameMode } from '../types/types';
import { io, Socket } from "socket.io-client";
import { useRouter } from 'next/navigation';

const API_URL = 'http://127.0.0.1:5000/api';

interface GoGameProps {
  size: number;
  initialMode?: string;
}

export const GoGame: React.FC<GoGameProps> = ({ size, initialMode }) => {
  const router = useRouter();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameMode, setGameMode] = useState<GameMode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const gameInitialized = useRef(false);

  useEffect(() => {
    if (!socketRef.current) {
      socketRef.current = io('http://127.0.0.1:5000');
      socketRef.current.on('connect', () => {
        console.log('Connected to server via WebSocket');
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

    if (gameState?.id && socketRef.current?.connected) {
      console.log(`Attempting to join game room: ${gameState.id}`);
      socketRef.current?.emit('join_game', { room: gameState.id });
    }

    return () => {
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

  useEffect(() => {
    if (initialMode && !gameInitialized.current) {
      gameInitialized.current = true;

      const modeMap: { [key: string]: GameMode } = {
        'pvp': 'human-human',
        'ai': 'human-bot',
        'spectate': 'bot-bot'
      };

      const mode = modeMap[initialMode] || 'human-bot';
      startNewGame(mode);
    }
  }, [initialMode, startNewGame]);

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
      gameInitialized.current = false;

      if (initialMode) {
        const modeMap: { [key: string]: GameMode } = {
          'pvp': 'human-human',
          'ai': 'human-bot',
          'spectate': 'bot-bot'
        };
        const mode = modeMap[initialMode] || 'human-bot';
        startNewGame(mode);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred while resetting the game.");
    } finally {
      setLoading(false);
    }
  }, [gameState, initialMode, startNewGame]);

  const handleBackToHome = () => {
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-[#f5f0e8] p-6">
      <div className="flex items-center justify-between mb-6 text-slate-800">
        <div className="flex items-center gap-4">
          <button
            onClick={handleBackToHome}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/70 hover:bg-white border border-slate-300 transition-all duration-200 shadow-sm hover:shadow-md"
          >
            <span>←</span>
            <span className="font-light">Back to Home</span>
          </button>
          <h1 className="text-2xl font-light tracking-wide">Go Game</h1>
        </div>
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg shadow-sm">
          {error}
        </div>
      )}

      {!gameState && loading && (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="relative mx-auto w-12 h-12 mb-4">
              <div className="w-12 h-12 border-4 border-slate-300 rounded-full" />
              <div className="absolute top-0 left-0 w-12 h-12 border-4 border-slate-800 border-t-transparent rounded-full animate-spin" />
            </div>
            <p className="text-slate-600 font-light">Creating your game...</p>
          </div>
        </div>
      )}

      {gameState && (
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-1/3 space-y-6">
            <GameInfo gameState={gameState} />
            <GameControls
              gameState={gameState}
              gameMode={gameMode}
              loading={loading}
              onPass={handlePass}
              onTogglePause={handleTogglePause}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              isPaused={isPaused}
            />
            <MoveHistory moveHistory={gameState.move_history} />
          </div>
          <div className="w-full lg:w-2/3">
            <GoBoard
              gameState={gameState}
              gameMode={gameMode}
              size={size}
              cellSize={40}
              loading={loading}
              onMove={handleMove}
            />
          </div>
        </div>
      )}
    </div>
  );
};