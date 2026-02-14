"use client"
import { useRouter } from "next/navigation";
import React, { useState } from 'react';

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleStartGame = async (mode: "human-human" | "human-bot" | "bot-bot") => {
    setLoading(true);

    const modeMap = {
      "human-human": "pvp",
      "human-bot": "ai",
      "bot-bot": "spectate"
    };

    await new Promise(resolve => setTimeout(resolve, 500));
    router.push(`/go?mode=${modeMap[mode]}`);
  };

  const ModeButton: React.FC<{
    mode: "human-human" | "human-bot" | "bot-bot";
    icon: string;
    title: string;
    description: string;
    primary?: boolean;
  }> = ({ mode, icon, title, description, primary = false }) => (
    <button
      onClick={() => handleStartGame(mode)}
      disabled={loading}
      className={`group relative px-8 py-5 text-lg font-light rounded-sm transition-all duration-300 shadow-lg hover:shadow-xl overflow-hidden disabled:cursor-not-allowed disabled:opacity-50 ${
        primary
          ? 'bg-slate-800 text-[#f5f0e8] hover:bg-slate-700'
          : 'bg-white text-slate-800 hover:bg-slate-50 border border-slate-300'
      }`}
    >
      <span className="relative z-10 flex items-center gap-3">
        <span className="text-2xl">{icon}</span>
        <span className="flex flex-col items-start">
          <span className="font-normal">{title}</span>
          <span className={`text-xs font-light ${
            primary ? 'text-[#f5f0e8]/70' : 'text-slate-600'
          }`}>
            {description}
          </span>
        </span>
      </span>
      {primary && (
        <div className="absolute inset-0 bg-gradient-to-r from-slate-700 to-slate-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      )}
    </button>
  );

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center p-8 bg-[#f5f0e8] overflow-hidden">
      <div className="absolute inset-0 z-0 opacity-40">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="goban" x="0" y="0" width="60" height="60" patternUnits="userSpaceOnUse">
              <rect width="60" height="60" fill="none"/>
              <line x1="30" y1="0" x2="30" y2="60" stroke="#d4c5b0" strokeWidth="0.5"/>
              <line x1="0" y1="30" x2="60" y2="30" stroke="#d4c5b0" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#goban)"/>
        </svg>
      </div>

      <div className="absolute top-12 left-12 z-0 opacity-15">
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-slate-800 to-slate-600 shadow-2xl" />
      </div>

      <div className="absolute bottom-16 right-16 z-0 opacity-15">
        <div className="w-32 h-32 rounded-full bg-gradient-to-br from-slate-100 to-white shadow-2xl border border-slate-300" />
      </div>

      <div className="absolute top-1/4 right-1/4 z-0 opacity-10">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-800 to-slate-600" />
      </div>

      <div className="relative z-10 text-center max-w-3xl">
        <div className="mb-6 inline-block">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 rounded-full bg-slate-800" />
            <div className="w-2 h-2 rounded-full bg-slate-600" />
            <div className="w-2 h-2 rounded-full bg-slate-400" />
          </div>
        </div>

<h1 className="text-6xl md:text-7xl font-light mb-4 tracking-wide text-slate-800">
  Go
</h1>

<div className="h-px w-24 mx-auto bg-slate-300 mb-6" />

<p className="text-lg md:text-xl font-light mb-4 text-slate-700">
  A board game of strategy played by placing black and white stones to control territory.
</p>

<p className="text-base mb-12 max-w-2xl mx-auto text-slate-600 leading-relaxed">
  This project explores the game of Go and how modern deep learning systems can learn to play it.
</p>

        <div className="flex flex-col gap-4 max-w-xl mx-auto">
          <ModeButton
            mode="human-bot"
            icon="🤖"
            title="Play Against AI"
            description="Challenge our Go AI bot"
            primary
          />

          <ModeButton
            mode="human-human"
            icon="👥"
            title="Human vs Human"
            description="Play against another person locally"
          />

          <ModeButton
            mode="bot-bot"
            icon="⚡"
            title="AI vs AI"
            description="Watch two AI bots play each other"
          />
        </div>

        {loading && (
          <div className="mt-8 flex items-center justify-center space-x-3 text-slate-800">
            <div className="relative">
              <div className="w-6 h-6 border-3 border-slate-300 rounded-full" />
              <div className="absolute top-0 left-0 w-6 h-6 border-3 border-slate-800 border-t-transparent rounded-full animate-spin" />
            </div>
            <p className="font-light">Creating your game...</p>
          </div>
        )}
      </div>
    </main>
  );
}