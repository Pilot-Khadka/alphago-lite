"use client"
import Link from "next/link";
import React, { useState } from 'react';

export default function Home() {
  const [isDark, setIsDark] = useState(true);

  return (
    <main
      className={`relative min-h-screen flex flex-col items-center justify-center p-8 transition-colors duration-500 ${
        isDark ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'
      }`}
    >
      <div
        className={`absolute inset-0 z-0 transition-opacity duration-1000 ${
          isDark ? 'opacity-30' : 'opacity-10'
        }`}
      >
        <div
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-[100px] ${
            isDark ? 'bg-indigo-500/50' : 'bg-indigo-300/50'
          }`}
        />
      </div>

      <div className="relative z-10 text-center">
        <div className={`text-7xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-800'}`}>
          囲
        </div>
        <h1 className={`text-4xl md:text-5xl font-extrabold mb-3 tracking-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>
          The Game of Go
        </h1>
        <p className={`text-lg mb-8 max-w-lg ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Go, a game of strategy and intuition, is one of the oldest board games still played today.
        </p>

        <Link href="/go">
          <button
            className={`
              px-8 py-4 text-lg font-bold rounded-2xl transition-all duration-300
              transform hover:scale-105 shadow-xl
              bg-gradient-to-br from-blue-500 to-indigo-600 text-white
              hover:from-blue-400 hover:to-indigo-500 hover:shadow-2xl hover:shadow-blue-500/25
            `}
          >
            Play Now
          </button>
        </Link>
      </div>

      <div className="absolute bottom-6 right-6">
        <button
          onClick={() => setIsDark(!isDark)}
          className={`p-3 rounded-full transition-colors duration-300 ${
            isDark ? 'bg-gray-800 text-gray-400 hover:text-white' : 'bg-gray-200 text-gray-600 hover:text-gray-900'
          }`}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>
    </main>
  );
}