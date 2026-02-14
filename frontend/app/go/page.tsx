"use client"
import { GoGame } from "../Components/GoGame"
import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';

function GoPageContent() {
  const searchParams = useSearchParams();
  const mode = searchParams.get('mode') || 'ai';

  return <GoGame size={19} initialMode={mode} />;
}

export default function GoPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-[#f5f0e8] flex items-center justify-center">
        <div className="text-slate-800 font-light">Loading game...</div>
      </div>
    }>
      <GoPageContent />
    </Suspense>
  );
}