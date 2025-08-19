import Link from "next/link";

export default function Home() {
  return (
    <main className=" relative flex flex-col items-center justify-center min-h-screen gap-4">
      {/* Background glow */}
      <div className="opacity-60 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 sm:w-[600px] sm:h-[600px] w-[300px] h-[300px] bg-yellow-400/40 rounded-full blur-3xl -z-10"></div>
      </div>

      <h1 className="text-2xl font-bold text-white">Go</h1>

      <div className="flex gap-4">
        <Link href="/go">
          <button className="p-4 bg-white text-slate-900 rounded-lg font-semibold cursor-pointer hover:bg-slate-200 transition">
            New Game
          </button>
        </Link>
      </div>
    </main>
  );
}


