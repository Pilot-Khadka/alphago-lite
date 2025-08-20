"use client"
import { GoGame } from "../Components/GoGame"

export default function GoPage() {
  console.log("Go page is loading!");
  return (
    <main className="flex flex-col items-center gap-4 p-8">
      <h1 className="text-2xl font-bold">Go</h1>
      <GoGame size={13} />
    </main>
  );
}