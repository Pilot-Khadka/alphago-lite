import GoBoard from "../Components/GoBoard";
export default function GoPage() {
    console.log("Go page is loading!");  // Check browser console
  return (
    <main className="flex flex-col items-center gap-4 p-8">
      <h1 className="text-2xl font-bold">Go</h1>
      <GoBoard size={9} />
    </main>
  );
}
