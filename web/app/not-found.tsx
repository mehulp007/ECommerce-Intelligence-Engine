import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-start gap-3 py-16">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-sm text-muted">That address does not match a page, customer, or product.</p>
      <Link href="/" className="text-sm text-accent">
        Back to the overview
      </Link>
    </div>
  );
}
