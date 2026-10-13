import { useEffect, useState } from "react";

/** The value, once it has stopped changing for `delay` milliseconds. */
export function useDebounced<T>(value: T, delay = 250): T {
  const [settled, setSettled] = useState(value);
  const key = JSON.stringify(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(JSON.parse(key) as T), delay);
    return () => clearTimeout(timer);
  }, [key, delay]);
  return settled;
}
