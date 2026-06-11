import { randomUUID } from "node:crypto";

/**
 * Generate a 32-character hex id (a UUIDv4 with the dashes stripped), matching
 * the Java SDK's `UUID.randomUUID().toString().replace("-", "")` for span and
 * trace ids.
 */
export function hexId(): string {
  return randomUUID().replace(/-/g, "");
}
