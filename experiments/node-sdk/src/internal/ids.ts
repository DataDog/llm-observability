import { randomUUID } from "node:crypto";

/**
 * Generate a 32-character hex id (a UUIDv4 with the dashes stripped) for use as
 * a span or trace id.
 */
export function hexId(): string {
  return randomUUID().replace(/-/g, "");
}
