// The browser's only conversation with the backend: `POST {base}/ask`.
//
// The base URL is NEXT_PUBLIC_API_URL. Next.js inlines that variable into the client bundle when the UI is *built*, so
// it cannot be changed at container start (see .env.example). It is read inside `ask` so a test can set it, and it must
// be written exactly as `process.env.NEXT_PUBLIC_API_URL`, which is what the inliner looks for.

export type Source = {
  section_id: string; // not unique in a list: several chunks of one section can all be cited
  label: string; // "Article 5"
  title: string | null; // null for recitals
  text: string;
};

export type Answer = {
  status: "answered" | "not_covered";
  answer: string;
  sources: Source[];
};

/** Anything that stopped a question from being answered. `message` is safe to show to the visitor as plain text. */
export class ApiError extends Error {}

const DEFAULT_BASE = "http://localhost:8000";

// Fixed texts. Nothing below puts a caught exception's text, or a body the server sent that we could not understand, in
// front of the visitor: those can be a proxy's HTML error page or a stack of internals, and say nothing useful.
export const NETWORK_MESSAGE = "Could not reach the server. Check your connection and try again.";
export const UNEXPECTED_MESSAGE = "The server sent a reply this page could not read. Please try again.";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseSource(value: unknown): Source | null {
  if (!isRecord(value)) return null;
  const { section_id, label, title, text } = value;
  if (typeof section_id !== "string" || typeof label !== "string" || typeof text !== "string") return null;
  if (title !== null && typeof title !== "string") return null;
  return { section_id, label, title, text };
}

/** The reply as an `Answer`, or null if it is not the shape the backend promises. Checked, not assumed: an answer
 *  with a missing field would otherwise render as the text "undefined". */
export function parseAnswer(data: unknown): Answer | null {
  if (!isRecord(data)) return null;
  const { status, answer, sources } = data;
  if ((status !== "answered" && status !== "not_covered") || typeof answer !== "string" || !Array.isArray(sources)) {
    return null;
  }
  const parsed = sources.map(parseSource);
  if (parsed.some((source) => source === null)) return null;
  return { status, answer, sources: parsed as Source[] };
}

/** The backend's own message from `{"error": {"code", "message"}}`, if the body has one. */
function errorMessage(data: unknown): string | null {
  if (!isRecord(data) || !isRecord(data.error)) return null;
  const { message } = data.error;
  return typeof message === "string" && message.trim() !== "" ? message : null;
}

export async function ask(question: string): Promise<Answer> {
  const base = (process.env.NEXT_PUBLIC_API_URL || DEFAULT_BASE).replace(/\/+$/, "");
  let response: Response;
  try {
    response = await fetch(`${base}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new ApiError(NETWORK_MESSAGE);
  }

  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    // Not JSON (a gateway's error page, an empty body): handled below like any other unreadable reply.
  }

  if (!response.ok) {
    throw new ApiError(errorMessage(data) ?? `The server returned an error (HTTP ${response.status}). Please try again.`);
  }
  const answer = parseAnswer(data);
  if (answer === null) throw new ApiError(UNEXPECTED_MESSAGE);
  return answer;
}
