import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, NETWORK_MESSAGE, UNEXPECTED_MESSAGE, ask, parseAnswer } from "../lib/api";

const SOURCE = { section_id: "article:5", label: "Article 5", title: "Prohibited AI practices", text: "The following shall be prohibited." };
const REPLY = { status: "answered", answer: "Some practices are prohibited.", citations: ["article:5"], sources: [{ ...SOURCE, score: 0.7 }] };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function mockFetch(result: Response | Error) {
  const fetchMock = vi.fn(async () => {
    if (result instanceof Error) throw result;
    return result;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the call to fail");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("the request", () => {
  it("posts the question as JSON to /ask on the configured backend", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.org");
    const fetchMock = mockFetch(json(REPLY));
    await ask("What is prohibited?");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://api.example.org/ask");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body as string)).toEqual({ question: "What is prohibited?" });
  });

  it("defaults to the local backend when NEXT_PUBLIC_API_URL is not set", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
    const fetchMock = mockFetch(json(REPLY));
    await ask("q");
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe("http://localhost:8000/ask");
  });

  it("does not double the slash when the configured URL ends with one", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.org///");
    const fetchMock = mockFetch(json(REPLY));
    await ask("q");
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe("https://api.example.org/ask");
  });

  it("sends the question exactly as given, including quotes and newlines", async () => {
    const fetchMock = mockFetch(json(REPLY));
    const awkward = 'He said "no"\nand <b>left</b> \u00e9';
    await ask(awkward);
    const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];
    expect(JSON.parse(init.body as string).question).toBe(awkward);
  });
});

describe("a good reply", () => {
  it("is returned with its sources, keeping only the fields the page uses", async () => {
    mockFetch(json(REPLY));
    expect(await ask("q")).toEqual({ status: "answered", answer: REPLY.answer, sources: [SOURCE] });
  });

  it("can be 'not covered' with no sources", async () => {
    mockFetch(json({ status: "not_covered", answer: "The retrieved text of the AI Act does not cover this question.", citations: [], sources: [] }));
    expect(await ask("q")).toEqual({ status: "not_covered", answer: "The retrieved text of the AI Act does not cover this question.", sources: [] });
  });

  it("accepts a source without a title (a recital)", async () => {
    mockFetch(json({ ...REPLY, sources: [{ ...SOURCE, section_id: "recital:1", label: "Recital 1", title: null }] }));
    const answer = await ask("q");
    expect(answer.sources[0].title).toBeNull();
  });
});

describe("an error from the backend", () => {
  it.each([
    [429, "rate_limited", "Too many requests. Please wait a moment and try again."],
    [422, "question_too_long", "The question is too long."],
    [503, "spend_cap_reached", "The daily usage limit for this demo has been reached. Please try again tomorrow."],
    [502, "llm_bad_output", "The answering service returned an unusable reply. Please try again."],
  ])("shows the server's own message for HTTP %i (%s)", async (status, code, message) => {
    mockFetch(json({ error: { code, message } }, status));
    expect((await failureOf(ask("q"))).message).toBe(message);
  });

  it("falls back to a fixed message with the status when the body is not JSON", async () => {
    mockFetch(new Response("<html><body>Bad gateway</body></html>", { status: 502 }));
    const error = await failureOf(ask("q"));
    expect(error.message).toBe("The server returned an error (HTTP 502). Please try again.");
    expect(error.message).not.toContain("Bad gateway"); // a proxy's page is never shown
  });

  it.each([
    ["an empty body", new Response("", { status: 500 })],
    ["JSON without an error", json({ detail: "boom" }, 500)],
    ["an error that is not an object", json({ error: "boom" }, 500)],
    ["a message that is not a string", json({ error: { code: "x", message: 5 } }, 500)],
    ["a blank message", json({ error: { code: "x", message: "   " } }, 500)],
    ["a JSON array", json([1, 2], 500)],
  ])("falls back to a fixed message for %s", async (_name, response) => {
    mockFetch(response);
    expect((await failureOf(ask("q"))).message).toBe("The server returned an error (HTTP 500). Please try again.");
  });

  it("treats an error status as an error even if the body looks like an answer", async () => {
    mockFetch(json(REPLY, 500));
    expect((await failureOf(ask("q"))).message).toBe("The server returned an error (HTTP 500). Please try again.");
  });
});

describe("a reply that cannot be used", () => {
  it("is a network error when the request itself fails, without the exception's text", async () => {
    mockFetch(new TypeError("Failed to fetch: connect ECONNREFUSED 10.0.0.7:8000"));
    const error = await failureOf(ask("q"));
    expect(error.message).toBe(NETWORK_MESSAGE);
    expect(error.message).not.toContain("10.0.0.7");
  });

  it("is rejected when a 200 body is not JSON", async () => {
    mockFetch(new Response("OK", { status: 200 }));
    expect((await failureOf(ask("q"))).message).toBe(UNEXPECTED_MESSAGE);
  });

  it.each([
    ["no status", { answer: "a", sources: [] }],
    ["an unknown status", { status: "maybe", answer: "a", sources: [] }],
    ["no answer", { status: "answered", sources: [] }],
    ["an answer that is not text", { status: "answered", answer: 5, sources: [] }],
    ["no sources", { status: "answered", answer: "a" }],
    ["sources that are not a list", { status: "answered", answer: "a", sources: {} }],
    ["a source that is not an object", { status: "answered", answer: "a", sources: ["article:5"] }],
    ["a source with no label", { status: "answered", answer: "a", sources: [{ section_id: "a", title: null, text: "t" }] }],
    ["a source with no text", { status: "answered", answer: "a", sources: [{ section_id: "a", label: "A", title: null }] }],
    ["a source with no section id", { status: "answered", answer: "a", sources: [{ label: "A", title: null, text: "t" }] }],
    ["a source title that is a number", { status: "answered", answer: "a", sources: [{ ...SOURCE, title: 7 }] }],
    ["a source title that is missing", { status: "answered", answer: "a", sources: [{ section_id: "a", label: "A", text: "t" }] }],
    ["a body that is a list", [REPLY]],
    ["a body that is null", null],
  ])("is rejected for %s, so the page never shows 'undefined'", async (_name, body) => {
    mockFetch(json(body));
    expect((await failureOf(ask("q"))).message).toBe(UNEXPECTED_MESSAGE);
  });

  it("is rejected if even one of several sources is bad", async () => {
    mockFetch(json({ ...REPLY, sources: [SOURCE, { ...SOURCE, text: 1 }] }));
    expect((await failureOf(ask("q"))).message).toBe(UNEXPECTED_MESSAGE);
  });
});

describe("parseAnswer", () => {
  it("returns null instead of throwing for anything unexpected", () => {
    for (const value of [undefined, null, 5, "text", [], {}, { status: "answered" }]) {
      expect(parseAnswer(value)).toBeNull();
    }
  });
});
