import { readFileSync } from "node:fs";
import { join } from "node:path";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import Page from "../app/page";

const NOT_LEGAL_ADVICE = "Not legal advice.";
const AMENDMENT_NOTICE =
  "Answers are based on the original Official Journal text of 12 July 2024 and do not reflect later amendments, including Regulation (EU) 2026/1744.";

const ART5 = { section_id: "article:5", label: "Article 5", title: "Prohibited AI practices", text: "The following AI practices shall be prohibited:\n(a) subliminal techniques" };
const REC1 = { section_id: "recital:1", label: "Recital 1", title: null, text: "The purpose of this Regulation is to improve the functioning of the internal market." };

function reply(over: Record<string, unknown> = {}) {
  return { status: "answered", answer: "Some practices are prohibited.", citations: ["article:5"], sources: [{ ...ART5, score: 0.7 }], ...over };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** A fetch whose replies are handed out by the test, one `resolve` per call, so the in-flight state can be inspected. */
function controlledFetch() {
  const pending: Array<(response: Response | Error) => void> = [];
  const fetchMock = vi.fn(
    () =>
      new Promise<Response>((resolve, reject) => {
        pending.push((result) => (result instanceof Error ? reject(result) : resolve(result)));
      }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, settle: (index: number, result: Response | Error) => act(async () => pending[index](result)) };
}

function stubFetch(...responses: Array<Response | Error>) {
  const queue = [...responses];
  const fetchMock = vi.fn(async () => {
    const next = queue.shift();
    if (next === undefined) throw new Error("more requests than scripted");
    if (next instanceof Error) throw next;
    return next;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const box = () => screen.getByLabelText("Your question") as HTMLInputElement;
// By name: before the first question the example questions are buttons too.
const button = () => screen.getByRole("button", { name: /^Ask/ }) as HTMLButtonElement;

async function askQuestion(user: ReturnType<typeof userEvent.setup>, text: string) {
  await user.clear(box());
  await user.type(box(), text);
  await user.click(button());
}

function expectNotices() {
  expect(screen.getByText(NOT_LEGAL_ADVICE)).toBeVisible();
  expect(screen.getByText(AMENDMENT_NOTICE)).toBeVisible();
}

afterEach(() => vi.unstubAllGlobals());

describe("the first view", () => {
  it("has a question box, a disabled Ask button, no sources and both notices", () => {
    render(<Page />);
    expect(box()).toHaveValue("");
    expect(button()).toBeDisabled();
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    expectNotices();
  });

  it("enables the button once there is something to send, and not for blank text", async () => {
    const user = userEvent.setup();
    render(<Page />);
    await user.type(box(), "   ");
    expect(button()).toBeDisabled();
    await user.type(box(), "x");
    expect(button()).toBeEnabled();
  });
});

describe("an answered question", () => {
  it("shows the question, the answer and the sources panel with label, title and text", async () => {
    stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");

    const conversation = screen.getByRole("region", { name: "Conversation" });
    expect(await within(conversation).findByText("Some practices are prohibited.")).toBeInTheDocument();
    expect(within(conversation).getByText("What is prohibited?")).toBeInTheDocument();
    expect(within(conversation).getByText("Answer")).toBeInTheDocument();

    const sources = screen.getByRole("complementary", { name: "Sources" });
    expect(within(sources).getByRole("heading", { name: "Article 5: Prohibited AI practices" })).toBeInTheDocument();
    expect(within(sources).getByText(/The following AI practices shall be prohibited:/)).toBeInTheDocument();
    expectNotices();
  });

  it("clears the box after an answer, ready for the next question", async () => {
    stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");
    await screen.findByText("Some practices are prohibited.");
    expect(box()).toHaveValue("");
  });

  it("shows a source with no title (a recital) by its label alone", async () => {
    stubFetch(json(reply({ sources: [{ ...REC1, score: 0.5 }] })));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "Why?");
    const sources = await screen.findByRole("complementary", { name: "Sources" });
    expect(within(sources).getByRole("heading", { name: "Recital 1" })).toBeInTheDocument();
    expect(within(sources).queryByText(/null|undefined/)).not.toBeInTheDocument();
  });

  it("lists every source, including two passages of the same section, without a React key warning", async () => {
    const warn = vi.spyOn(console, "error").mockImplementation(() => {});
    stubFetch(json(reply({ sources: [{ ...ART5, text: "first part" }, { ...ART5, text: "second part" }, REC1] })));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "q");
    const sources = await screen.findByRole("complementary", { name: "Sources" });
    expect(within(sources).getAllByRole("listitem")).toHaveLength(3);
    expect(within(sources).getByText("first part")).toBeInTheDocument();
    expect(within(sources).getByText("second part")).toBeInTheDocument();
    expect(warn).not.toHaveBeenCalled(); // React warns on duplicate keys through console.error
    warn.mockRestore();
  });

  it("keeps earlier turns and shows only the latest answer's sources", async () => {
    stubFetch(json(reply({ answer: "First answer." })), json(reply({ answer: "Second answer.", sources: [{ ...REC1, score: 0.4 }] })));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "first question");
    await screen.findByText("First answer.");
    await askQuestion(user, "second question");
    await screen.findByText("Second answer.");

    const conversation = screen.getByRole("region", { name: "Conversation" });
    expect(within(conversation).getByText("first question")).toBeInTheDocument();
    expect(within(conversation).getByText("First answer.")).toBeInTheDocument();
    const sources = screen.getByRole("complementary", { name: "Sources" });
    expect(within(sources).getByRole("heading", { name: "Recital 1" })).toBeInTheDocument();
    expect(within(sources).queryByRole("heading", { name: /Article 5/ })).not.toBeInTheDocument();
  });

  it("sends the trimmed question", async () => {
    const fetchMock = stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "  padded question  ");
    await screen.findByText("Some practices are prohibited.");
    const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];
    expect(JSON.parse(init.body as string)).toEqual({ question: "padded question" });
  });
});

describe("while a question is in flight", () => {
  it("shows a loading indicator and disables the button, and says so on the button", async () => {
    const { settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "slow question");

    expect(screen.getByRole("status")).toHaveTextContent("Searching the Act");
    expect(button()).toBeDisabled();
    expect(button()).toHaveTextContent("Asking");
    expectNotices();

    await settle(0, json(reply()));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(await screen.findByText("Some practices are prohibited.")).toBeInTheDocument();
  });

  it("sends one request when Enter is pressed again while the first is in flight", async () => {
    const { fetchMock, settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await user.type(box(), "double submit");
    // The browser already ignores Enter in a form whose submit button is disabled, so this does not reach the
    // handler's own guard (the next test does). It checks what a visitor can actually do.
    await user.type(box(), "{Enter}{Enter}");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await settle(0, json(reply()));
  });

  it("sends one request even when a second submit reaches the form without going through the button", async () => {
    // Autofill, a password manager or `form.requestSubmit()` can submit a form whatever its button says, so the
    // handler guards itself and does not rely on the disabled button.
    const { fetchMock, settle } = controlledFetch();
    const user = userEvent.setup();
    const { container } = render(<Page />);
    await user.type(box(), "double submit");
    const form = container.querySelector("form") as HTMLFormElement;
    fireEvent.submit(form);
    fireEvent.submit(form);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await settle(0, json(reply()));
  });

  it("sends nothing when the form is submitted with a blank question, by any route", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<Page />);
    const form = container.querySelector("form") as HTMLFormElement;
    fireEvent.submit(form);
    fireEvent.change(box(), { target: { value: "   \n  " } });
    fireEvent.submit(form);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("hides the previous answer's sources until the new answer arrives", async () => {
    const { settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "first");
    await settle(0, json(reply({ answer: "First answer." })));
    expect(await screen.findByRole("complementary", { name: "Sources" })).toBeInTheDocument();

    await askQuestion(user, "second");
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    await settle(1, json(reply({ answer: "Second answer.", sources: [{ ...REC1, score: 0.4 }] })));
    expect(await screen.findByRole("heading", { name: "Recital 1" })).toBeInTheDocument();
  });

  it("allows asking again once the request has finished, even if it failed", async () => {
    const fetchMock = stubFetch(new TypeError("down"), json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "try twice");
    await screen.findByRole("alert");
    expect(button()).toBeEnabled();
    await user.click(button());
    await screen.findByText("Some practices are prohibited.");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("a question that is not covered", () => {
  it("is shown as its own state, with the server's text, and no sources panel", async () => {
    stubFetch(json({ status: "not_covered", answer: "The retrieved text of the AI Act does not cover this question.", citations: [], sources: [] }));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is the capital of France?");

    const conversation = screen.getByRole("region", { name: "Conversation" });
    expect(await within(conversation).findByText("The retrieved text of the AI Act does not cover this question.")).toBeInTheDocument();
    expect(within(conversation).getByText("Not covered")).toBeInTheDocument();
    expect(within(conversation).queryByText("Answer")).not.toBeInTheDocument();
    expect(within(conversation).queryByRole("alert")).not.toBeInTheDocument(); // not an error
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    expectNotices();
  });

  it("is styled differently from an answer and from an error", async () => {
    stubFetch(json({ status: "not_covered", answer: "n", citations: [], sources: [] }), json(reply({ answer: "a" })));
    const user = userEvent.setup();
    const { container } = render(<Page />);
    await askQuestion(user, "one");
    await screen.findByText("Not covered");
    await askQuestion(user, "two");
    await screen.findByText("a");
    expect(container.querySelectorAll(".reply.not-covered")).toHaveLength(1);
    expect(container.querySelectorAll(".reply.answered")).toHaveLength(1);
  });
});

describe("an error", () => {
  it("shows the server's message as an alert, keeps the question in the box, and keeps the notices", async () => {
    stubFetch(json({ error: { code: "rate_limited", message: "Too many requests. Please wait a moment and try again." } }, 429));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Too many requests. Please wait a moment and try again.");
    expect(box()).toHaveValue("What is prohibited?");
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    expectNotices();
  });

  it("explains a network failure in fixed words", async () => {
    stubFetch(new TypeError("Failed to fetch"));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "q");
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not reach the server. Check your connection and try again.");
  });

  it("explains a reply it cannot read", async () => {
    stubFetch(json({ status: "answered" }));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "q");
    expect(await screen.findByRole("alert")).toHaveTextContent("The server sent a reply this page could not read. Please try again.");
  });

  it("hides the sources of an earlier answer when the latest turn failed", async () => {
    stubFetch(json(reply()), json({ error: { code: "x", message: "Nope." } }, 503));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "first");
    await screen.findByRole("complementary", { name: "Sources" });
    await askQuestion(user, "second");
    await screen.findByRole("alert");
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
  });
});

describe("the example questions", () => {
  const EXAMPLES = [
    "Which AI practices are prohibited?",
    "What obligations apply to providers of general-purpose AI models?",
    "What does the AI Act say about human oversight of high-risk AI systems?",
    "What penalties apply when an operator breaks the AI Act?",
  ];
  const examples = () => within(screen.getByRole("list", { name: "Example questions" })).getAllByRole("button");

  it("are four buttons on the first view, under a one-line intro, with both notices", () => {
    render(<Page />);
    expect(examples().map((example) => example.textContent)).toEqual(EXAMPLES);
    examples().forEach((example) => expect(example).toBeEnabled());
    expect(screen.getByText(/Answers are generated from passages of the Act/)).toBeVisible();
    expectNotices();
  });

  it("send exactly that question, through the same request as a typed one", async () => {
    const fetchMock = stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await user.click(screen.getByRole("button", { name: EXAMPLES[1] }));

    const conversation = screen.getByRole("region", { name: "Conversation" });
    expect(await within(conversation).findByText("Some practices are prohibited.")).toBeInTheDocument();
    expect(within(conversation).getByText(EXAMPLES[1])).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/ask$/);
    expect(JSON.parse(init.body as string)).toEqual({ question: EXAMPLES[1] });
    expect(box()).toHaveValue("");
    expect(box()).toHaveFocus(); // the examples are gone, so focus waits where the next question is typed
  });

  it("can be chosen from the keyboard", async () => {
    const fetchMock = stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    screen.getByRole("button", { name: EXAMPLES[0] }).focus();
    await user.keyboard("{Enter}");
    await screen.findByText("Some practices are prohibited.");
    expect(JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)).toEqual({ question: EXAMPLES[0] });
  });

  it("are disabled while the question is loading, and a second click sends nothing", async () => {
    const { fetchMock, settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await user.click(screen.getByRole("button", { name: EXAMPLES[0] }));

    expect(screen.getByRole("status")).toHaveTextContent("Searching the Act");
    examples().forEach((example) => expect(example).toBeDisabled());
    expect(button()).toBeDisabled();
    expectNotices();
    // A disabled button ignores clicks; this also reaches past it, the way a script could.
    fireEvent.click(screen.getByRole("button", { name: EXAMPLES[2] }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await settle(0, json(reply()));
  });

  it("are hidden after the first turn, whether it was answered or failed", async () => {
    stubFetch(json({ error: { code: "rate_limited", message: "Too many requests. Please wait a moment and try again." } }, 429));
    const user = userEvent.setup();
    render(<Page />);
    await user.click(screen.getByRole("button", { name: EXAMPLES[3] }));
    await screen.findByRole("alert");

    expect(screen.queryByRole("list", { name: "Example questions" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["Back to start", "Ask"]);
    expect(box()).toHaveValue(EXAMPLES[3]); // kept in the box to be sent again, like a typed question
    expectNotices();
  });

  it("are hidden after a typed first question too", async () => {
    stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");
    await screen.findByText("Some practices are prohibited.");
    expect(screen.queryByRole("list", { name: "Example questions" })).not.toBeInTheDocument();
  });
});

describe("the demo's limits", () => {
  // Read from the backend's own table, so a reworded message there fails here instead of quietly turning a limit
  // back into an "Error".
  const api = readFileSync(join(__dirname, "..", "..", "backend", "askact", "api.py"), "utf-8");
  function backendMessage(code: string): string {
    const match = api.match(new RegExp(`"${code}": \\(\\d+, "([^"]+)"\\)`));
    if (!match) throw new Error(`no message for ${code} in backend/askact/api.py`);
    return match[1];
  }
  const LIMITS = [
    { code: "rate_limited", status: 429, label: "Rate limit" },
    { code: "question_too_long", status: 422, label: "Question too long" },
    { code: "request_too_large", status: 413, label: "Question too long" },
    { code: "spend_cap_reached", status: 503, label: "Daily limit reached" },
  ];

  it.each(LIMITS)("$code is shown as a limit, not an error, and keeps the question", async ({ code, status, label }) => {
    const message = backendMessage(code);
    stubFetch(json({ error: { code, message } }, status));
    const user = userEvent.setup();
    const { container } = render(<Page />);
    await askQuestion(user, "What is prohibited?");

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(label)).toBeInTheDocument();
    expect(within(alert).getByText(message)).toBeInTheDocument();
    expect(within(alert).queryByText("Error")).not.toBeInTheDocument();
    expect(container.querySelectorAll(".reply.limit")).toHaveLength(1);
    expect(container.querySelectorAll(".reply.error")).toHaveLength(0);
    expect(box()).toHaveValue("What is prohibited?");
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    expectNotices();
  });

  it("says how to recover from a question that is too long", async () => {
    stubFetch(json({ error: { code: "question_too_long", message: backendMessage("question_too_long") } }, 422));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "q");
    expect(await screen.findByRole("alert")).toHaveTextContent("Shorten it and ask again.");
  });

  it("shows any other message as an error, as before", async () => {
    stubFetch(json({ error: { code: "llm_unavailable", message: backendMessage("llm_unavailable") } }, 503));
    const user = userEvent.setup();
    const { container } = render(<Page />);
    await askQuestion(user, "q");
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Error")).toBeInTheDocument();
    expect(container.querySelectorAll(".reply.error")).toHaveLength(1);
    expect(container.querySelectorAll(".reply.limit")).toHaveLength(0);
  });

  it("matches the whole message only, so a limit's words inside another message stay an error", async () => {
    stubFetch(json({ error: { code: "x", message: `Proxy said: ${backendMessage("rate_limited")}` } }, 500));
    const user = userEvent.setup();
    const { container } = render(<Page />);
    await askQuestion(user, "q");
    await screen.findByRole("alert");
    expect(container.querySelectorAll(".reply.error")).toHaveLength(1);
  });
});

describe("for keyboard and screen-reader users", () => {
  it("keeps focus in the question box after the Ask button is clicked", async () => {
    stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");
    await screen.findByText("Some practices are prohibited.");
    expect(box()).toHaveFocus();
  });

  it("announces an answer and a 'not covered' politely once they arrive", async () => {
    stubFetch(json(reply()), json({ status: "not_covered", answer: "n", citations: [], sources: [] }));
    const user = userEvent.setup();
    const { container } = render(<Page />);
    const live = () => container.querySelector('[aria-live="polite"]');
    expect(live()).toHaveTextContent("");
    await askQuestion(user, "one");
    await screen.findByText("Some practices are prohibited.");
    expect(live()).toHaveTextContent("Answer received, with 1 passage from the Act.");
    await askQuestion(user, "two");
    await screen.findByText("Not covered");
    expect(live()).toHaveTextContent("Not covered by the retrieved text.");
  });
});

describe("while waiting, and typing", () => {
  it("shows the question being asked at once, with a way to stop waiting", async () => {
    const { settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "slow question");
    const conversation = screen.getByRole("region", { name: "Conversation" });
    expect(within(conversation).getByText("slow question")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop waiting" })).toBeEnabled();
    await settle(0, json(reply()));
    expect(screen.queryByRole("button", { name: "Stop waiting" })).not.toBeInTheDocument();
    expect(within(conversation).getAllByText("slow question")).toHaveLength(1); // the pending echo became the turn
  });

  it("drops the reply after Stop waiting, keeps the question in the box, and can ask again", async () => {
    const { fetchMock, settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "slow question");
    await user.click(screen.getByRole("button", { name: "Stop waiting" }));

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(box()).toHaveValue("slow question");
    expect(box()).toHaveFocus();
    expect(button()).toBeEnabled();
    await settle(0, json(reply({ answer: "Too late." })));
    expect(screen.queryByText("Too late.")).not.toBeInTheDocument();

    await user.click(button());
    await settle(1, json(reply({ answer: "On time." })));
    expect(await screen.findByText("On time.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("keeps what was typed while waiting, instead of clearing it with the answer", async () => {
    const { settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "first");
    await user.type(box(), " and a follow-up");
    await settle(0, json(reply()));
    await screen.findByText("Some practices are prohibited.");
    expect(box()).toHaveValue("first and a follow-up");
  });

  it("selects the kept question after an error, so typing replaces it instead of adding to it", async () => {
    stubFetch(json({ error: { code: "rate_limited", message: "Too many requests. Please wait a moment and try again." } }, 429));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "What is prohibited?");
    await screen.findByRole("alert");
    expect(box().selectionStart).toBe(0);
    expect(box().selectionEnd).toBe("What is prohibited?".length);
  });

  it("counts characters near the 500 limit, and says when the question is over it", async () => {
    render(<Page />);
    expect(screen.queryByText(/of 500 characters/)).not.toBeInTheDocument();
    fireEvent.change(box(), { target: { value: "a".repeat(450) } });
    expect(screen.getByText("450 of 500 characters")).toBeInTheDocument();
    expect(box()).toHaveAttribute("aria-describedby", "question-count");
    fireEvent.change(box(), { target: { value: "😀".repeat(501) } }); // an emoji counts as one, as on the server
    expect(screen.getByText("501 of 500 characters. Shorten it before asking.")).toBeInTheDocument();
  });
});

describe("going back to the start", () => {
  it("clears the conversation and shows the first view again, with focus in the box", async () => {
    stubFetch(json(reply()));
    const user = userEvent.setup();
    render(<Page />);
    expect(screen.queryByRole("button", { name: "Back to start" })).not.toBeInTheDocument();
    await askQuestion(user, "What is prohibited?");
    await screen.findByText("Some practices are prohibited.");

    await user.click(screen.getByRole("button", { name: "Back to start" }));
    expect(screen.queryByText("Some practices are prohibited.")).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Example questions" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Back to start" })).not.toBeInTheDocument();
    expect(box()).toHaveValue("");
    expect(box()).toHaveFocus();
    expectNotices();
  });

  it("drops a reply that was still on its way", async () => {
    const { settle } = controlledFetch();
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "first");
    await settle(0, json(reply({ answer: "First answer." })));
    await askQuestion(user, "second");
    await user.click(screen.getByRole("button", { name: "Back to start" }));
    await settle(1, json(reply({ answer: "Second answer." })));
    expect(screen.queryByText("Second answer.")).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Example questions" })).toBeInTheDocument();
  });
});

describe("passages and help", () => {
  it("marks and focuses the passage a cited section links to", async () => {
    stubFetch(json(reply({ sources: [{ ...ART5, text: "first part" }, { ...ART5, text: "second part" }, REC1] })));
    const user = userEvent.setup();
    render(<Page />);
    await askQuestion(user, "q");
    await screen.findByRole("complementary", { name: "Sources" });

    await user.click(screen.getByRole("link", { name: "Recital 1" }));
    const passage = screen.getByRole("heading", { name: "Recital 1" }).closest("li") as HTMLElement;
    expect(passage).toHaveClass("is-marked");
    expect(passage).toHaveFocus();
    expect(screen.getByRole("link", { name: "Article 5 (2)" })).toBeInTheDocument();
  });

  it("links to the README, the evaluation results and the specification, and says nothing is saved", () => {
    render(<Page />);
    expect(screen.getByRole("link", { name: "README" })).toHaveAttribute("href", expect.stringContaining("github.com/asadhky/AsktheEUAIAct"));
    expect(screen.getByRole("link", { name: "Evaluation results" })).toHaveAttribute("href", expect.stringMatching(/eval\/results\.md$/));
    expect(screen.getByRole("link", { name: "Specification" })).toHaveAttribute("href", expect.stringMatching(/\.kiro\/specs\/ask-the-eu-ai-act$/));
    expect(screen.getByText(/Questions are not saved/)).toBeVisible();
  });
});
