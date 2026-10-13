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
const button = () => screen.getByRole("button") as HTMLButtonElement;

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
