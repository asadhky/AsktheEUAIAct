import { readFileSync } from "node:fs";
import { join } from "node:path";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import Page from "../app/page";

// Answers come from a language model and passages from a document nobody here controls, so both are untrusted.
// They must appear on the page as the literal characters, and must not create elements.
const HOSTILE = "<img src=x onerror=alert(1)> and <script>alert(1)</script> and <b>bold</b> &amp; &lt;i&gt;";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

async function askOnce(response: Response, question = "a question") {
  vi.stubGlobal("fetch", vi.fn(async () => response));
  const user = userEvent.setup();
  const view = render(<Page />);
  await user.type(screen.getByLabelText("Your question"), question);
  await user.click(screen.getByRole("button"));
  return view;
}

/** No element that the hostile text could have produced, anywhere in the document. */
function expectNoInjectedElements(root: HTMLElement) {
  expect(root.querySelector("img")).toBeNull();
  expect(root.querySelector("script")).toBeNull();
  expect(root.querySelector("b")).toBeNull();
  expect(root.querySelector("i")).toBeNull();
  expect(root.querySelector("[onerror]")).toBeNull();
}

afterEach(() => vi.unstubAllGlobals());

describe("untrusted text is shown as text", () => {
  it("in an answer", async () => {
    const { container } = await askOnce(json({ status: "answered", answer: HOSTILE, citations: [], sources: [] }));
    const conversation = await screen.findByRole("region", { name: "Conversation" });
    await within(conversation).findByText(HOSTILE);
    expect(conversation.textContent).toContain(HOSTILE); // the exact characters, nothing decoded or stripped
    expectNoInjectedElements(container);
  });

  it("in a source passage, a source title and a source label", async () => {
    const source = { section_id: "article:5", label: `Article <b>5</b>`, title: "<script>alert(1)</script>", text: HOSTILE, score: 0.5 };
    const { container } = await askOnce(json({ status: "answered", answer: "ok", citations: ["article:5"], sources: [source] }));
    const sources = await screen.findByRole("complementary", { name: "Sources" });
    expect(sources.textContent).toContain(HOSTILE);
    expect(sources.textContent).toContain("Article <b>5</b>");
    expect(sources.textContent).toContain("<script>alert(1)</script>");
    expectNoInjectedElements(container);
  });

  it("in an error message from the server", async () => {
    const { container } = await askOnce(json({ error: { code: "x", message: HOSTILE } }, 500));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(HOSTILE);
    expectNoInjectedElements(container);
  });

  it("in a 'not covered' reply", async () => {
    const { container } = await askOnce(json({ status: "not_covered", answer: HOSTILE, citations: [], sources: [] }));
    await screen.findByText("Not covered");
    expect(container.textContent).toContain(HOSTILE);
    expectNoInjectedElements(container);
  });

  it("in the visitor's own question, when it is echoed in the conversation", async () => {
    const { container } = await askOnce(json({ status: "answered", answer: "ok", citations: [], sources: [] }), HOSTILE);
    await screen.findByText("ok");
    expect(screen.getByRole("region", { name: "Conversation" }).textContent).toContain(HOSTILE);
    expectNoInjectedElements(container);
  });

  it("keeps line breaks through CSS, not by turning text into markup", async () => {
    const { container } = await askOnce(json({ status: "answered", answer: "line one\nline two", citations: [], sources: [] }));
    await screen.findByText(/line one/);
    const text = container.querySelector(".reply .text");
    expect(text?.textContent).toBe("line one\nline two");
    expect(container.querySelector("br")).toBeNull();
    // Without this rule the newline would collapse to a space on screen.
    const css = readFileSync(join(__dirname, "..", "app", "globals.css"), "utf-8");
    expect(css).toMatch(/\.text\s*\{[^}]*white-space:\s*pre-wrap/);
  });
});
