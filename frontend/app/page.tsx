"use client";

import { type FormEvent, type MouseEvent, useEffect, useRef, useState } from "react";
import { type Answer, ApiError, type Source, ask } from "../lib/api";

// Every piece of text from the server, whether an answer, a passage of the Act or an error message, is put on the page
// as a React text node, so the browser never parses it as markup. The answer comes from a language model and the
// passages from a document nobody here controls, so neither is trusted. (There is a test for this, and one that fails
// if the raw-HTML escape hatch appears anywhere in this directory.)

// Static text, written for this page. A page file may only export the page itself, so the tests repeat them.
const EXAMPLES = [
  "Which AI practices are prohibited?",
  "What obligations apply to providers of general-purpose AI models?",
  "What does the AI Act say about human oversight of high-risk AI systems?",
  "What penalties apply when an operator breaks the AI Act?",
];

// The demo's own limits, told apart from faults so that a refusal by design does not look like breakage. `ask` gives
// the page only the message, not the error code, so these are the backend's exact messages (backend/askact/api.py,
// ERRORS; a test reads that file and fails if they drift). A message not listed here is shown as an error, as before.
const LIMITS: Record<string, { label: string; hint?: string }> = {
  "Too many requests. Please wait a moment and try again.": { label: "Rate limit" },
  "The question is too long.": { label: "Question too long", hint: "Shorten it and ask again." },
  "The request body is too large.": { label: "Question too long", hint: "Shorten it and ask again." },
  "The daily usage limit for this demo has been reached. Please try again tomorrow.": { label: "Daily limit reached" },
};

// The backend's default MAX_QUESTION_CHARS. The page cannot read the server's setting, so the count is a guide; the
// server's own check still decides, and its "too long" reply says so if a deployment set a lower limit.
const QUESTION_LIMIT = 500;
const COUNT_FROM = 400;

const REPO = "https://github.com/asadhky/AsktheEUAIAct";

type Turn = { id: number; question: string } & ({ outcome: "reply"; reply: Answer } | { outcome: "error"; message: string });

// The id carries the turn as well as the position, so a passage reached through a link (and highlighted) stays
// unhighlighted when a later answer brings a new list.
const sourceId = (turnId: number, index: number) => `source-${turnId}-${index}`;

/** One entry per cited section, in the order the sources list them, pointing at that section's first passage. */
function citedSections(sources: Source[]) {
  const sections: Array<{ sectionId: string; label: string; first: number; count: number }> = [];
  sources.forEach((source, index) => {
    const known = sections.find((section) => section.sectionId === source.section_id);
    if (known) known.count += 1;
    else sections.push({ sectionId: source.section_id, label: source.label, first: index, count: 1 });
  });
  return sections;
}

function SourceItem({ source, id, part, of, marked }: { source: Source; id: string; part: number; of: number; marked: boolean }) {
  return (
    // tabIndex -1: a passage link moves focus here, so a keyboard or screen reader continues from the passage.
    <li className={marked ? "source is-marked" : "source"} id={id} tabIndex={-1}>
      <div className="source-head">
        <h3>
          <span className="source-label">{source.label}</span>
          {source.title ? <span className="source-title">: {source.title}</span> : null}
        </h3>
        {/* Several passages of one long article read alike; the count says which part this is. */}
        {of > 1 ? (
          <span className="source-part">
            Passage {part} of {of}
          </span>
        ) : null}
      </div>
      <p className="text">{source.text}</p>
    </li>
  );
}

function ErrorReply({ message }: { message: string }) {
  const limit = Object.hasOwn(LIMITS, message) ? LIMITS[message] : null;
  // Both are alerts: either way the question was not answered, and a screen reader should hear why.
  return limit ? (
    <p className="reply limit" role="alert">
      <span className="who">{limit.label}</span>
      <span className="text">{message}</span>
      {limit.hint ? <span className="hint">{limit.hint}</span> : null}
    </p>
  ) : (
    <p className="reply error" role="alert">
      <span className="who">Error</span>
      <span className="text">{message}</span>
    </p>
  );
}

function TurnView({
  turn,
  sources,
  isLatest,
  onShowSource,
}: {
  turn: Turn;
  sources: Source[];
  isLatest: boolean;
  onShowSource: (index: number) => void;
}) {
  function show(event: MouseEvent, index: number) {
    // The link still works without this; with it, only the sources panel scrolls, not the whole page.
    event.preventDefault();
    onShowSource(index);
  }

  return (
    <article className="turn" data-turn={turn.id}>
      <p className="question">
        <span className="who">You</span>
        <span className="text">{turn.question}</span>
      </p>
      {turn.outcome === "error" ? (
        <ErrorReply message={turn.message} />
      ) : turn.reply.status === "not_covered" ? (
        <p className="reply not-covered">
          <span className="who">Not covered</span>
          <span className="text">{turn.reply.answer}</span>
        </p>
      ) : (
        <div className="reply answered">
          <span className="who">Answer</span>
          <span className="text">{turn.reply.answer}</span>
          {/* Ties the answer to its evidence: each cited section links to its passage in the sources panel. Only the
              latest answer has a panel, so an earlier one says where its passages went instead. */}
          {sources.length > 0 ? (
            <p className="cites">
              <span className="cites-label">Passages:</span>
              {citedSections(sources).map((section) => (
                <a
                  key={section.sectionId}
                  href={`#${sourceId(turn.id, section.first)}`}
                  onClick={(event) => show(event, section.first)}
                >
                  {section.label}
                  {section.count > 1 ? ` (${section.count})` : null}
                </a>
              ))}
            </p>
          ) : !isLatest && turn.reply.sources.length > 0 ? (
            <p className="cites cites-note">Sources are shown for the latest answer only.</p>
          ) : null}
        </div>
      )}
    </article>
  );
}

/** "auto" for visitors who asked for reduced motion, so a scroll jumps instead of gliding. */
function motion(): ScrollBehavior {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  // State only changes on the next render, so two submits in the same tick would both see `pending === false`.
  // The ref is what actually stops a duplicate; the disabled button is for the visitor's benefit.
  const inFlight = useRef(false);
  const nextId = useRef(1);
  // The id of the request whose reply the page is waiting for. "Stop waiting" and "Back to start" clear it, and a reply
  // that arrives for any other id is dropped. (The request itself cannot be withdrawn: the server still answers it,
  // and it still counts towards the rate limit and the daily cap.)
  const awaiting = useRef<number | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [waited, setWaited] = useState(0);
  const [marked, setMarked] = useState<string | null>(null);

  const input = useRef<HTMLInputElement>(null);
  const sourcesPanel = useRef<HTMLElement>(null);

  // Seconds since the question was sent, shown after a few so a long wait does not look like a hang.
  useEffect(() => {
    if (!pending) return;
    setWaited(0);
    const started = Date.now();
    const timer = setInterval(() => setWaited(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [pending]);

  // A new turn below the fold is brought into view. Not the first: then the page is still at its top.
  useEffect(() => {
    const newest = turns.at(-1);
    if (turns.length < 2 || !newest) return;
    const element = document.querySelector<HTMLElement>(`[data-turn="${newest.id}"]`);
    if (!element) return;
    const { top } = element.getBoundingClientRect();
    if (top < 0 || top > window.innerHeight * 0.6) element.scrollIntoView?.({ block: "start", behavior: motion() });
  }, [turns]);

  function submit(event: FormEvent) {
    event.preventDefault();
    // The Ask button disables itself while the question is sent, which would drop a click's focus to the page.
    input.current?.focus();
    void send(draft.trim());
  }

  // An example goes through exactly the path a typed question does: it is put in the box (so it stays there to be
  // sent again if it fails) and sent by `send`, with the same guard, rate limit and spend cap.
  function askExample(question: string) {
    if (inFlight.current) return;
    setDraft(question);
    // The examples disappear once the answer arrives, so focus goes where the next question is typed.
    input.current?.focus();
    void send(question);
  }

  async function send(question: string) {
    if (inFlight.current || question === "") return;
    inFlight.current = true;
    setPending(true);
    const id = nextId.current++;
    awaiting.current = id;
    setPendingQuestion(question);
    try {
      const reply = await ask(question);
      if (awaiting.current !== id) return;
      setTurns((current) => [...current, { id, question, outcome: "reply", reply }]);
      // Only if the box still holds what was sent: anything typed while waiting is the visitor's next question.
      setDraft((current) => (current.trim() === question ? "" : current));
    } catch (error) {
      if (awaiting.current !== id) return;
      // `ask` only throws ApiError, whose message is fixed text or the server's own message. The question stays in the
      // box so it can be sent again, and is selected, so typing a new one replaces it instead of adding to it.
      const message = error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
      setTurns((current) => [...current, { id, question, outcome: "error", message }]);
      input.current?.select();
    } finally {
      if (awaiting.current === id) finish();
    }
  }

  function finish() {
    awaiting.current = null;
    inFlight.current = false;
    setPending(false);
  }

  // Stops waiting for the reply; the question stays in the box.
  function stopWaiting() {
    finish();
    input.current?.focus();
  }

  // Back to the first view: the conversation is cleared, as it is when the page is reloaded.
  function startOver() {
    finish();
    setTurns([]);
    setDraft("");
    setMarked(null);
    window.scrollTo?.({ top: 0 });
    input.current?.focus();
  }

  function showSource(turnId: number, index: number) {
    const id = sourceId(turnId, index);
    const element = document.getElementById(id);
    setMarked(id);
    if (!element) return;
    const panel = sourcesPanel.current;
    // On a wide screen the panel scrolls on its own; on a narrow one it is part of the page.
    if (panel && panel.scrollHeight > panel.clientHeight && getComputedStyle(panel).overflowY !== "visible") {
      const head = panel.querySelector<HTMLElement>(".sources-head")?.offsetHeight ?? 0;
      panel.scrollTo?.({ top: element.offsetTop - head, behavior: motion() });
    } else {
      element.scrollIntoView?.({ block: "start", behavior: motion() });
    }
    element.focus({ preventScroll: true });
  }

  // The sources belong to the latest turn. They are hidden while the next question is loading, so that one question's
  // passages are never shown next to another's loading state, and for a turn that has none (not covered, or an error).
  const latest = turns.at(-1);
  // In characters as the server counts them (code points, so an emoji is one), of the question as it would be sent.
  const length = Array.from(draft.trim()).length;
  const sources = !pending && latest?.outcome === "reply" ? latest.reply.sources : [];
  // Errors and limits are alerts and the loading line is a status, so a screen reader already hears those. This says
  // that an answer, or a "not covered", has arrived. The answer itself is read from the conversation.
  const announcement =
    pending || latest?.outcome !== "reply"
      ? ""
      : latest.reply.status === "not_covered"
        ? "Not covered by the retrieved text."
        : `Answer received, with ${sources.length} ${sources.length === 1 ? "passage" : "passages"} from the Act.`;

  return (
    // Before the first question the page is one narrow, centred column; it widens once there are sources to show.
    <main className={turns.length === 0 ? "first-view" : undefined}>
      <header>
        <div className="title-row">
          <h1>Ask the EU AI Act</h1>
          {turns.length > 0 ? (
            <button type="button" className="back" onClick={startOver}>
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
                <path d="M13 8H3.5M7 4.5 3.5 8 7 11.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Back to start
            </button>
          ) : null}
        </div>
        <p className="intro">
          Ask a question in plain English. Answers are generated from passages of the Act, and those passages are shown
          with them so you can check.
        </p>
        <div className="notices">
          <p>Not legal advice.</p>
          <p>
            Answers are based on the original Official Journal text of 12 July 2024 and do not reflect later amendments,
            including Regulation (EU) 2026/1744.
          </p>
        </div>
      </header>

      <div className={turns.length === 0 ? "layout first-view" : "layout"}>
        <section className="conversation" aria-label="Conversation">
          {turns.map((turn) => (
            <TurnView
              key={turn.id}
              turn={turn}
              sources={turn === latest ? sources : []}
              isLatest={turn === latest}
              onShowSource={(index) => showSource(turn.id, index)}
            />
          ))}
          {pending ? (
            // The question being asked is shown at once, so the visitor sees what the wait is for.
            <article className="turn pending">
              <p className="question">
                <span className="who">You</span>
                <span className="text">{pendingQuestion}</span>
              </p>
              <div className="loading">
                <p role="status">
                  <span className="spinner" aria-hidden="true" />
                  Searching the Act&hellip;
                </p>
                {/* Outside the status, so a screen reader is not told the time again every second. */}
                {waited >= 5 ? (
                  <span className="waited" aria-hidden="true">
                    {waited} s
                  </span>
                ) : null}
                <button type="button" className="stop" onClick={stopWaiting}>
                  Stop waiting
                </button>
              </div>
            </article>
          ) : null}
        </section>

        {/* After the conversation in the page order, so on a narrow screen the sources follow the answer they belong
            to and the box for the next question comes last. */}
        <div className="composer">
          <form onSubmit={submit}>
            <label htmlFor="question">Your question</label>
            <div className="row">
              <input
                ref={input}
                id="question"
                type="text"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                autoComplete="off"
                aria-describedby={length > COUNT_FROM ? "question-count" : undefined}
              />
              <button type="submit" disabled={pending || draft.trim() === ""}>
                {pending ? "Asking\u2026" : "Ask"}
              </button>
            </div>
            {length > COUNT_FROM ? (
              <p id="question-count" className={length > QUESTION_LIMIT ? "count over" : "count"}>
                {length} of {QUESTION_LIMIT} characters{length > QUESTION_LIMIT ? ". Shorten it before asking." : ""}
              </p>
            ) : null}
          </form>

          {/* Only before the first question. They stay while it loads, disabled, so the page does not jump. */}
          {turns.length === 0 ? (
            <div className="examples">
              <h2 id="examples-heading">Example questions</h2>
              <ul aria-labelledby="examples-heading">
                {EXAMPLES.map((example) => (
                  <li key={example}>
                    <button type="button" className="example" disabled={pending} onClick={() => askExample(example)}>
                      <span className="example-text">{example}</span>
                      <svg className="example-arrow" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
                        <path d="M3 8h9.5M9 4.5 12.5 8 9 11.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>

        {latest && sources.length > 0 ? (
          // On a wide screen the panel stays in view beside the conversation and scrolls on its own, so it is
          // focusable: a keyboard can then scroll it too.
          <aside className="sources" aria-label="Sources" tabIndex={0} ref={sourcesPanel}>
            <div className="sources-head">
              <h2>Sources</h2>
              <p className="sources-for">
                For <q>{latest.question}</q>
              </p>
            </div>
            <ul>
              {sources.map((source, index) => {
                const same = sources.filter((other) => other.section_id === source.section_id);
                return (
                  // Not `section_id` alone: several chunks of one section can be cited, so it repeats.
                  <SourceItem
                    key={`${source.section_id}-${index}`}
                    id={sourceId(latest.id, index)}
                    source={source}
                    part={same.indexOf(source) + 1}
                    of={same.length}
                    marked={marked === sourceId(latest.id, index)}
                  />
                );
              })}
            </ul>
          </aside>
        ) : null}

        {/* In the grid, under the question box, so the sticky sources panel has room to stay in view to the end. */}
        <footer className="about">
          <p>Questions are not saved: reloading the page or going back to the start clears them.</p>
          <p>
            How it works: <a href={`${REPO}#readme`}>README</a>
            <span aria-hidden="true"> · </span>
            <a href={`${REPO}/blob/main/eval/results.md`}>Evaluation results</a>
            <span aria-hidden="true"> · </span>
            <a href={`${REPO}/tree/main/.kiro/specs/ask-the-eu-ai-act`}>Specification</a>
          </p>
        </footer>
      </div>

      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </main>
  );
}
