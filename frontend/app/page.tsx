"use client";

import { type FormEvent, useRef, useState } from "react";
import { type Answer, ApiError, type Source, ask } from "../lib/api";

// Every piece of text from the server, whether an answer, a passage of the Act or an error message, is put on the page
// as a React text node, so the browser never parses it as markup. The answer comes from a language model and the
// passages from a document nobody here controls, so neither is trusted. (There is a test for this, and one that fails
// if the raw-HTML escape hatch appears anywhere in this directory.)

type Turn = { id: number; question: string } & ({ outcome: "reply"; reply: Answer } | { outcome: "error"; message: string });

function SourceItem({ source }: { source: Source }) {
  return (
    <li className="source">
      <h3>
        {source.label}
        {source.title ? <span className="source-title">: {source.title}</span> : null}
      </h3>
      <p className="text">{source.text}</p>
    </li>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <article className="turn">
      <p className="question">
        <span className="who">You</span>
        <span className="text">{turn.question}</span>
      </p>
      {turn.outcome === "error" ? (
        <p className="reply error" role="alert">
          <span className="who">Error</span>
          <span className="text">{turn.message}</span>
        </p>
      ) : (
        <p className={turn.reply.status === "not_covered" ? "reply not-covered" : "reply answered"}>
          <span className="who">{turn.reply.status === "not_covered" ? "Not covered" : "Answer"}</span>
          <span className="text">{turn.reply.answer}</span>
        </p>
      )}
    </article>
  );
}

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  // State only changes on the next render, so two submits in the same tick would both see `pending === false`.
  // The ref is what actually stops a duplicate; the disabled button is for the visitor's benefit.
  const inFlight = useRef(false);
  const nextId = useRef(1);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const question = draft.trim();
    if (inFlight.current || question === "") return;
    inFlight.current = true;
    setPending(true);
    const id = nextId.current++;
    try {
      const reply = await ask(question);
      setTurns((current) => [...current, { id, question, outcome: "reply", reply }]);
      setDraft("");
    } catch (error) {
      // `ask` only throws ApiError, whose message is fixed text or the server's own message. The question stays in the
      // box so it can be sent again.
      const message = error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
      setTurns((current) => [...current, { id, question, outcome: "error", message }]);
    } finally {
      inFlight.current = false;
      setPending(false);
    }
  }

  // The sources belong to the latest turn. They are hidden while the next question is loading, so that one question's
  // passages are never shown next to another's loading state, and for a turn that has none (not covered, or an error).
  const latest = turns.at(-1);
  const sources = !pending && latest?.outcome === "reply" ? latest.reply.sources : [];

  return (
    <main>
      <header>
        <h1>Ask the EU AI Act</h1>
        <div className="notices">
          <p>Not legal advice.</p>
          <p>
            Answers are based on the original Official Journal text of 12 July 2024 and do not reflect later amendments,
            including Regulation (EU) 2026/1744.
          </p>
        </div>
      </header>

      <div className="layout">
        <section className="conversation" aria-label="Conversation">
          {turns.map((turn) => (
            <TurnView key={turn.id} turn={turn} />
          ))}
          {pending ? (
            <p className="loading" role="status">
              <span className="spinner" aria-hidden="true" />
              Searching the Act&hellip;
            </p>
          ) : null}

          <form onSubmit={submit}>
            <label htmlFor="question">Your question</label>
            <div className="row">
              <input
                id="question"
                type="text"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                autoComplete="off"
              />
              <button type="submit" disabled={pending || draft.trim() === ""}>
                {pending ? "Asking\u2026" : "Ask"}
              </button>
            </div>
          </form>
        </section>

        {sources.length > 0 ? (
          <aside className="sources" aria-label="Sources">
            <h2>Sources</h2>
            <ul>
              {sources.map((source, index) => (
                // Not `section_id` alone: several chunks of one section can be cited, so it repeats.
                <SourceItem key={`${source.section_id}-${index}`} source={source} />
              ))}
            </ul>
          </aside>
        ) : null}
      </div>
    </main>
  );
}
