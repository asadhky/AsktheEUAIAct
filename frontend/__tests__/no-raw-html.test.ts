import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

// The task forbids the React escape hatch for inserting raw HTML anywhere in this project. A test that scans files
// is only worth something if it can be shown to find the thing, so `scan` is exercised on a sample as well.

const ROOT = join(__dirname, "..");
const SKIPPED_DIRECTORIES = new Set(["node_modules", ".next"]);
const FORBIDDEN = "dangerouslySetInnerHTML";

function filesUnder(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return SKIPPED_DIRECTORIES.has(name) ? [] : filesUnder(path);
    return [path];
  });
}

function scan(files: string[], read: (path: string) => string): string[] {
  return files.filter((file) => read(file).includes(FORBIDDEN));
}

describe("no raw HTML insertion", () => {
  const files = filesUnder(ROOT);

  it("looks at the source files (the scan is not empty)", () => {
    const names = files.map((file) => relative(ROOT, file));
    for (const expected of ["app/page.tsx", "app/layout.tsx", "lib/api.ts", "package.json"]) {
      expect(names).toContain(expected);
    }
    expect(names.some((name) => name.startsWith("node_modules"))).toBe(false);
    expect(names.some((name) => name.startsWith(".next"))).toBe(false);
  });

  it("finds the forbidden name when it is there", () => {
    expect(scan(["a.tsx", "b.tsx"], (file) => (file === "b.tsx" ? `<div ${FORBIDDEN}={{ __html: x }} />` : "<div />"))).toEqual(["b.tsx"]);
  });

  it("finds nothing in the project's own files, other than this test's definition of the name", () => {
    const self = join(ROOT, "__tests__", "no-raw-html.test.ts");
    const offenders = scan(files.filter((file) => file !== self), (path) => readFileSync(path, "utf-8"));
    expect(offenders.map((file) => relative(ROOT, file))).toEqual([]);
  });
});
