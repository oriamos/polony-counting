import type { Analysis, CountClass, Region } from '../api/types';

export const COUNT_CLASSES: CountClass[] = ['red', 'green', 'yellow'];
const ORDER = COUNT_CLASSES;

/** Same wording as the server (`slide_save.default_label`): 'red, not yellow' | 'yellow' | 'red and yellow' ... */
export function defaultLabel(classes: CountClass[]): string {
  const set = new Set(classes);
  const names = ORDER.filter((c) => c !== 'yellow' && set.has(c)) as string[];
  const yellow = set.has('yellow');
  if (!names.length) return yellow ? 'yellow' : '';
  const head = names.length === 1 ? names[0] : names.join(' and ');
  if (!yellow) return `${head}, not yellow`;
  return names.length < 2 ? `${head} and yellow` : `${names.slice(0, -1).join(', ')}, ${names[names.length - 1]} and yellow`;
}

export interface GridField {
  field: number;
  name: string;
  polygon: [number, number][];
  counted: boolean;
  region?: Region;
}

/** Every grid field the researcher can switch: counted ones (field regions) plus switched-off ones. Sorted by number. */
export function gridFields(a: Analysis): GridField[] {
  const on = a.regions
    .filter((r) => r.kind === 'counting' && r.field_index != null)
    .map((r) => ({ field: r.field_index as number, name: r.name, polygon: r.polygon as [number, number][], counted: true, region: r }));
  const have = new Set(on.map((f) => f.field));
  const off = (a.slide.skipped_fields ?? [])
    .filter((s) => !have.has(s.field_index))
    .map((s) => ({ field: s.field_index, name: s.name, polygon: s.polygon, counted: false }));
  return [...on, ...off].sort((x, y) => x.field - y.field);
}

/** Selections the server will accept: at least one class each, labels unique. Returns messages (empty = ok). */
export function selectionProblems(sel: { classes: CountClass[]; label?: string }[]): string[] {
  const out: string[] = [];
  if (!sel.length) out.push('Add at least one thing to count.');
  sel.forEach((s, i) => {
    if (!s.classes.length) out.push(`Row ${i + 1}: tick at least one of red / green / yellow.`);
  });
  const labels = sel.map((s) => (s.label?.trim() || defaultLabel(s.classes)).toLowerCase());
  if (new Set(labels).size !== labels.length) out.push('Two rows have the same "what counted" text; make them different.');
  return out;
}

/** Folder (without the file name) of a path in either separator style. */
export function dirOf(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  return i > 0 ? p.slice(0, i) : p;
}
