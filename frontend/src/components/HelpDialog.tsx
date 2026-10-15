import { useApp } from '../state/store';
import { Modal } from './Modal';

const ROWS: [string, string][] = [
  ['Wheel', 'Zoom about the cursor'],
  ['Drag (empty space) / Space+drag / middle-drag', 'Pan'],
  ['F', 'Fit image to window'],
  ['1 – 5', 'Raw red / raw green / composite / processed red / processed green'],
  ['H', 'Hide / show ALL overlays (check for omissions)'],
  ['V', 'Select tool (click, Shift+click multi-select, Shift+drag box select, drag a spot to move it)'],
  ['A', 'Add mode: click empty space to add a spot (class chosen in toolbar; R/G/Y/U switch class)'],
  ['R / G / Y / U', 'Set class of selected spot(s): red-only / green-only / yellow / uncertain'],
  ['Del', 'Delete selected spot(s)'],
  ['M', 'Merge selected spots (select 2 or more)'],
  ['S', 'Split selected spot, then click two positions'],
  ['Enter', 'Confirm (mark reviewed) selected spot and advance in the review queue'],
  ['J / K', 'Next / previous item in the review queue'],
  ['Ctrl+Z / Ctrl+Shift+Z', 'Undo / redo (server-side)'],
  ['Polygon tool', 'Click vertices; double-click, Enter or click the first point to finish; Backspace removes last vertex; Esc cancels'],
  ['Grid pick tool', 'Click a grid field to count it, click again to skip it; A all, N none, I invert; Esc leaves the tool'],
  ['Region vertex', 'Drag to move; double-click an edge to add; Alt+click to remove'],
  [']', 'Experiment: open the next slide that is not fully approved (on the experiment page or from one of its slides)'],
  ['Esc', 'Cancel current tool / clear selection'],
  ['?', 'This help'],
];

export function HelpDialog() {
  const open = useApp((s) => s.helpOpen);
  const patch = useApp((s) => s.patch);
  if (!open) return null;
  return (
    <Modal title="Keyboard shortcuts" onClose={() => patch({ helpOpen: false })}>
      <table className="tbl help" data-testid="help-dialog">
        <tbody>
          {ROWS.map(([k, d]) => (
            <tr key={k}>
              <th scope="row">
                <kbd>{k}</kbd>
              </th>
              <td>{d}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Modal>
  );
}
