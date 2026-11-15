/**
 * One folder with every conversation whose only *named* participant is the donor.
 *
 * Two kinds end up here, and only reading them tells them apart:
 *   - a note-to-self: nobody else in the thread at all;
 *   - a conversation whose partner Meta stripped (`participants: ["", "Elias …"]`), which looks
 *     identical once the empty name is dropped.
 *
 * Rendered by the same function the model's text comes from (`renderMessage`), so the formatting is
 * the real thing — stamps, `<no content>`, notices, markers — with one difference: labels are the
 * export's own names, because these threads may not be in the bundle at all and a label needs a
 * bundle.
 *
 *   node --experimental-strip-types 26_THESIS/make_selflike.ts \
 *        --export "26_THESIS/data/0_export unzipped/SoMe-chats" --out "26_THESIS/data/5_named-only-me"
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { renderMessage, type PromptMessage } from '/Users/e/Documents/thesis/_active/thesis-deid/src/core/standard/minimal.ts';
import { repairMojibake, stripDirectionMarks } from '/Users/e/Documents/thesis/_active/thesis-deid/src/core/standard/format.ts';

const DONOR = 'Elias Salvador Smidt Torjani';
const args = process.argv.slice(2);
const flag = (name: string): string => {
  const i = args.indexOf(name);
  return i === -1 ? '' : args[i + 1] ?? '';
};
const exportRoot = flag('--export');
const outRoot = flag('--out');
if (!exportRoot || !outRoot) throw new Error('--export and --out are required');

interface Row {
  file: string;
  platform: string;
  thread: string;
  names: Set<string>;
  unnamed: number;
  messages: any[];
}

function walk(dir: string, platform: string, found: Map<string, Row>): void {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(path, platform, found);
      continue;
    }
    if (!/^message_\d+\.json$/.test(entry.name)) continue;
    let data: any;
    try {
      data = JSON.parse(readFileSync(path, 'utf8'));
    } catch {
      continue;
    }
    const key = path.slice(0, path.lastIndexOf('/'));
    const row = found.get(key) ?? { file: key, platform, thread: key.split('/').slice(-2).join('/'), names: new Set<string>(), unnamed: 0, messages: [] };
    for (const p of data.participants ?? []) {
      const name = (typeof p === 'string' ? p : p?.name ?? '').trim();
      if (name) row.names.add(name);
      else row.unnamed += 1;
    }
    row.messages.push(...(data.messages ?? []));
    found.set(key, row);
  }
}

const rows = new Map<string, Row>();
for (const [dir, platform] of [['FB', 'fb'], ['IG', 'ig']] as const) {
  const root = join(exportRoot, dir);
  if (existsSync(root)) walk(root, platform, rows);
}

const onlyMe = [...rows.values()]
  .filter((r) => r.names.size === 1 && r.names.has(DONOR))
  .sort((a, b) => b.messages.length - a.messages.length);

if (existsSync(outRoot)) rmSync(outRoot, { recursive: true, force: true });
mkdirSync(outRoot, { recursive: true });

const index: string[] = [
  '# Conversations where the only named participant is the donor',
  '',
  'One file per conversation, flat, named by platform · how many messages · how many senders · the',
  'export folder. Rendered with the same function the model reads, with the export’s own names as',
  'labels.',
  '',
  'The `senders` column is counted from the messages, not from `participants`: Meta’s metadata can',
  'name only the donor while the messages carry a room. `reader` is what the reader then does — a',
  'thread with more than two named senders is dropped as a group, which is why the “nobody else”',
  'threads here are not in the bundle.',
  '',
  '| file | messages | senders | unnamed | reader | what it looks like |',
  '| --- | --- | --- | --- | --- | --- |',
];

for (const row of onlyMe) {
  row.messages.sort((a, b) => (a.timestamp_ms ?? a.timestamp ?? 0) - (b.timestamp_ms ?? b.timestamp ?? 0));
  let previous: number | undefined;
  const lines = [`# ${row.platform} ${row.thread}`, ''];
  for (const m of row.messages) {
    const ts = Number(m.timestamp_ms ?? m.timestamp ?? 0);
    // The export ships its strings double-encoded (UTF-8 read as Latin-1): `hvornÃ¥r` is `hvornår`.
    // The reader repairs this; a view built straight from the export has to ask the same function.
    const sender = repairMojibake(String(m.sender_name ?? m.senderName ?? '').trim())[0] || '(no sender)';
    const text = repairMojibake(stripDirectionMarks(String(m.content ?? m.text ?? '')))[0];
    const media: PromptMessage['media'] = [];
    for (const [field, kind] of [['photos', 'photos'], ['videos', 'videos'], ['audio_files', 'audio'], ['files', 'file'], ['gifs', 'gif'], ['sticker', 'sticker']] as const) {
      const items = Array.isArray(m[field]) ? m[field] : m[field] ? [m[field]] : [];
      for (const item of items) if (item?.uri) media.push({ name: String(item.uri).split('/').pop()!, kind: kind as never });
    }
    lines.push(
      renderMessage(
        {
          ts,
          sender,
          text,
          urls: [],
          media,
          reactions: (m.reactions ?? []).map((r: any) => ({ emoji: repairMojibake(String(r.reaction ?? ''))[0], actor: repairMojibake(String(r.actor ?? ''))[0] })),
          share: m.share ? { url: m.share.link, title: repairMojibake(String(m.share.share_text ?? ''))[0], owner: m.share.original_content_owner } : undefined,
          unsent: Boolean(m.is_unsent ?? m.isUnsent),
        },
        (raw) => raw,
        previous,
      ),
    );
    previous = ts;
  }
  // Counted from the messages, because that is where the truth is: `participants` named only the donor
  // while three people talked (`inbox/justyou_1437078516411345`, 35 messages, dropped as a group).
  const senders = new Set(
    row.messages.map((m) => repairMojibake(String(m.sender_name ?? m.senderName ?? '').trim())[0]).filter(Boolean),
  );
  const isEmpty = (line: string) => line.includes('<no content>');
  const empties = lines.filter(isEmpty).length;
  const shape = empties === 0 ? 'all real' : empties === row.messages.length ? 'every message empty' : `${empties} of ${row.messages.length} empty`;
  // Two threads of the same length and the same partner-shape rendered to the same filename, and the
  // second silently overwrote the first — 17 rows in the index, 16 files on disk. The export folder
  // is the identity; keep its tail, which is the part that differs.
  const tail = row.file.split('/').filter(Boolean).slice(-1)[0].slice(0, 22);
  const name = `${row.platform} · ${row.messages.length} msgs · ${senders.size} senders · ${tail}.txt`;
  writeFileSync(join(outRoot, name), lines.join('\n') + '\n');
  const verdict =
    senders.size > 2
      ? `dropped · group (${senders.size} senders)`
      : row.unnamed
        ? 'kept · partner unnamed'
        : senders.size === 2
          ? 'kept · one named partner'
          : 'kept · note to self';
  index.push(`| \`${name}\` | ${row.messages.length} | ${senders.size} | ${row.unnamed ? 'yes' : 'no'} | ${verdict} | ${shape} |`);
}

writeFileSync(join(outRoot, 'index.md'), index.join('\n') + '\n');
console.log(`${onlyMe.length} conversations written to ${outRoot}`);
