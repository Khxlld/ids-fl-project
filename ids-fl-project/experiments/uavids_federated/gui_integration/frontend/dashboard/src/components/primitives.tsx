import type { ReactNode } from 'react'

export function Panel({
  title,
  note,
  children,
  flush = false,
}: {
  title: string
  note?: ReactNode
  children: ReactNode
  /** Drop body padding for edge-to-edge content such as log lists. */
  flush?: boolean
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">{title}</h2>
        {note !== undefined && <span className="panel-note">{note}</span>}
      </div>
      <div className={flush ? 'panel-body tight' : 'panel-body'}>{children}</div>
    </section>
  )
}

export function Readout({
  label,
  value,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  tone?: 'default' | 'ok' | 'crit' | 'warn' | 'dim'
}) {
  const toneClass = tone === 'default' ? '' : ` is-${tone}`
  return (
    <div className="readout">
      <span className="readout-k">{label}</span>
      <span className={`readout-v${toneClass}`}>{value}</span>
    </div>
  )
}

export function Cell({
  value,
  label,
  tone = 'default',
}: {
  value: ReactNode
  label: string
  tone?: 'default' | 'ok' | 'crit' | 'warn'
}) {
  const toneClass = tone === 'default' ? '' : ` is-${tone}`
  return (
    <div className="cell">
      <div className={`cell-v${toneClass}`}>{value}</div>
      <div className="cell-k">{label}</div>
    </div>
  )
}

export function KeyValue({ items }: { items: [string, ReactNode][] }) {
  return (
    <div className="kv">
      {items.map(([key, value]) => (
        <div key={key}>
          <div className="kv-k">{key}</div>
          <div className="kv-v">{value}</div>
        </div>
      ))}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>
}

export const formatClock = (iso: string): string => {
  const date = new Date(iso)
  return Number.isNaN(date.getTime())
    ? '--:--:--'
    : date.toLocaleTimeString('en-GB', { hour12: false })
}

export const pct = (value: number | undefined, digits = 1): string =>
  value === undefined || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`
