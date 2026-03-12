import { useEffect, useMemo, useState } from 'react'
import './App.css'

const formatNumber = (value) =>
  Number(value ?? 0).toLocaleString('en-US', { maximumFractionDigits: 1 })

const buildSparkPath = (points, width = 280, height = 120) => {
  if (!points.length) return ''
  const max = Math.max(...points)
  const min = Math.min(...points)
  const scaleX = width / (points.length - 1 || 1)
  const scaleY = max === min ? 1 : height / (max - min)

  return points
    .map((p, i) => {
      const x = i * scaleX
      const y = height - (p - min) * scaleY
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

const fallbackData = {
  summary: { detections: 0, accuracy: 0, latency_p95_ms: 0, uptime: 0 },
  trend: { points: [], forecast_change_pct: 0 },
  emotions: [],
  events: [],
}

function App() {
  const [data, setData] = useState(fallbackData)
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchData = async () => {
    try {
      setStatus((s) => (s === 'ready' ? 'refreshing' : 'loading'))
      const res = await fetch('/api/dashboard')
      if (!res.ok) throw new Error(`API ${res.status}`)
      const json = await res.json()
      setData(json)
      setLastUpdated(Date.now())
      setStatus('ready')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, 5000)
    return () => clearInterval(id)
  }, [])

  const trendPoints = useMemo(() => data.trend.points ?? [], [data])
  const emotions = data.emotions ?? []
  const events = data.events ?? []
  const summary = data.summary ?? fallbackData.summary

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Emotion Recognition • Live Ops</p>
          <h1>Operational Command Dashboard</h1>
          <p className="lede">
            Real data from the trained speech emotion model — metrics refresh continuously from the local dataset and
            checkpoint.
          </p>
          <div className="actions">
            <button className="btn primary" onClick={fetchData}>
              Refresh now
            </button>
            <button className="btn ghost">Export Snapshot</button>
          </div>
        </div>
        <div className="pill">
          <span className="pulse" />
          {status === 'ready' ? 'Live stream · 5s polling' : 'Connecting to model...'}
        </div>
      </header>

      {status === 'error' && (
        <div className="error">
          <strong>API error:</strong> {error} — ensure the backend is running (`python backend/server.py`).
        </div>
      )}

      <section className="grid stats">
        <article className="stat-card">
          <div className="stat-label">Realtime detections</div>
          <div className="stat-value">{formatNumber(summary.detections)}</div>
          <div className="stat-sub">Total labeled samples in dataset</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Overall accuracy</div>
          <div className="stat-value">
            {summary.accuracy?.toFixed(2) ?? '--'}
            <span className="suffix">%</span>
          </div>
          <div className="pill subtle">Evaluated on held-out set</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Latency p95</div>
          <div className="stat-value">
            {summary.latency_p95_ms?.toFixed(1) ?? '--'}
            <span className="suffix"> ms</span>
          </div>
          <div className={`stat-chip ${summary.latency_p95_ms > 95 ? 'warn' : 'ok'}`}>
            {summary.latency_p95_ms > 95 ? 'Investigate' : 'Stable'}
          </div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Uptime</div>
          <div className="stat-value">
            {summary.uptime?.toFixed(2) ?? '--'}
            <span className="suffix">%</span>
          </div>
          <div className="stat-sub">Placeholder until service uptime is wired</div>
        </article>
      </section>

      <section className="grid main">
        <article className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Signal trend</p>
              <h2>Detection volume</h2>
            </div>
            <div className="legend">
              <span className="dot primary" />
              <span>Live window</span>
            </div>
          </div>
          <div className="chart">
            <svg viewBox="0 0 320 140" role="img" aria-label="Detection sparkline">
              <defs>
                <linearGradient id="sparkFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="rgba(124,199,255,0.45)" />
                  <stop offset="100%" stopColor="rgba(124,199,255,0)" />
                </linearGradient>
              </defs>
              <path
                className="spark-area"
                d={`${buildSparkPath(trendPoints, 320, 120)} L320,140 L0,140 Z`}
                fill="url(#sparkFill)"
              />
              <path className="spark-line" d={buildSparkPath(trendPoints, 320, 120)} />
            </svg>
            <div className="chart-meta">
              <div>
                <p className="chart-label">Current window</p>
                <p className="chart-value">
                  {trendPoints.length ? formatNumber(trendPoints.at(-1)) : '—'} events/min
                </p>
              </div>
              <div>
                <p className="chart-label">Forecast</p>
                <p className="chart-value muted">
                  {summary.accuracy ? '+ uses live model outputs' : 'Awaiting backend'}
                </p>
              </div>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Quality</p>
              <h2>Model health</h2>
            </div>
            <div className="legend">
              <span className="dot soft" />
              <span>Target</span>
            </div>
          </div>
          <div className="health-grid">
            <div className="dial">
              <div
                className="dial-ring"
                style={{
                  background: `conic-gradient(#5CF0B5 ${(summary.accuracy || 0) * 3.6}deg, rgba(255,255,255,0.08) 0deg)`
                }}
              >
                <div className="dial-core">
                  <span>{summary.accuracy?.toFixed(1) ?? '--'}%</span>
                  <small>Accuracy</small>
                </div>
              </div>
              <p className="muted">Goal 95%</p>
            </div>
            <div className="health-list">
              <div className="row">
                <span>Latency p95</span>
                <div className="bar">
                  <span style={{ width: `${Math.min(summary.latency_p95_ms || 0, 120)}%` }} />
                </div>
                <span className="value">
                  {summary.latency_p95_ms?.toFixed(1) ?? '--'} ms
                </span>
              </div>
              <div className="row">
                <span>Uptime</span>
                <div className="bar">
                  <span style={{ width: `${summary.uptime ?? 0}%` }} />
                </div>
                <span className="value">{summary.uptime?.toFixed(2) ?? '--'}%</span>
              </div>
              <div className="pill subtle">
                {lastUpdated ? `Updated ${new Date(lastUpdated).toLocaleTimeString()}` : 'Waiting for first pull'}
              </div>
            </div>
          </div>
        </article>
      </section>

      <section className="grid split">
        <article className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Live detections</p>
              <h2>Top emotions</h2>
            </div>
            <div className="legend">
              <span className="dot primary" />
              <span>Confidence</span>
              <span className="dot soft" />
              <span>Latency</span>
            </div>
          </div>
          <div className="list">
            {emotions.map((item) => (
              <div key={item.name} className="list-row">
                <div className="identity">
                  <span className="chip" style={{ background: item.color }} />
                  <span>{item.name}</span>
                </div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: `${item.confidence}%` }} />
                </div>
                <span className="value">{item.confidence?.toFixed(1) ?? '--'}%</span>
                <div className="latency">
                  <span className="dot soft" />
                  {item.latency_ms?.toFixed(1) ?? '--'} ms
                </div>
              </div>
            ))}
            {!emotions.length && <p className="muted">Waiting for backend metrics…</p>}
          </div>
        </article>

        <article className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Narrative</p>
              <h2>Recent events</h2>
            </div>
            <div className="pill subtle">Autorefresh</div>
          </div>
          <div className="timeline">
            {events.map((event) => (
              <div key={event.title} className="event">
                <span className={`event-dot ${event.tone}`} />
                <div>
                  <p className="event-title">{event.title}</p>
                  <p className="muted">{event.meta}</p>
                </div>
              </div>
            ))}
            {!events.length && <p className="muted">No events yet — run the backend to populate.</p>}
          </div>
        </article>
      </section>
    </div>
  )
}

export default App
